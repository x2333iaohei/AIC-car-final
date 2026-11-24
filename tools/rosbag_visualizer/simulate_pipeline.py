#!/usr/bin/env python3
"""Full closed-pipeline replay of LineFollowerNode + DetectionTurnController + CmdMuxNode on ROS 2 bags."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from bag import decode_image
from panels import debug_overlay, text
from params import clamp, load_params
from vision import RedHistory, Vision

AI_MSGS_DEFS = """
std_msgs/Header header
int16 fps
ai_msgs/Perf[] perfs
ai_msgs/Target[] targets
ai_msgs/Target[] disappeared_targets
================================================================================
MSG: ai_msgs/Perf
string type
builtin_interfaces/Time stamp_start
builtin_interfaces/Time stamp_end
float64 time_ms_duration
================================================================================
MSG: ai_msgs/Target
string type
uint64 track_id
ai_msgs/Roi[] rois
ai_msgs/Attribute[] attributes
ai_msgs/Point[] points
ai_msgs/Capture[] captures
================================================================================
MSG: ai_msgs/Roi
string type
sensor_msgs/RegionOfInterest rect
float32 confidence
================================================================================
MSG: ai_msgs/Attribute
string type
float32 value
float32 confidence
================================================================================
MSG: ai_msgs/Point
string type
geometry_msgs/Point32[] point
float32[] confidence
================================================================================
MSG: ai_msgs/Capture
std_msgs/Header header
sensor_msgs/Image img
ai_msgs/Feature features
ai_msgs/DBResult db_result
================================================================================
MSG: ai_msgs/Feature
std_msgs/Header header
float32[] features
================================================================================
MSG: ai_msgs/DBResult
string db_type
string match_id
float32 distance
float32 similarity
"""


def build_store():
    store = get_typestore(Stores.ROS2_HUMBLE)
    store.register(get_types_from_msg(AI_MSGS_DEFS, "ai_msgs/msg/PerceptionTargets"))
    return store


def load_all_node_params(yaml_path: Path):
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    lf_vis = load_params(yaml_path, "/line_follower_node")
    lf_raw = dict(doc.get("line_follower_node", {}).get("ros__parameters", {}))
    lf_raw.update(lf_vis)
    det_raw = dict(doc.get("detection_turn_controller", {}).get("ros__parameters", {}))
    mux_raw = dict(doc.get("cmd_mux_node", {}).get("ros__parameters", {}))
    return lf_raw, det_raw, mux_raw


def parse_digit_label(label: str) -> int:
    if not label:
        return -1
    s = label[1:] if label[0] in ("d", "D") else label
    if not s or not s.isdigit():
        return -1
    val = int(s)
    return val if 1 <= val <= 8 else -1


class SimCmdMux:
    def __init__(self, p):
        self.stop_timeout = float(p.get("stop_timeout", 0.50))
        self.avoid_timeout = float(p.get("avoid_timeout", 0.12))
        self.high_timeout = float(p.get("high_timeout", 0.12))
        self.last_stop = None
        self.last_avoid = None
        self.last_high = None
        self.out_cmd = (0.0, 0.0)
        self.out_src = "IDLE"

    def on_stop(self, t, v, w):
        self.last_stop = (t, (v, w))
        self.out_cmd = (v, w)
        self.out_src = "STOP"

    def on_avoid(self, t, v, w):
        self.last_avoid = (t, (v, w))
        if self.last_stop and (t - self.last_stop[0]) <= self.stop_timeout:
            self.out_cmd, self.out_src = self.last_stop[1], "STOP"
        else:
            self.out_cmd, self.out_src = (v, w), "AVOID"

    def on_high(self, t, v, w):
        self.last_high = (t, (v, w))
        if self.last_stop and (t - self.last_stop[0]) <= self.stop_timeout:
            self.out_cmd, self.out_src = self.last_stop[1], "STOP"
        elif self.last_avoid and (t - self.last_avoid[0]) <= self.avoid_timeout:
            self.out_cmd, self.out_src = self.last_avoid[1], "AVOID"
        else:
            self.out_cmd, self.out_src = (v, w), "DETECT"

    def on_low(self, t, v, w, low_tag="FOLLOW"):
        if self.last_stop and (t - self.last_stop[0]) <= self.stop_timeout:
            self.out_cmd, self.out_src = self.last_stop[1], "STOP"
        elif self.last_avoid and (t - self.last_avoid[0]) <= self.avoid_timeout:
            self.out_cmd, self.out_src = self.last_avoid[1], "AVOID"
        elif self.last_high and (t - self.last_high[0]) <= self.high_timeout:
            self.out_cmd, self.out_src = self.last_high[1], "DETECT"
        else:
            self.out_cmd, self.out_src = (v, w), low_tag


class SimDetectionTurn:
    def __init__(self, p):
        self.conf_th = float(p.get("confidence_threshold", 0.55))
        self.w = int(p.get("det_frame_width", 640))
        self.h = int(p.get("det_frame_height", 480))
        self.window = max(1, int(p.get("frame_window", 3)))
        self.min_hits = max(1, int(p.get("min_candidate_frames", 2)))
        self.cooldown = float(p.get("turn_cooldown", 1.0))
        self.duration = float(p.get("turn_duration", 0.44))
        self.linear = float(p.get("turn_linear", 0.20))
        self.yaw_gain = float(p.get("yaw_gain", 1.65))
        self.max_yaw = float(p.get("max_yaw_rate", 1.00))
        self.y_offset = float(p.get("target_y_offset", -0.02))
        self.y_tol = float(p.get("y_align_tolerance", 0.025))
        self.dir_sign = float(p.get("turn_direction_sign", -1.0))
        self.centroid_timeout = float(p.get("centroid_timeout", 0.30))

        self.started = False
        self.yellow_stopped = False
        self.centroid = None
        self.turning = False
        self.turn_cmd = (0.0, 0.0)
        self.turn_digit = -1
        self.turn_end = -1.0
        self.cooldown_until = -1.0
        self.next_timer = -1.0
        self.clear_session()
        self.events = []
        self.last_rois = []

    def clear_session(self):
        self.session_active = False
        self.session_frames = 0
        self.best_digit = -1
        self.best_conf = 0.0
        self.best_xd = 0.0
        self.best_yd = 0.0

    def on_start(self, val: bool):
        if not val and self.started:
            self.clear_session()
            self.turning = False
        self.started = val

    def on_state(self, t: float, state: str):
        if state == "yellow_line_detected":
            self.yellow_stopped = True
            self.clear_session()
            self.turning = False
            self.cooldown_until = t
        else:
            self.yellow_stopped = False

    def on_centroid(self, t: float, nx: float, ny: float):
        self.centroid = (t, clamp(nx, 0.0, 1.0), clamp(ny, 0.0, 1.0))

    def step_timer(self, t: float, mux: SimCmdMux, lf_det_cb):
        if not self.turning:
            return
        while self.next_timer <= t:
            if not self.started or self.yellow_stopped or self.next_timer >= self.turn_end:
                self.turning = False
                return
            mux.on_high(self.next_timer, *self.turn_cmd)
            lf_det_cb(self.next_timer, self.turn_cmd[1])
            self.next_timer += 0.05

    def _trigger(self, t: float, mux: SimCmdMux, lf_det_cb):
        if self.best_digit < 0:
            return
        chosen = self.best_digit
        xd = self.best_xd
        yd = self.best_yd
        yaw = clamp(self.dir_sign * xd * self.yaw_gain, -self.max_yaw, self.max_yaw)
        if abs(yaw) < 1e-4:
            return
        self.turn_cmd = (self.linear, yaw)
        self.turn_digit = chosen
        self.turn_end = t + self.duration
        self.cooldown_until = self.turn_end + self.cooldown
        self.turning = True
        self.next_timer = t + 0.05
        mux.on_high(t, *self.turn_cmd)
        lf_det_cb(t, yaw)
        self.events.append({
            "t": round(t, 2), "digit": chosen,
            "conf": round(self.best_conf, 3), "x_diff": round(xd, 3),
            "y_diff": round(yd, 3), "ang": round(yaw, 3), "dur": self.duration,
        })

    def on_detection(self, t: float, msg, mux: SimCmdMux, lf_det_cb):
        self.last_rois = []
        for tgt in msg.targets:
            for roi in tgt.rois:
                d = parse_digit_label(roi.type)
                if d > 0 and roi.rect.width > 0 and roi.rect.height > 0:
                    self.last_rois.append((
                        t, d, float(roi.confidence),
                        int(roi.rect.x_offset), int(roi.rect.y_offset),
                        int(roi.rect.width), int(roi.rect.height),
                    ))
        if not self.started or self.yellow_stopped:
            return
        self.step_timer(t, mux, lf_det_cb)
        if self.turning or t < self.cooldown_until:
            return
        if self.centroid is None or (t - self.centroid[0]) > self.centroid_timeout:
            return

        if not self.session_active:
            self.clear_session()
            self.session_active = True

        _, cnx, cny = self.centroid
        target_y = clamp(cny + self.y_offset, 0.0, 1.0)
        candidates = []
        for _, d, cf, rx, ry, rw, rh in self.last_rois:
            if cf < self.conf_th:
                continue
            cx = clamp((rx + rw * 0.5) / self.w, 0.0, 1.0)
            cy = clamp((ry + rh * 0.5) / self.h, 0.0, 1.0)
            if cy < 0.48 or cy > 0.72:
                continue
            candidates.append((d, cf, cx, cy))

        if candidates:
            chosen = max(candidates, key=lambda item: (item[0], item[1]))
            d, cf, cx, cy = chosen
            yd = cy - target_y
            if abs(yd) <= self.y_tol:
                xd = cx - cnx
                if d > self.best_digit or (d == self.best_digit and cf > self.best_conf):
                    self.best_digit = d
                    self.best_conf = cf
                    self.best_xd = xd
                    self.best_yd = yd

        self.session_frames += 1
        if self.session_frames >= self.window:
            self._trigger(t, mux, lf_det_cb)
            self.clear_session()


class SimLineFollower:
    def __init__(self, p):
        self.p = p
        self.vision = Vision(p)
        self.red_hist = RedHistory(p)
        self.started = False
        self.yellow_latched = False
        self.yellow_clear = 0
        self.yellow_count = 0
        self.yellow_stop_active = False
        self.permanent_stop = False
        self.yellow_t = -1.0

        self.prev_err = None
        self.prev_err_t = -1.0
        self.filt_deriv = 0.0

        self.tj_branch_decided = False
        self.tj_armed = False
        self.tj_confirm = 0
        self.tj_merging = False
        self.tj_start_t = -1.0

        self.last_motion = (0.08, 0.0)
        self.last_det_t = -1.0
        self.last_state = "lost"
        self.events = []

    def on_start(self, t: float, val: bool):
        if not self.started and val and self.permanent_stop:
            self.yellow_count = 0
            self.yellow_stop_active = False
            self.yellow_latched = False
            self.yellow_clear = 0
            self.permanent_stop = False
            self.tj_branch_decided = False
            self.tj_armed = False
            self.tj_confirm = 0
            self.tj_merging = False
            self.prev_err = None
            self.filt_deriv = 0.0
        self.started = val
        if not val:
            self.vision.reset()
            self.red_hist.reset()
            self.prev_err = None
            self.filt_deriv = 0.0
            self.tj_confirm = 0
            self.tj_merging = False

    def on_det_cmd(self, t: float, ang_z: float):
        if not self.started or not self.p.get("enable_tjunction_merge", False):
            return
        req_yc = int(self.p.get("tjunction_branch_yellow_count", 2))
        if self.yellow_count != req_yc or self.tj_branch_decided or abs(ang_z) < 0.05:
            return
        self.tj_branch_decided = True
        trig_sign = int(self.p.get("tjunction_branch_trigger_sign", -1))
        if (trig_sign < 0 and ang_z < -0.05) or (trig_sign > 0 and ang_z > 0.05):
            self.tj_armed = True
            self.tj_confirm = 0
            self.events.append((round(t, 2), f"T_MERGE_ARMED(ang={ang_z:+.2f}, yellow={self.yellow_count})"))
        else:
            self.tj_armed = False
            self.events.append((round(t, 2), f"T_MERGE_OUTER_BRANCH(ang={ang_z:+.2f})"))

    def on_image(self, t: float, img: np.ndarray, det: SimDetectionTurn, mux: SimCmdMux):
        p = self.p
        res = self.vision.process(img, self.started)
        h, w = img.shape[:2]
        if res.target is not None:
            nx = res.target[0] / float(w)
            ny = (res.top + res.target[1]) / float(h)
            det.on_centroid(t, nx, ny)

        if not self.started:
            mux.on_stop(t, 0.0, 0.0)
            return res, (0.0, 0.0), "PAUSED"

        det.step_timer(t, mux, self.on_det_cmd)

        # Yellow latching
        if res.yellow_detected:
            self.yellow_clear = 0
            if not self.yellow_latched:
                self.yellow_latched = True
                self.yellow_count += 1
                self.yellow_t = t
                self.tj_branch_decided = False
                self.tj_confirm = 0
                self.tj_merging = False
                if self.yellow_count != int(p.get("tjunction_branch_yellow_count", 2)):
                    self.tj_armed = False
                if self.yellow_count >= int(p["yellow_permanent_count"]):
                    self.permanent_stop = True
                    self.yellow_stop_active = False
                    self.events.append((round(t, 2), f"YELLOW_PERMANENT_STOP({self.yellow_count})"))
                else:
                    self.yellow_stop_active = True
                    self.events.append((round(t, 2), f"YELLOW_STOP({self.yellow_count}/{p['yellow_permanent_count']})"))
        elif self.yellow_latched:
            self.yellow_clear += 1
            if self.yellow_clear >= int(p["yellow_clear_frames"]):
                self.yellow_latched = False
                self.yellow_clear = 0

        since_y = t - self.yellow_t if self.yellow_t >= 0 else 1e9
        if self.yellow_stop_active and since_y >= float(p["yellow_stop_duration"]):
            self.yellow_stop_active = False
        yellow_stop_now = self.permanent_stop or (self.yellow_stop_active and since_y < float(p["yellow_stop_duration"]))

        self.red_hist.update(res, True, yellow_stop_now)

        max_yaw = float(p["max_yaw_rate"])
        low_tag = "FOLLOW"
        pub_stop = False
        pub_avoid = False

        if yellow_stop_now:
            lin, yaw = 0.0, 0.0
            state = "yellow_line_detected"
            pub_stop = True
            self.prev_err = None
            self.filt_deriv = 0.0
        elif res.target is not None and res.error is not None:
            self.last_det_t = t
            state = "tracking"
            kd = max(0.0, float(p.get("kd", 0.0)))
            deriv_term = 0.0
            if self.prev_err is not None:
                dt = t - self.prev_err_t
                if 0.005 <= dt <= 0.25:
                    raw_d = (res.error - self.prev_err) / dt
                    self.filt_deriv = 0.55 * self.filt_deriv + 0.45 * raw_d
                    deriv_term = kd * self.filt_deriv
                else:
                    self.filt_deriv = 0.0
            self.prev_err = res.error
            self.prev_err_t = t

            yaw = clamp(-(float(p["kp"]) * res.error + deriv_term), -max_yaw, max_yaw)
            scale = 1.0 - float(p["speed_gain_damping"]) * min(1.0, abs(res.error))
            lin = clamp(float(p["target_speed"]) * scale, float(p["min_speed"]), float(p["max_speed"]))
            if p["enable_continuous_curve_assist"]:
                if res.reverse_curve:
                    lin = min(lin, float(p["reverse_curve_speed"]))
                elif res.curve_level >= float(p["sharp_curve_threshold"]):
                    lin = min(lin, float(p["sharp_curve_speed"]))
        else:
            self.prev_err = None
            self.filt_deriv = 0.0
            state = "lost" if (t - self.last_det_t) >= float(p["line_lost_stop_time"]) else self.last_state
            lin = float(p["line_lost_forward_speed"])
            yaw = clamp(self.last_motion[1] * float(p["line_lost_yaw_hold_factor"]), -max_yaw, max_yaw)

        # T-junction merge state machine
        if not yellow_stop_now and p.get("enable_tjunction_merge", False):
            gate_ok = (not p.get("tjunction_require_branch_gate", True)) or self.tj_armed
            if not self.tj_merging:
                if gate_ok and res.tjunction_candidate:
                    self.tj_confirm += 1
                    if self.tj_confirm >= int(p.get("tjunction_confirm_frames", 3)):
                        self.tj_merging = True
                        self.tj_armed = False
                        self.tj_confirm = 0
                        self.tj_start_t = t
                        self.events.append((round(t, 2), f"T_MERGE_START(rows={res.tjunction_full_rows})"))
                else:
                    self.tj_confirm = 0
            if self.tj_merging:
                elapsed = t - self.tj_start_t
                open_ahead = (
                    elapsed >= float(p.get("tjunction_min_turn_duration", 0.35))
                    and res.upper_road_ratio >= float(p.get("tjunction_exit_upper_road_ratio", 0.12))
                    and res.tjunction_full_rows < 20
                )
                if open_ahead or elapsed >= float(p.get("tjunction_max_turn_duration", 1.25)):
                    self.tj_merging = False
                    self.prev_err = None
                    self.filt_deriv = 0.0
                    self.events.append((round(t, 2), f"T_MERGE_EXIT(dt={elapsed:.2f}s, up={res.upper_road_ratio:.2f})"))
                else:
                    mdir = 1 if int(p.get("tjunction_merge_direction", -1)) >= 0 else -1
                    lin = float(p.get("tjunction_merge_speed", 0.16))
                    yaw = clamp(mdir * abs(float(p.get("tjunction_merge_yaw_rate", 1.15))), -max_yaw, max_yaw)
                    state = "tracking"
                    low_tag = "T_MERGE"

        # Soft red avoidance
        red = res.red
        if not yellow_stop_now and p["enable_soft_red_avoidance"] and red.risk is not None and red.box is not None:
            near_th = float(p.get("red_zone_near_ratio", p["red_near_ratio"]) if red.kind == "line" else p["red_near_ratio"])
            confirmed_active = (
                self.red_hist.votes >= int(p["red_vote_frames"])
                and red.bottom_ratio >= near_th
                and (red.in_corridor or red.very_close)
            )
            allow_merge = (not self.tj_merging) or red.very_close
            if confirmed_active and allow_merge:
                risk = float(red.risk)
                base_yaw = float(p["red_line_soft_yaw"] if red.kind == "line" else p["red_soft_yaw"])
                rep = base_yaw * (1.0 + float(p["red_repulsion_gain"]) * risk)
                if res.reverse_curve and red.kind != "line":
                    lim = float(p["red_soft_curve_yaw_limit"])
                    rep = min(rep, lim + risk * (max_yaw - lim))
                lk = 1.0 - (1.0 - float(p["red_lane_keep_min_factor"])) * risk
                if red.side < 0:
                    if yaw > 0:
                        yaw *= lk
                    yaw -= rep
                else:
                    if yaw < 0:
                        yaw *= lk
                    yaw += rep
                yaw = clamp(yaw, -max_yaw, max_yaw)
                r_base = float(p["red_line_speed"] if red.kind == "line" else p["red_soft_speed"])
                safe_v = max(float(p["red_min_forward_speed"]), r_base * (1.0 - 0.60 * risk))
                lin = max(float(p["red_min_forward_speed"]), min(lin, safe_v))
                if risk >= float(p.get("red_mux_preempt_risk", 0.28)):
                    pub_avoid = True

        if not yellow_stop_now and lin > 0:
            self.last_motion = (lin, yaw)

        det.on_state(t, state)
        self.last_state = state

        mux.on_low(t, lin, yaw, low_tag)
        if pub_avoid and not pub_stop:
            mux.on_avoid(t, lin, yaw)
        if pub_stop and lin == 0.0 and yaw == 0.0:
            mux.on_stop(t, 0.0, 0.0)
        return res, (lin, yaw), low_tag


def draw_sim_frame(img, res, lf_params, lf: SimLineFollower, det: SimDetectionTurn, mux: SimCmdMux, old_cmd, rel_t):
    canvas = debug_overlay(img, res, lf_params, annotations=True)
    h, w = canvas.shape[:2]

    # Draw recent YOLO ROIs
    for rt, d, cf, rx, ry, rw, rh in det.last_rois:
        if (rel_t - rt) <= 0.15:
            ok = cf >= det.conf_th
            col = (80, 255, 80) if ok else (130, 130, 130)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), col, 2 if ok else 1)
            cv2.putText(canvas, f"{d}:{cf:.2f}", (rx, max(16, ry - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)

    # Draw steering arrows at bottom center: Orange = OLD recorded cmd_vel, Green = NEW simulated cmd_vel
    bx, by = w // 2, h - 18
    if old_cmd is not None:
        ov, ow = old_cmd
        ang_o = clamp(ow / 1.5, -1.0, 1.0) * (math.pi * 0.42)
        ox = int(bx - 95 * math.sin(ang_o))
        oy = int(by - (25 + 65 * clamp(ov / 0.30, 0.0, 1.2)) * math.cos(ang_o))
        cv2.arrowedLine(canvas, (bx, by), (ox, oy), (0, 140, 255), 3, tipLength=0.22)

    nv, nw = mux.out_cmd
    ang_n = clamp(nw / 1.5, -1.0, 1.0) * (math.pi * 0.42)
    nx = int(bx - 110 * math.sin(ang_n))
    ny = int(by - (25 + 75 * clamp(nv / 0.30, 0.0, 1.2)) * math.cos(ang_n))
    src_col = {
        "STOP": (0, 230, 255),
        "AVOID": (255, 80, 255),
        "DETECT": (255, 220, 60),
        "T_MERGE": (0, 255, 180),
        "FOLLOW": (60, 255, 80),
    }.get(mux.out_src, (180, 180, 180))
    cv2.arrowedLine(canvas, (bx, by), (nx, ny), src_col, 4, tipLength=0.25)

    # Top banner comparing OLD vs NEW
    banner = np.zeros((62, w, 3), dtype=np.uint8)
    banner[:] = (24, 28, 34)
    old_str = f"({old_cmd[0]:+.2f}, {old_cmd[1]:+.2f})" if old_cmd else "(n/a)"
    t_flag = " [T-MERGE]" if lf.tj_merging else (" [T-ARMED]" if lf.tj_armed else "")
    d_flag = f" [DIGIT:{det.turn_digit}]" if det.turning else ""
    text(banner, f"t={rel_t:5.2f}s | Y={lf.yellow_count} | MUX={mux.out_src}{t_flag}{d_flag}",
         10, 24, w - 20, scale=0.56, color=src_col)
    text(banner, f"NEW cmd=({nv:+.2f} m/s, {nw:+.2f} rad/s)  vs  OLD bag cmd={old_str}",
         10, 50, w - 20, scale=0.52, color=(230, 235, 240))
    return np.vstack([banner, canvas])


def run_bag_sim(bag_path: Path, yaml_path: Path, out_mp4: Path, store):
    lf_p, det_p, mux_p = load_all_node_params(yaml_path)
    lf = SimLineFollower(lf_p)
    det = SimDetectionTurn(det_p)
    mux = SimCmdMux(mux_p)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    last_old_cmd = None
    has_start_topic = False

    stats = {
        "frames": 0, "active_frames": 0,
        "avoid_preempt_frames": 0, "red_zone_frames": 0, "cone_avoid_frames": 0,
        "tmerge_frames": 0, "max_yaw_diff": 0.0,
        "keyframes": {},
    }

    wanted = {
        "/image_raw", "/hobot_dnn_detection", "hobot_dnn_detection",
        "/line_follower/start_cmd", "line_follower/start_cmd",
        "/rosout", "/cmd_vel",
    }

    with Reader(bag_path) as r:
        t0 = r.start_time
        topic_names = {c.topic for c in r.connections}
        has_start_topic = bool(topic_names & {"/line_follower/start_cmd", "line_follower/start_cmd"})
        conns = [c for c in r.connections if c.topic in wanted]
        frame_idx = 0
        for conn, ns, raw in r.messages(connections=conns):
            t = (ns - t0) / 1e9
            msg = store.deserialize_cdr(raw, conn.msgtype)
            tp = conn.topic.lstrip("/")

            if tp == "line_follower/start_cmd":
                lf.on_start(t, bool(msg.data))
                det.on_start(bool(msg.data))
            elif tp == "rosout" and not has_start_topic:
                if "Start signal changed: false -> true" in msg.msg:
                    lf.on_start(t, True)
                    det.on_start(True)
                elif "Start signal changed: true -> false" in msg.msg:
                    lf.on_start(t, False)
                    det.on_start(False)
            elif tp == "cmd_vel":
                last_old_cmd = (float(msg.linear.x), float(msg.angular.z))
            elif tp == "hobot_dnn_detection":
                det.on_detection(t, msg, mux, lf.on_det_cmd)
            elif tp == "image_raw":
                frame_idx += 1
                stats["frames"] += 1
                img = decode_image(msg)
                res, _, low_tag = lf.on_image(t, img, det, mux)

                if lf.started and not lf.permanent_stop:
                    stats["active_frames"] += 1
                    if mux.out_src == "AVOID":
                        stats["avoid_preempt_frames"] += 1
                    if res.red.kind == "line":
                        stats["red_zone_frames"] += 1
                    if res.red.kind == "cone" and (res.red.risk or 0.0) > 0.0:
                        stats["cone_avoid_frames"] += 1
                    if lf.tj_merging:
                        stats["tmerge_frames"] += 1
                    if last_old_cmd is not None:
                        stats["max_yaw_diff"] = max(stats["max_yaw_diff"], abs(mux.out_cmd[1] - last_old_cmd[1]))

                # Write 15fps video (every 2nd 30fps frame)
                if frame_idx % 2 == 1:
                    vis = draw_sim_frame(img, res, lf_p, lf, det, mux, last_old_cmd, t)
                    if writer is None:
                        vh, vw = vis.shape[:2]
                        writer = cv2.VideoWriter(
                            str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (vw, vh)
                        )
                    writer.write(vis)

                    # Capture diagnostic keyframes at interesting moments
                    if lf.tj_merging and "tmerge" not in stats["keyframes"]:
                        stats["keyframes"]["tmerge"] = (round(t, 2), vis.copy())
                    if res.red.kind == "line" and (res.red.risk or 0.0) > 0.05 and "red_zone" not in stats["keyframes"]:
                        stats["keyframes"]["red_zone"] = (round(t, 2), vis.copy())
                    if mux.out_src == "AVOID" and det.turning and "avoid_preempt_det" not in stats["keyframes"]:
                        stats["keyframes"]["avoid_preempt_det"] = (round(t, 2), vis.copy())

    if writer is not None:
        writer.release()
    return lf, det, mux, stats


def main():
    AIC = Path(r"E:\desktop\AIC")
    RING_YAML = AIC / "ws_rdkx5_ring/src/line_follow_ws/src/line_follower/config/line_follower_params.yaml"
    FORK_YAML = AIC / "ws_rdkx5_fork/src/line_follow_ws/src/line_follower/config/line_follower_params.yaml"
    RING_BAGS = AIC / "rosbags_20261124/rosbags_20261124"
    OUT_DIR = AIC / "rosbag_visualizer/output"

    store = build_store()

    suite = [
        ("RING", "fork_01", RING_BAGS / "fork_01_20261124_155301", RING_YAML, OUT_DIR / "sim_ring/fork_01_ring_sim.mp4"),
        ("RING", "fork_02", RING_BAGS / "fork_02_20261124_155832", RING_YAML, OUT_DIR / "sim_ring/fork_02_ring_sim.mp4"),
        ("RING", "fork_03", RING_BAGS / "fork_03_20261124_160117", RING_YAML, OUT_DIR / "sim_ring/fork_03_ring_sim.mp4"),
        ("RING", "fork_04", RING_BAGS / "fork_04_20261124_160357", RING_YAML, OUT_DIR / "sim_ring/fork_04_ring_sim.mp4"),
        ("FORK", "fork_06", AIC / "fork_06_20261124_191654", FORK_YAML, OUT_DIR / "sim_fork/fork_06_fork_sim.mp4"),
        ("FORK", "fork_11", AIC / "fork_11_20261124_191858", FORK_YAML, OUT_DIR / "sim_fork/fork_11_fork_sim.mp4"),
        ("FORK", "fork_12", AIC / "fork_12_20261124_193903", FORK_YAML, OUT_DIR / "sim_fork/fork_12_fork_sim.mp4"),
        ("FORK", "fork_13", AIC / "fork_13_20261124_194108", FORK_YAML, OUT_DIR / "sim_fork/fork_13_fork_sim.mp4"),
    ]

    snap_dir = OUT_DIR / "sim_keyframes"
    snap_dir.mkdir(parents=True, exist_ok=True)

    for map_tag, name, bpath, ypath, mp4 in suite:
        lf, det, mux, st = run_bag_sim(bpath, ypath, mp4, store)
        for k, (kt, kimg) in st["keyframes"].items():
            cv2.imwrite(str(snap_dir / f"{name}_{k}_{kt:.2f}s.png"), kimg)
        print(f"[{map_tag}] {name} -> MP4={mp4.name}")
        print(f"  Stats: active={st['active_frames']}f, red_zone={st['red_zone_frames']}f, "
              f"cone_avoid={st['cone_avoid_frames']}f, mux_avoid={st['avoid_preempt_frames']}f, "
              f"tmerge={st['tmerge_frames']}f")
        print(f"  LF events : {lf.events}")
        print(f"  DET turns : {det.events}", flush=True)


if __name__ == "__main__":
    main()

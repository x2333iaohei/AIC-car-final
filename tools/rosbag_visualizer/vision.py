"""OpenCV port of the visual decisions in LineFollowerNode::process_image/detect_red."""

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

from params import clamp


def band_run(binary, y0, y1, reference):
    y0 = clamp(y0, 0, binary.shape[0] - 1)
    y1 = clamp(y1, y0 + 1, binary.shape[0])
    band = binary[y0:y1]
    active = np.count_nonzero(band, axis=0) >= max(1, int(len(band) * .30))
    edges = np.flatnonzero(np.diff(active.astype(np.int8), prepend=0, append=0))
    center, best_distance, best_width = None, math.inf, 0
    for low, exclusive_end in edges.reshape(-1, 2):
        width = exclusive_end - low
        if width < 20:
            continue
        high = exclusive_end - 1
        candidate = (low + high) * .5
        distance = 0. if low <= reference <= high else abs(candidate - reference)
        if (distance < best_distance or
                (abs(distance - best_distance) < 1e-6 and width > best_width)):
            center, best_distance, best_width = float(candidate), distance, int(width)
    return center, best_width


def band_center(binary, y0, y1, reference):
    return band_run(binary, y0, y1, reference)[0]


def external_contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def rotate_divider(target, base, degrees):
    x, y = target[0] - base[0], target[1] - base[1]
    angle = math.radians(degrees if x >= 0. else -degrees)
    c, s = math.cos(angle), math.sin(angle)
    return base[0] + x*c - y*s, base[1] + x*s + y*c


@dataclass
class Red:
    mask: Optional[np.ndarray] = None
    kind: str = ""
    box: Optional[tuple] = None
    line_box: Optional[tuple] = None
    center: Optional[tuple] = None
    bottom: Optional[tuple] = None
    side: Optional[int] = None
    area_ratio: Optional[float] = None
    divider: Optional[tuple] = None
    status: str = ""
    bottom_ratio: Optional[float] = None
    in_corridor: Optional[bool] = None
    risk: Optional[float] = None
    very_close: bool = False


@dataclass
class Frame:
    top: int
    bottom: int
    roi: np.ndarray
    blur: np.ndarray
    hsv: np.ndarray
    road: np.ndarray
    opened: np.ndarray
    closed: np.ndarray
    yellow: Optional[np.ndarray]
    yellow_ratio: Optional[float]
    yellow_detected: bool
    contours: tuple
    box: Optional[tuple]
    near_band: tuple
    far_band: tuple
    near: Optional[float]
    far: Optional[float]
    near_error: float
    far_error: float
    curve_level: float
    reverse_candidate: bool
    reverse_curve: bool
    confirm_frames: int
    clear_frames: int
    target: Optional[tuple]
    target_source: str
    far_weight: Optional[float]
    error: Optional[float]
    red: Red
    tjunction_candidate: bool = False
    upper_road_ratio: float = 1.0
    tjunction_full_rows: int = 0


class RedHistory:
    """Rebuild red target votes and the debug fields produced before overlay publication."""

    def __init__(self, params):
        self.p = params
        self.reset()

    def reset(self):
        self.side, self.votes = None, 0

    def update(self, result, started, yellow_stop):
        if started is not True:
            return
        red, p = result.red, self.p
        soft = p["enable_soft_red_avoidance"]
        if red.box is None:
            if red.mask is not None or soft:
                self.reset()
            return

        self.votes = self.votes + 1 if red.side == self.side else 1
        self.side = red.side
        confirmed = self.votes >= p["red_vote_frames"]
        red.in_corridor, red.risk = False, 0.
        h, w = result.roi.shape[:2]
        red.bottom_ratio = clamp(red.bottom[1] / h, 0., 1.)
        if not soft:
            if confirmed:
                self.reset()
            return

        bx, by = w*.5, h
        dx, dy = red.divider[0]-bx, red.divider[1]-by
        rx, ry = red.bottom[0]-bx, red.bottom[1]-by
        length = math.hypot(dx, dy)
        lateral = abs(dx*ry-dy*rx)/length if length > 1e-6 else 1e9
        red.in_corridor = lateral <= clamp(p["red_corridor_half_width_ratio"], 0., 1.)*w
        near = clamp(
            p.get("red_zone_near_ratio", p["red_near_ratio"]) if red.kind == "line" else p["red_near_ratio"],
            0., 1.,
        )
        very_close = clamp(p["red_very_close_ratio"], 0., 1.)
        if (not yellow_stop and confirmed and red.bottom_ratio >= near and
                (red.in_corridor or red.bottom_ratio >= very_close)):
            red.risk = clamp((red.bottom_ratio-near)/max(.001, very_close-near), 0., 1.)
            red.very_close = red.bottom_ratio >= very_close


class Vision:
    def __init__(self, params):
        self.p = params
        size = params["morph_kernel"]
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        self.red_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kw = max(5, int(params.get("red_zone_close_kernel_w", 35)))
        kh = max(3, int(params.get("red_zone_close_kernel_h", 13)))
        self.zone_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        self.gaussian_size = params["gaussian_kernel"]
        self.gaussian_kernel = cv2.getGaussianKernel(self.gaussian_size, 0, cv2.CV_32F)
        self.reset()

    def reset(self):
        self.reverse = False
        self.confirm = self.clear = 0

    def _curve(self, near, far, center):
        p = self.p
        n = clamp((near - center) / center, -1., 1.) if near is not None else 0.
        f = clamp((far - center) / center, -1., 1.) if far is not None else 0.
        enabled = p["enable_continuous_curve_assist"]
        candidate = (enabled and n*f < 0. and
                     min(abs(n), abs(f)) >= max(0., p["reverse_curve_min_error"]) and
                     abs(f - n) >= max(0., p["reverse_curve_threshold"]))
        if not enabled:
            self.reset()
        elif candidate:
            self.clear = 0
            self.confirm += 1
            if not self.reverse and self.confirm >= p["reverse_curve_confirm_frames"]:
                self.reverse = True
                self.confirm = p["reverse_curve_confirm_frames"]
        else:
            self.confirm = 0
            self.clear = self.clear + 1 if self.reverse else 0
            if self.clear >= p["reverse_curve_clear_frames"]:
                self.reverse = False
                self.clear = 0
        return n, f, candidate

    def _target(self, near, far, near_w, far_w, width, near_band, far_band, n, f, candidate, fallback):
        ny, fy = sum(near_band)*.5, sum(far_band)*.5
        if near is None and far is None:
            return fallback, "contour" if fallback else "none", None
        if far is None:
            return (near, ny), "near", 0.
        if near is None:
            return (far, fy), "far", 1.

        p = self.p
        weight = .30
        if self.reverse:
            if not p["enable_curve_preview_steering"] or n*f < 0.:
                return (near, ny), "near_reverse_curve", 0.
        if p["enable_curve_preview_steering"]:
            cap = clamp(p["curve_preview_max_far_weight"], .30, .65)
            if n*f < 0.:
                weight = cap * (1. - clamp(abs(n) / .25, 0., 1.))
                if candidate:
                    weight *= 1. - clamp(self.confirm / p["reverse_curve_confirm_frames"], 0., 1.)
            else:
                weight += (cap - .30) * clamp(max(0., abs(f) - abs(n)) / .20, 0., 1.)
                if near_w >= int(.90 * width) and far_w < int(.80 * width):
                    weight = max(weight, cap)
        return ((1.-weight)*near + weight*far, .70*ny + .30*fy), "dual", weight

    def _tjunction(self, closed, yellow, yellow_detected):
        p = self.p
        h, w = closed.shape
        if not p.get("enable_tjunction_merge", False) or h < 40 or w < 100:
            return False, 1.0, 0
        bridged = cv2.bitwise_or(closed, yellow) if yellow is not None else closed
        _, labels = cv2.connectedComponents(bridged, connectivity=8)
        sy0 = clamp(int(h * .78), 0, h - 1)
        sx0 = clamp(int(w * .15), 0, w - 1)
        sx1 = clamp(int(w * .85), sx0 + 1, w)
        seed_labels = set(np.unique(labels[sy0:, sx0:sx1])) - {0}
        if not seed_labels:
            return False, 0.0, 0
        keep = np.isin(labels, list(seed_labels))
        road = np.where(keep & (closed > 0), np.uint8(255), np.uint8(0))

        up_h = max(1, int(h * .35))
        up_patch = road[:up_h, sx0:sx1]
        upper_ratio = cv2.countNonZero(up_patch) / float(max(1, up_patch.size))
        if yellow_detected:
            return False, upper_ratio, 0

        slices = ((20, 120), (140, 240), (270, 370), (400, 500), (520, 620))
        y_tops = []
        for s0, s1 in slices:
            x0 = clamp(int(round(s0 * w / 640.)), 0, w - 2)
            x1 = clamp(int(round(s1 * w / 640.)), x0 + 1, w)
            min_w = max(5, int((x1 - x0) * .50))
            idx = np.where(np.count_nonzero(road[:, x0:x1], axis=1) >= min_w)[0]
            if len(idx) == 0:
                return False, upper_ratio, 0
            y_tops.append(int(idx[0]))

        min_yt, max_yt = min(y_tops), max(y_tops)
        spread = max_yt - min_yt
        y_start = min(h - 1, max_yt + 4)
        y_end = clamp(int(h * .94), y_start + 1, h)
        sub = road[y_start:y_end, :]
        edge_w = max(4, int(round(24. * w / 640.)))
        edge_min = max(3, int(round(18. * w / 640.)))
        rw = np.count_nonzero(sub, axis=1)
        lok = np.count_nonzero(sub[:, :edge_w], axis=1) >= edge_min
        rok = np.count_nonzero(sub[:, w - edge_w:], axis=1) >= edge_min
        full_rows = int(np.sum((rw >= int(round(.94 * w))) & lok & rok))

        cand = (
            min_yt >= int(p.get("tjunction_wall_min_y_ratio", .23) * h)
            and max_yt <= int(p.get("tjunction_wall_max_y_ratio", .62) * h)
            and spread <= int(p.get("tjunction_wall_max_spread_ratio", .10) * h)
            and (y_end - y_start) >= 45
            and full_rows >= int(p.get("tjunction_min_full_rows", 55))
        )
        return cand, upper_ratio, full_rows

    def _red(self, roi, road_mask, target, started):
        p = self.p
        if not p["enable_red_filter"]:
            return Red(status="disabled by YAML")
        if started is False:
            return Red(status="paused")
        if target is None:
            return Red(status="no road target")
        status = "computed; start unknown" if started is None else "reconstructed"
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        masks = [cv2.inRange(hsv, (p[f"red_h_low_{i}"], p["red_s_min"], p["red_v_min"]),
                             (p[f"red_h_high_{i}"], 255, 255)) for i in (1, 2)]
        mask = cv2.bitwise_or(*masks)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.red_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.red_kernel)
        area_total = mask.size
        height, width = mask.shape
        line, cone = None, None
        for contour in external_contours(mask):
            area = cv2.contourArea(contour)
            if area <= 0.:
                continue
            box = cv2.boundingRect(contour)
            w, h = box[2:]
            if (w / h >= max(1., p["red_line_aspect_ratio_min"]) and
                    w >= width * clamp(p["red_line_min_width_ratio"], 0., 1.)):
                if line is None or area > line[0]:
                    line = (area, box)
            elif (w >= p["red_cone_min_bbox_width"] and h >= p["red_cone_min_bbox_height"] and
                  area / area_total >= max(0., p["red_cone_area_ratio_threshold"])):
                if cone is None or area > cone[0]:
                    cone = (area, box)

        if p.get("enable_red_zone_cluster", False) and road_mask is not None:
            zs = min(70, p["red_s_min"])
            zv = min(60, p["red_v_min"])
            zm1 = cv2.inRange(hsv, (0, zs, zv), (12, 255, 255))
            zm2 = cv2.inRange(hsv, (158, zs, zv), (180, 255, 255))
            raw_zone = cv2.bitwise_or(zm1, zm2)
            num_l, labels, stats, _ = cv2.connectedComponentsWithStats(raw_zone, connectivity=8)
            stripe_boxes = []
            clean_raw = np.zeros_like(raw_zone)
            for idx in range(1, num_l):
                if stats[idx, cv2.CC_STAT_AREA] >= 6:
                    clean_raw[labels == idx] = 255
                    stripe_boxes.append((
                        int(stats[idx, cv2.CC_STAT_LEFT]),
                        int(stats[idx, cv2.CC_STAT_TOP]),
                        int(stats[idx, cv2.CC_STAT_WIDTH]),
                        int(stats[idx, cv2.CC_STAT_HEIGHT]),
                    ))
            if len(stripe_boxes) >= 2:
                clustered = cv2.morphologyEx(clean_raw, cv2.MORPH_CLOSE, self.zone_kernel)
                min_wr = clamp(p.get("red_zone_min_width_ratio", .11), .05, .90)
                min_ar = max(.0005, p.get("red_zone_min_area_ratio", .0030))
                max_fr = clamp(p.get("red_zone_max_fill_ratio", .30), .08, .60)
                for zc in external_contours(clustered):
                    zx, zy, zw, zh = cv2.boundingRect(zc)
                    if zw < int(width * min_wr) or zh < 18 or (zw / float(zh)) < 1.30:
                        continue
                    b_area = float(zw * zh)
                    if (b_area / area_total) < min_ar:
                        continue
                    raw_fill = cv2.countNonZero(clean_raw[zy:zy+zh, zx:zx+zw]) / max(1., b_area)
                    if not (.025 <= raw_fill <= max_fr):
                        continue
                    n_in = sum(
                        1 for sx, sy, sw, sh in stripe_boxes
                        if sx >= zx - 2 and sy >= zy - 2 and sx + sw <= zx + zw + 2 and sy + sh <= zy + zh + 2
                    )
                    if n_in < 2:
                        continue
                    px0, py0 = max(0, zx - 12), max(0, zy - 8)
                    px1, py1 = min(width, zx + zw + 12), min(height, zy + zh + 8)
                    patch = road_mask[py0:py1, px0:px1]
                    if patch.size > 0 and (cv2.countNonZero(patch) / float(patch.size)) < .50:
                        continue
                    if line is None or b_area > line[0]:
                        line = (b_area, (zx, zy, zw, zh))
                if line is not None and cone is not None:
                    lx, ly, lw, lh = line[1]
                    cx, cy, cw, ch = cone[1]
                    if cx >= lx - 6 and cy >= ly - 6 and cx + cw <= lx + lw + 6 and cy + ch <= ly + lh + 6:
                        cone = None

        if line is not None and cone is not None:
            cone_bot = cone[1][1] + cone[1][3]
            line_bot = line[1][1] + line[1][3]
            use_line = line_bot >= cone_bot - 15
        else:
            use_line = (line is not None) and (cone is None)
        selected = line if use_line else cone
        if selected is None:
            return Red(mask=mask, status=status)

        area, box = selected
        x, y, w, h = box
        center = (x + w*.5, y + h*.5)
        bottom = (center[0], y+h)
        base = (width*.5, float(height))
        divider = rotate_divider(target, base, p["red_divider_angle_deg"])
        if not use_line:
            cross = ((divider[0] - base[0])*(center[1] - base[1]) -
                     (divider[1] - base[1])*(center[0] - base[0]))
            side = 1 if cross >= 0. else -1
        else:
            if road_mask is not None and p.get("enable_red_zone_cluster", False):
                by0 = clamp(y - 10, 0, height - 1)
                by1 = clamp(y + h + 10, by0 + 1, height)
                bh_band = max(1, by1 - by0)
                left_space = cv2.countNonZero(road_mask[by0:by1, :x]) / float(bh_band) if x > 0 else 0.
                rx0 = clamp(x + w, 0, width)
                right_space = cv2.countNonZero(road_mask[by0:by1, rx0:]) / float(bh_band) if rx0 < width else 0.
                if right_space > left_space + 12.:
                    side = -1
                elif left_space > right_space + 12.:
                    side = 1
                else:
                    side = 1 if p["red_line_fallback_direction"] >= 0 else -1
                bottom = (float(min(width - 1, x + w) if side < 0 else max(0, x)), float(y + h))
            else:
                remaining = width - x - w
                if remaining > x + 10:
                    side = -1
                elif x > remaining + 10:
                    side = 1
                else:
                    side = 1 if p["red_line_fallback_direction"] >= 0 else -1
        return Red(mask, "line" if use_line else "cone", box, line[1] if line else None,
                   center, bottom, side, area/area_total, divider, status)

    def process(self, bgr, started):
        """Process one source image. False resets curve history after computing its target."""
        p = self.p
        height, width = bgr.shape[:2]
        top = clamp(int(height * (1. - p["roi_top_ratio"])), 0, height - 1)
        bottom = clamp(height - int(height * p["roi_bottom_ratio"]), top + 1, height)
        roi = bgr[top:bottom]
        g = self.gaussian_size
        if top == 0 and bottom == height:
            blur = cv2.GaussianBlur(bgr, (g, g), 0)
        elif g == 1:
            blur = roi.copy()
        else:
            pad_top, pad_end = max(0, top - g//2), min(height, bottom + g//2)
            padded = cv2.sepFilter2D(bgr[pad_top:pad_end], -1, self.gaussian_kernel,
                                     self.gaussian_kernel, borderType=cv2.BORDER_DEFAULT)
            blur = padded[top-pad_top:bottom-pad_top]
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        yellow, ratio, detected = None, None, False
        if p["use_yellow_filter"]:
            yellow = cv2.inRange(hsv, (p["yellow_h_low"], p["yellow_s_min"], p["yellow_v_min"]),
                                 (p["yellow_h_high"], 255, 255))
            ratio = cv2.countNonZero(yellow) / yellow.size
            detected = ratio >= p["yellow_area_ratio_threshold"]
        if p["use_white_filter"]:
            road = cv2.inRange(hsv, (p["white_h_low"], 0, p["white_v_min"]),
                               (p["white_h_high"], p["white_s_max"], 255))
        else:
            mode = cv2.THRESH_BINARY_INV if p["invert_binary"] else cv2.THRESH_BINARY
            _, road = cv2.threshold(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY),
                                    p["binary_threshold"], 255, mode)
        opened = cv2.morphologyEx(road, cv2.MORPH_OPEN, self.kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel,
                                  iterations=p["morph_iterations"])

        contours = external_contours(closed)
        box, fallback = None, None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= p["min_contour_area"]:
                box = cv2.boundingRect(largest)
                moments = cv2.moments(largest)
                if moments["m00"]:
                    x, y = (moments[key]/moments["m00"] for key in ("m10", "m01"))
                else:
                    x, y = box[0]+box[2]/2., box[1]+box[3]/2.
                fallback = float(np.float32(x)), float(np.float32(y))

        h = bottom - top
        near_band = int(h * .62), int(h * .88)
        far_band = int(h * .34), int(h * .56)
        center = width*.5
        near, near_w = band_run(closed, *near_band, center)
        far, far_w = band_run(closed, *far_band, center if near is None else near)
        n, f, candidate = self._curve(near, far, center)
        target, source, weight = self._target(
            near, far, near_w, far_w, width, near_band, far_band, n, f, candidate, fallback
        )
        error = clamp((target[0] - center)/center, -1., 1.) if target else None
        if target:
            target = tuple(float(np.float32(value)) for value in target)
        tj_cand, up_ratio, tj_rows = self._tjunction(closed, yellow, detected)
        if started is False:
            self.reset()
            n, f = 0., 0.
        return Frame(top, bottom, roi, blur, hsv, road, opened, closed, yellow, ratio, detected,
                     contours, box, near_band, far_band, near, far, n, f,
                     max(abs(n), abs(f)), candidate, self.reverse, self.confirm, self.clear,
                     target, source, weight, error, self._red(roi, closed, target, started),
                     tj_cand, up_ratio, tj_rows)

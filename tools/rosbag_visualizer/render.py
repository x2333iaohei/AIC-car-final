#!/usr/bin/env python3
"""Render a ROS 2 recording to an annotated offline MP4 and analysis files."""

import argparse
from collections import Counter
import csv
import heapq
import hashlib
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import yaml

from bag import Bag, decode_image, header_ns, telemetry_value
from panels import Dashboard, FocusDashboard, VideoClock, speed_chart
from params import load_params
from telemetry import Second, StampClock, Timeline, YellowLatch, braking
from vision import RedHistory, Vision


TOPICS = {
    "image": "/image_raw", "odom": "/odom", "cmd": "/cmd_vel",
    "state": "/line_follower/track_state", "start": "/line_follower/start_cmd",
    "centroid": "/line_follower/centroid", "log": "/rosout",
}
DEFAULT_DEBUG_IMAGE_TOPIC = "/line_follower/debug_image"
LOCAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = LOCAL_DIR / "output"
DEFAULT_LOCAL_PARAMS = (
    LOCAL_DIR.parents[1]
    / "src"
    / "line_follow_ws"
    / "src"
    / "line_follower"
    / "config"
    / "line_follower_params.yaml"
)

FRAME_FIELDS = (
    "frame", "segment", "time_ns", "bag_time_ns", "image_clock", "relative_s", "source_fps",
    "image_width_px", "image_height_px", "roi_top_px", "roi_bottom_px",
    "odom_v", "odom_w", "odom_time_ns", "odom_clock", "cmd_v", "cmd_w", "cmd_time_ns",
    "track_state", "state_age_s", "started", "recorded_centroid_norm_x", "recorded_centroid_norm_y",
    "reconstructed_target_x_px", "reconstructed_target_y_px", "target_source",
    "near_x_px", "far_x_px", "near_error", "far_error", "curve_level",
    "reverse_curve_candidate", "reverse_curve", "far_weight", "yellow_area_ratio",
    "yellow_detected", "reconstructed_yellow_count", "red_kind",
    "red_bottom_ratio", "red_in_corridor", "red_risk_reconstructed",
)

EVENT_FIELDS = (
    "time_ns", "relative_s", "source", "time_clock", "event", "yellow_count", "yellow_total",
    "brake_cmd_ns", "odom_stop_ns", "trigger_to_brake_cmd_s", "brake_duration_s", "odom_signed_m",
    "odom_forward_m", "metric_status", "detail",
)


def dimensions(raw):
    try:
        first, second = map(int, raw.lower().split("x"))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("Expected NxM") from None
    if not 1 <= first <= 4096 or not 1 <= second <= 4096:
        raise argparse.ArgumentTypeError("Dimensions must be in 1–4096")
    return first, second


def parse_layout(raw):
    return "focus" if raw.lower() == "focus" else dimensions(raw)


def bounded_number(low, high):
    def parse(raw):
        value = float(raw)
        if not math.isfinite(value) or not low <= value <= high:
            raise argparse.ArgumentTypeError(f"Expected a value in {low}–{high}")
        return value

    return parse


def resolve_params_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if DEFAULT_LOCAL_PARAMS.is_file():
        return DEFAULT_LOCAL_PARAMS
    raise ValueError(
        "Missing --params YAML and default local config not found at "
        f"{DEFAULT_LOCAL_PARAMS}"
    )


def resolve_output_path(explicit: Path | None, bag_path: Path, layout):
    if explicit is not None:
        if explicit.suffix.lower() != ".mp4":
            raise ValueError("--output must end in .mp4")
        output = explicit.resolve()
        sidecars = output.with_name(output.stem+"_analysis")
        if output.exists() or sidecars.exists():
            raise FileExistsError("Choose a new output path; the MP4 or analysis folder exists.")
        return output, sidecars
    bag_name = bag_path.stem if bag_path.is_file() else bag_path.name
    bag_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in bag_name).strip("._") or "rosbag"
    layout_tag = "focus" if layout == "focus" else f"{layout[0]}x{layout[1]}"
    stem = f"{bag_name}_{layout_tag}"
    candidate = (DEFAULT_OUTPUT_DIR / f"{stem}.mp4").resolve()
    sidecars = candidate.with_name(candidate.stem+"_analysis")
    index = 1
    while candidate.exists() or sidecars.exists():
        candidate = (DEFAULT_OUTPUT_DIR / f"{stem}_{index:03d}.mp4").resolve()
        sidecars = candidate.with_name(candidate.stem+"_analysis")
        index += 1
    return candidate, sidecars


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bag", required=True, type=Path)
    cli.add_argument("--params", type=Path, default=None,
                     help=f"ROS parameter YAML (default: {DEFAULT_LOCAL_PARAMS})")
    cli.add_argument("--output", type=Path, default=None,
                     help=f"MP4 path (default: {DEFAULT_OUTPUT_DIR}\\<bag>_<layout>.mp4); "
                          "sidecars in <stem>_analysis")
    cli.add_argument("--layout", type=parse_layout, default="focus",
                     help="focus (default), or rows x columns for detailed panels")
    cli.add_argument("--resolution", type=dimensions, default=(1280, 800), help="Width x height")
    cli.add_argument("--fps", type=bounded_number(.1, 120), default=15.)
    cli.add_argument("--backend", choices=("auto", "rosbag2_py", "rosbags"), default="auto")
    cli.add_argument("--storage", choices=("auto", "sqlite3", "mcap"), default="auto")
    cli.add_argument("--node", default="/line_follower_node", help="YAML node and ROS logger name")
    cli.add_argument("--clock", choices=("auto", "bag"), default="auto")
    cli.add_argument("--max-header-skew", type=bounded_number(0., 60.), default=2.)
    cli.add_argument("--sync-max-age", type=bounded_number(.001, 10.), default=.25)
    cli.add_argument("--frame-gap", type=bounded_number(.02, 10.), default=.5)
    cli.add_argument("--odom-max-gap", type=bounded_number(.001, 3.), default=.20)
    cli.add_argument("--stop-speed", type=bounded_number(.001, 1.), default=.02)
    cli.add_argument("--stop-yaw", type=bounded_number(.001, 2.), default=.05)
    cli.add_argument("--stop-dwell", type=bounded_number(.01, 3.), default=.20)
    cli.add_argument("--keyframes", type=int, default=48, help="Maximum annotated PNGs; 0 disables")
    cli.add_argument("--debug-image-topic", default=DEFAULT_DEBUG_IMAGE_TOPIC,
                     help="Optional recorded debug image topic; offline overlay is reconstructed when absent")
    cli.add_argument("--overlay-source", choices=("auto", "reconstructed", "recorded"), default="auto",
                     help="auto uses recorded debug_image when present and falls back to full offline overlay")
    for role, topic in TOPICS.items():
        cli.add_argument(f"--{role}-topic", default=topic)
    return cli


def index_telemetry(bag, db, args, clock, topics):
    inputs = {topic: role for role, topic in topics.items() if role not in ("image", "debug_image")}
    counts = Counter()
    for topic, record, msg in bag.messages(set(inputs)):
        role = inputs[topic]
        try:
            value = telemetry_value(role, msg, args.node)
        except (AttributeError, ValueError, TypeError):
            counts[f"invalid_{role}"] += 1
            continue
        if value is None:
            continue
        if any(isinstance(x, float) and not math.isfinite(x) for x in value):
            counts[f"invalid_{role}"] += 1
            continue
        ns, source = clock.choose(record, header_ns(msg))
        db.add(role, ns, value, source)
        counts[role] += 1
        if role == "log":
            if value[0] == "yellow_trigger":
                db.add("yellow_log", ns, value, source)
            elif topics["start"] not in bag.seen and value[0] in ("start_true", "start_false", "start_waiting"):
                db.add("start", ns, (value[0] == "start_true",), source)
        if sum(counts.values()) % 10000 == 0:
            db.flush()
    db.flush()
    if topics["start"] not in bag.seen and not db.has("start") and db.has("state"):
        first_state = next(db.range("state"), None)
        if first_state is not None:
            db.add("start", first_state.ns, (True,), first_state.clock)
            db.flush()
    return counts


def shown(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def sample_value(sample, index):
    return None if sample is None else sample.value[index]


def frame_data(db, ts, origin, segment, frame_no, raw_ns, img_clock, fps, vision, latch, age, shape):
    odom = db.latest("odom", ts, age)
    cmd = db.latest("cmd", ts, age)
    # Controller publishes the tracking state on transitions. Hold the last observation
    # with its age, and display that age explicitly instead of implying periodic data.
    state = db.latest("state", ts)
    centroid = db.latest("centroid", ts, age)
    target = vision.target
    return dict(zip(FRAME_FIELDS, (
        frame_no, segment, ts, raw_ns, img_clock, (ts-origin)/Second, fps,
        shape[1], shape[0], vision.top, vision.bottom,
        sample_value(odom, 0), sample_value(odom, 1), odom.ns if odom else None,
        odom.clock if odom else None, sample_value(cmd, 0), sample_value(cmd, 1),
        cmd.ns if cmd else None, sample_value(state, 0),
        (ts-state.ns)/Second if state else None, latch.started,
        sample_value(centroid, 0), sample_value(centroid, 1), target[0] if target else None,
        target[1]+vision.top if target else None, vision.target_source, vision.near, vision.far,
        vision.near_error if vision.near is not None and latch.started is not False else None,
        vision.far_error if vision.far is not None and latch.started is not False else None,
        vision.curve_level, vision.reverse_candidate, vision.reverse_curve, vision.far_weight,
        vision.yellow_ratio, vision.yellow_detected, latch.count if latch.started is not None else None,
        vision.red.kind, vision.red.bottom_ratio, vision.red.in_corridor, vision.red.risk,
    )))


def footer(data, result, event):
    def val(key, digits=2):
        return shown(data[key], digits)

    clock = data["image_clock"]
    mode = "REVERSE" if result.reverse_curve else "NORMAL"
    return [
        f"OFFLINE RECONSTRUCTION  |  frame {data['frame']:06}  |  segment {data['segment']}  |  "
        f"t={val('relative_s', 3)} s  |  {clock}={data['time_ns']} ns  |  input {val('source_fps', 1)} FPS",
        f"RECORDED  |  odom v={val('odom_v')} m/s, w={val('odom_w')} rad/s  |  "
        f"cmd v={val('cmd_v')} m/s, w={val('cmd_w')} rad/s  |  start={data['started']}",
        f"RECORDED  |  track={data['track_state'] or 'n/a'} (age {val('state_age_s')} s)  |  "
        f"centroid normalized=({val('recorded_centroid_norm_x')},{val('recorded_centroid_norm_y')})",
        f"REPLAY   |  target px=({val('reconstructed_target_x_px', 1)},{val('reconstructed_target_y_px', 1)}) "
        f"{result.target_source}  |  N/F err={val('near_error')}/{val('far_error')} "
        f"level={result.curve_level:.2f} {mode} {result.confirm_frames}/{result.clear_frames}",
        f"REPLAY   |  yellow={'yes' if result.yellow_detected else 'no'} area={val('yellow_area_ratio', 4)} "
        f"count={data['reconstructed_yellow_count']}  |  {event}",
    ]


def render_frames(bag, db, args, clock, topics, params, video, results):
    focused = args.layout == "focus"
    panel = FocusDashboard(args.resolution) if focused else Dashboard(args.resolution, args.layout)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, panel.size)
    if not writer.isOpened():
        writer.release()
        raise OSError("OpenCV could not open the MP4 encoder (mp4v)")
    keydir = results / "keyframes"
    keydir.mkdir()
    cfr = VideoClock(writer, args.fps)
    vision, latch, red_history = Vision(params), YellowLatch(params), RedHistory(params)
    counts = Counter()
    origin = previous = minimum = maximum = None
    age = int(args.sync_max_age*Second)
    gap_ns = int(args.frame_gap*Second)
    segment = 0
    force_segment = False
    last_candidate_ns = None
    previously_yellow = False

    image_topic = topics["image"]
    debug_topic = topics.get("debug_image") or getattr(args, "debug_image_topic", DEFAULT_DEBUG_IMAGE_TOPIC)
    overlay_source = getattr(args, "overlay_source", "auto")
    debug_only_fallback = image_topic not in bag.seen and debug_topic in bag.seen
    if debug_only_fallback:
        image_topic = debug_topic
        use_recorded_debug = False
    else:
        use_recorded_debug = (
            overlay_source != "reconstructed"
            and debug_topic in bag.seen
            and debug_topic != image_topic
        )

    def process_frame(ns, receipt, source, image, recorded_debug):
        nonlocal origin, previous, minimum, maximum, segment, force_segment
        nonlocal last_candidate_ns, previously_yellow
        if ns == previous:
            counts["duplicate_image_stamp_skipped"] += 1
            return
        delta = None if previous is None else ns-previous
        new_segment = delta is None or delta < 0 or delta > gap_ns or force_segment
        if new_segment:
            segment += 1
            origin = ns if origin is None else origin
            vision.reset()
            latch.reset()
            red_history.reset()
            last_candidate_ns = None
            previously_yellow = False
            started = db.latest("start", ns)
            if started:
                latch.start(started.value[0])
            if segment > 1:
                why = ("image_decode_error" if force_segment else
                       "backward_image_stamp" if delta < 0 else "long_image_gap")
                db.add("replay", ns, ("image_segment_reset", None, None, why), source)
                counts[why] += 1
        else:
            for switch in db.range("start", previous+1, ns):
                if latch.start(switch.value[0]):
                    red_history.reset()
                if not switch.value[0]:
                    vision.reset()

        current = vision.process(image, latch.started)
        events = latch.update(ns, current.yellow_detected)
        red_history.update(current, latch.started, latch.stopping(ns))
        transition = any(kind == "yellow_pixel_trigger" for kind, _, _ in events)
        mask_rising = current.yellow_detected and not previously_yellow
        previously_yellow = current.yellow_detected
        if mask_rising and not transition:
            events.append(("yellow_mask_rising", None, None))
        for kind, count, total in events:
            db.add("replay", ns, (kind, count, total, f"segment {segment}"), source)
            counts[kind] += 1
        counts["frames_processed"] += 1
        if recorded_debug is not None:
            counts["recorded_debug_overlay_frames"] += 1
        else:
            counts["reconstructed_debug_overlay_frames"] += 1
        if latch.started is None:
            counts["frames_start_unknown"] += 1
        number = counts["frames_processed"]
        row = frame_data(db, ns, origin, segment, number, receipt, source,
                         None if new_segment else Second/delta, current, latch, age, image.shape)
        frames.writerow(row)
        log = db.latest("yellow_log", ns, 600_000_000)
        recorded_event = log and (new_segment or log.ns > previous) and ns-log.ns <= gap_ns
        if transition or mask_rising:
            last_candidate_ns = ns
        if log:
            point = f"RECORDED yellow {log.value[1]}/{log.value[2]} ({(ns-log.ns)/Second:.2f}s ago)"
        elif last_candidate_ns is not None and ns-last_candidate_ns <= 600_000_000:
            point = "RECONSTRUCTED yellow observation"
        else:
            point = ""
        rendered = None

        def canvas():
            nonlocal rendered
            if rendered is None:
                if focused:
                    rendered = panel.compose(image, current, params, row, point, recorded_debug=recorded_debug)
                else:
                    rendered = panel.compose(image, current, params, footer(row, current, point),
                                             recorded_debug=recorded_debug)
            return rendered

        cfr.push(ns, canvas, new_segment)
        if ((new_segment or transition or mask_rising or recorded_event) and
                counts["keyframes"] < args.keyframes):
            tag = "recorded_yellow" if recorded_event else (
                "vision_yellow" if transition else "segment")
            image_path = keydir / f"{number:08}_{tag}.png"
            if not cv2.imwrite(str(image_path), canvas()):
                raise OSError(f"Cannot write {image_path}")
            counts["keyframes"] += 1
        previous, force_segment = ns, False
        minimum = ns if minimum is None else min(minimum, ns)
        maximum = ns if maximum is None else max(maximum, ns)

    try:
        with (results / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
            frames = csv.DictWriter(stream, FRAME_FIELDS)
            frames.writeheader()
            wanted = {image_topic, debug_topic} if use_recorded_debug else {image_topic}
            pending = None
            last_debug = None

            def flush_pending():
                nonlocal pending, last_debug
                if pending is None:
                    return
                p_ns, p_receipt, p_source, p_image = pending
                matched = None
                if last_debug is not None and abs(p_ns - last_debug[0]) <= age:
                    matched = last_debug[1]
                    last_debug = None
                process_frame(p_ns, p_receipt, p_source, p_image, matched)
                pending = None

            for topic, receipt, msg in bag.messages(wanted):
                if use_recorded_debug and topic == debug_topic:
                    counts["debug_images_seen"] += 1
                    try:
                        dbg_image = decode_image(msg)
                    except (cv2.error, ValueError, TypeError, AttributeError):
                        counts["debug_image_decode_failed"] += 1
                        continue
                    dbg_ns, _ = clock.choose(receipt, header_ns(msg))
                    if pending is not None and abs(dbg_ns - pending[0]) <= age:
                        p_ns, p_receipt, p_source, p_image = pending
                        pending = None
                        last_debug = None
                        process_frame(p_ns, p_receipt, p_source, p_image, dbg_image)
                    else:
                        last_debug = (dbg_ns, dbg_image)
                    continue

                counts["images_seen"] += 1
                if use_recorded_debug:
                    flush_pending()
                try:
                    image = decode_image(msg)
                except (cv2.error, ValueError, TypeError, AttributeError):
                    counts["image_decode_failed"] += 1
                    force_segment = True
                    continue
                ns, source = clock.choose(receipt, header_ns(msg))
                counts[f"image_{source}_stamp"] += 1
                if debug_only_fallback:
                    process_frame(ns, receipt, source, image, image)
                elif use_recorded_debug:
                    if last_debug is not None and ns == last_debug[0]:
                        matched = last_debug[1]
                        last_debug = None
                        process_frame(ns, receipt, source, image, matched)
                    else:
                        pending = (ns, receipt, source, image)
                else:
                    process_frame(ns, receipt, source, image, None)
            if use_recorded_debug:
                flush_pending()
    finally:
        writer.release()
    if origin is None:
        raise ValueError(f"No decodable images on {image_topic}; available: {', '.join(bag.seen)}")
    if not video.is_file() or video.stat().st_size == 0:
        raise OSError("MP4 encoder produced no output")
    counts.update(output_frames=cfr.count, held_output_frames=cfr.duplicates, segments=segment)
    db.flush()
    return origin, minimum, maximum, counts


def state_events(db):
    previous = None
    for sample in db.range("state"):
        state, = sample.value
        if state != previous:
            # Recording can begin mid-stop; the initial state is not a trigger.
            kind = ("track_state_initial" if previous is None else
                    "yellow_state_enter" if state == "yellow_line_detected" else "track_state_change")
            yield sample.ns, "recorded_track_state", sample.clock, (kind, None, None, state)
        previous = state


def events_csv(db, args, results, origin):
    streams = [
        ((s.ns, "recorded_rosout", s.clock, s.value) for s in db.range("log")),
        ((s.ns, "reconstructed_vision", s.clock, s.value) for s in db.range("replay")),
        state_events(db),
    ]
    statuses = Counter()
    with (results / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, EVENT_FIELDS)
        writer.writeheader()
        for ns, source, stamp, (event, count, total, detail) in heapq.merge(*streams, key=lambda x: x[0]):
            row = dict(time_ns=ns, relative_s=(ns-origin)/Second, source=source,
                       time_clock=stamp, event=event, yellow_count=count, yellow_total=total,
                       detail=detail)
            fallback = False
            if event == "yellow_state_enter":
                # State is republished throughout a stop. If a nearby log exists,
                # assign metrics only to the log trigger to avoid duplicate stops.
                fallback = next(db.range("yellow_log", ns-250_000_000, ns+250_000_000), None) is None
            if event == "yellow_trigger" or fallback:
                b = braking(db, ns, max_gap=args.odom_max_gap, dwell=args.stop_dwell,
                            stop_speed=args.stop_speed, stop_yaw=args.stop_yaw)
                row.update(brake_cmd_ns=b.command_ns, odom_stop_ns=b.stopped_ns,
                           trigger_to_brake_cmd_s=None if b.command_ns is None else (b.command_ns-ns)/Second,
                           brake_duration_s=b.duration, odom_signed_m=b.distance,
                           odom_forward_m=b.forward_distance, metric_status=b.status)
                statuses[b.status] += 1
            writer.writerow(row)
    return statuses


def export_speed(db, results, origin):
    with (results / "speed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_ns", "relative_s", "source", "sample_clock", "linear_x_mps", "angular_z_radps"))
        # Bind each label eagerly. The generators themselves stream Query results.
        def samples(role):
            for sample in db.range(role):
                yield sample.ns, role, sample

        for ns, role, s in heapq.merge(samples("odom"), samples("cmd"), key=lambda x: x[0]):
            writer.writerow((ns, (ns-origin)/Second, role, s.clock, *s.value))


def report(args, bag, topics, recorded, images, brake_status, results, origin, params_path):
    layout = ("focus view" if args.layout == "focus"
              else f"{args.layout[0]} x {args.layout[1]} tiles")
    lines = [
        "# Offline ROS 2 bag analysis", "",
        f"- Input: `{args.bag.name}` ({bag.storage}, {bag.backend})",
        f"- Parameter YAML: `{params_path.name}` for `{args.node}` "
        f"(SHA-256 `{hashlib.sha256(params_path.read_bytes()).hexdigest()}`)",
        f"- OpenCV: {cv2.__version__}",
        f"- Video: {layout}, {args.fps:g} FPS, "
        f"{args.resolution[0]} x {args.resolution[1]}",
        f"- Overlay mode: `{args.overlay_source}` "
        f"(recorded debug_image frames: {images.get('recorded_debug_overlay_frames', 0)}, "
        f"reconstructed overlay frames: {images.get('reconstructed_debug_overlay_frames', 0)})",
        f"- Timeline origin: `{origin}` ns. Policy: {args.clock}, maximum header skew "
        f"{args.max_header_skew:g} s. Speed/centroid as-of limit: {args.sync_max_age:g} s.", "",
        "| Role | Topic | Recorded type | Indexed samples |", "|---|---|---|---:|",
    ]
    for role, topic in topics.items():
        if role == "image":
            count = images["images_seen"]
            recorded_type = bag.seen.get(topic, "missing")
        elif role == "debug_image":
            count = images.get("debug_images_seen", 0)
            recorded_type = bag.seen.get(topic, "missing (offline overlay reconstructed)")
        else:
            count = recorded[role]
            recorded_type = bag.seen.get(topic, "missing")
        lines.append(f"| {role} | `{topic}` | {recorded_type} | {count} |")
    lines += ["", "The log count only includes recognized events from the selected follower logger.",
              "", "## Coverage", ""]
    for values in (images, brake_status):
        lines.extend(f"- {key}: {value}" for key, value in sorted(values.items()))
    for topic, count in sorted(bag.decode_errors.items()):
        lines.append(f"- ROS message deserialization errors on `{topic}`: {count}")
    for key, count in sorted(recorded.items()):
        if key.startswith("invalid_"):
            lines.append(f"- {key}: {count}")
    lines += [
        "", "## Reading the files", "",
        "- `frames.csv` has one row per processed source image. MP4 uses fixed output FPS; "
        "large gaps and backward header jumps become separate segments, so use CSV timestamps "
        "for analysis. `speed.csv` keeps every valid recorded Odom and command sample.",
        "- In `events.csv`, `recorded_rosout` is a controller log, "
        "`recorded_track_state` is a recorded state transition and `reconstructed_vision` "
        "is an offline candidate. The image counter is relative to the replay segment and can "
        "differ if recording began mid-run or frames or start transitions are missing.",
        "- The brake interval starts at the first observed forward-to-zero/reverse command "
        "near the recorded trigger. Its end is the first Odom sample in a sustained low-speed "
        f"window (|v| ≤ {args.stop_speed:g} m/s, |yaw| ≤ {args.stop_yaw:g} rad/s, "
        f"{args.stop_dwell:g} s). Integrals use linear interpolation and trapezoids, with a "
        f"{args.odom_max_gap:g} s maximum Odom gap. Signed displacement and forward-only travel "
        "are separate. Blank fields and metric_status preserve missing evidence.",
        "- Integrated Odom speed estimates vehicle travel during braking. It does not give "
        "camera-to-yellow-line distance, bumper clearance, or ground truth. No camera geometry "
        "is inferred from pixel position.",
        "- The video overlay is reconstructed. Recorded command/state remains authoritative for "
        "the recorded topics. Red votes and risk are computed from the replayed segment. "
        "ROS scheduling, startup history, timer callbacks and command arbitration are outside this replay.",
    ]
    (results / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(args):
    if args.keyframes < 0:
        raise ValueError("--keyframes must be nonnegative")
    params_path = resolve_params_path(args.params)
    output, sidecars = resolve_output_path(args.output, args.bag.resolve(), args.layout)
    topics = {role: getattr(args, role+"_topic") for role in TOPICS}
    if len(set(topics.values())) != len(topics):
        raise ValueError("Each stream requires a distinct topic")
    topics["debug_image"] = args.debug_image_topic
    params = load_params(params_path, args.node)
    bag = Bag(args.bag, args.storage, args.backend)
    clock = StampClock(args.clock, args.max_header_skew)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".aic-analysis-", dir=output.parent) as staging, Timeline() as db:
        staged = Path(staging)
        staged_results = staged / sidecars.name
        staged_results.mkdir()
        staged_video = staged / output.name
        recorded = index_telemetry(bag, db, args, clock, topics)
        origin, minimum, maximum, frames = render_frames(
            bag, db, args, clock, topics, params, staged_video, staged_results)
        brake_status = events_csv(db, args, staged_results, origin)
        export_speed(db, staged_results, origin)
        speed_chart(db, staged_results / "speed.png", minimum, maximum,
                    int(args.sync_max_age*Second))
        report(args, bag, topics, recorded, frames, brake_status, staged_results, origin, params_path)
        staged_results.rename(sidecars)
        staged_video.rename(output)
    print(f"{frames['frames_processed']} source frames, {frames['output_frames']} video frames")
    print(output)
    print(sidecars)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ImportError, RuntimeError, ValueError, KeyError, yaml.YAMLError, cv2.error) as err:
        print(f"rosbag_visualizer: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

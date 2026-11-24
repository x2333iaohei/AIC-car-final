"""Focused and detailed video layouts for offline replay."""

import math

import cv2
import numpy as np

from telemetry import Second
from vision import rotate_divider


WHITE = (232, 235, 239)
MUTED = (164, 178, 195)
YELLOW = (30, 225, 240)


def text(canvas, value, x, y, width, scale=.45, color=WHITE):
    if width <= 0:
        return
    value = str(value).encode("ascii", "replace").decode("ascii")
    font = cv2.FONT_HERSHEY_SIMPLEX
    measured = cv2.getTextSize(value, font, scale, 1)[0][0]
    if measured > width:
        scale *= width / measured
    cv2.putText(canvas, value, (x, y), font, scale, color, 1, cv2.LINE_AA)


def rectangle(image, box, color, top=0):
    x, y, w, h = box
    cv2.rectangle(image, (x, y+top), (x+w-1, y+top+h-1), color, 2)


def band_image(result):
    out = cv2.cvtColor(result.closed, cv2.COLOR_GRAY2BGR)
    for band, x, color in ((result.far_band, result.far, (255, 180, 70)),
                           (result.near_band, result.near, (100, 235, 95))):
        y0, y1 = band
        cv2.rectangle(out, (0, y0), (out.shape[1]-1, max(y0, y1-1)), color, 1)
        if x is not None:
            cv2.circle(out, (int(x), int((y0+y1)*.5)), 4, color, -1)
    if result.target:
        cv2.circle(out, tuple(map(int, result.target)), 6, (0, 0, 255), -1)
    return out


def debug_overlay(bgr, result, params, annotations=True):
    overlay = bgr.copy()
    mask = cv2.cvtColor(result.closed, cv2.COLOR_GRAY2BGR)
    overlay[result.top:result.bottom] = cv2.addWeighted(result.roi, .6, mask, .4, 0)
    if result.target:
        if result.box:
            rectangle(overlay, result.box, (0, 255, 0), result.top)
        x, y = result.target
        point = (int(x), int(y)+result.top)
        cv2.circle(overlay, point, 8, (0, 0, 255), -1)
        center = overlay.shape[1] // 2
        cv2.line(overlay, (center, result.top), (center, result.bottom), (0, 255, 255), 2)
        base = (overlay.shape[1]*.5, float(result.bottom))
        end = rotate_divider((x, y+result.top), base, params["red_divider_angle_deg"])
        cv2.line(overlay, tuple(map(round, base)), tuple(map(round, end)), (0, 255, 255), 2)
        red = result.red
        if red.box:
            rectangle(overlay, red.box, (0, 0, 255), result.top)
            for (rx, ry), color in ((red.center, (0, 0, 255)), (red.bottom, (0, 255, 255))):
                cv2.circle(overlay, (round(rx), round(ry+result.top)), 6, color, -1)
            context = ""
            if red.risk is not None:
                corridor = "IN_PATH" if red.in_corridor else "OUT_PATH"
                context = f" bottom={red.bottom_ratio:.2f} risk={red.risk:.2f} {corridor}"
                if red.very_close:
                    context += " VERY_CLOSE"
            if annotations:
                text(overlay, f"{red.kind.upper()} area={red.area_ratio:.4f}{context}",
                     10, 90, overlay.shape[1]-18, color=(40, 90, 255))
        if red.line_box:
            rectangle(overlay, red.line_box, (255, 0, 255), result.top)
    width = overlay.shape[1]-18
    if annotations and result.yellow_detected:
        text(overlay, "YELLOW LINE DETECTED", 10, 30, width, color=YELLOW)
    if annotations and params["enable_continuous_curve_assist"]:
        mode = "REV_CURVE" if result.reverse_curve else "NORMAL"
        text(overlay, f"{mode} near={result.near_error:+.2f} far={result.far_error:+.2f} "
             f"level={result.curve_level:.2f}", 10, 60, width, color=YELLOW)
    return overlay


def panels(bgr, result, params, recorded_debug=None):
    contours = result.roi.copy()
    if result.contours:
        cv2.drawContours(contours, result.contours, -1, (255, 166, 50), 1)
        for contour in result.contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(contours, (x, y), (x+w-1, y+h-1), (130, 140, 170), 1)
    if result.box:
        rectangle(contours, result.box, (0, 255, 0))
    hue = result.hsv.copy()
    # Give hue its own color map, with S and V displayed alongside it.
    hue[:, :, 1] = 255
    hue[:, :, 2] = np.where(result.hsv[:, :, 1] == 0, 0, 255)
    hsv_display = np.hstack([cv2.cvtColor(hue, cv2.COLOR_HSV2BGR),
                             cv2.cvtColor(result.hsv[:, :, 1], cv2.COLOR_GRAY2BGR),
                             cv2.cvtColor(result.hsv[:, :, 2], cv2.COLOR_GRAY2BGR)])
    road_label = "HSV white mask" if params["use_white_filter"] else (
        "Gray threshold (inverted)" if params["invert_binary"] else "Gray threshold")
    yellow_label = ("Yellow mask (disabled)" if result.yellow is None else
                    f"Yellow mask | {result.yellow_ratio:.3%}")
    overlay_tile = (
        ("12 Recorded debug_image", recorded_debug, "")
        if recorded_debug is not None else
        ("12 Reconstructed overlay", debug_overlay(bgr, result, params, annotations=True), "")
    )
    return [
        ("01 Original frame", bgr, ""),
        ("02 ROI", result.roi, ""),
        ("03 Gaussian blur", result.blur, ""),
        ("04 HSV channels | H / S / V", hsv_display, ""),
        ("05 " + road_label, result.road, ""),
        ("06 Morphology open", result.opened, ""),
        ("07 Morphology close", result.closed, ""),
        ("08 " + yellow_label, result.yellow, "Disabled by YAML"),
        ("09 Red mask | raw ROI path", result.red.mask, result.red.status),
        ("10 External contours / boxes", contours, ""),
        ("11 Near / Far / Target", band_image(result), ""),
        overlay_tile,
    ]


class Dashboard:
    def __init__(self, size, layout):
        self.size = width, height = size
        self.rows, self.cols = layout
        self.footer = 104
        if (width % 2 or height % 2 or self.rows*self.cols < 12 or
                self.rows*self.cols > 48 or width//self.cols < 112 or
                (height-self.footer)//self.rows < 66):
            raise ValueError("Use even MP4 dimensions and a layout with 12–48 usable tiles "
                             "(minimum 112x66 per tile plus 104 px footer).")

    def compose(self, raw, result, params, lines, recorded_debug=None):
        width, height = self.size
        canvas = np.full((height, width, 3), (21, 19, 17), dtype=np.uint8)
        grid_height = height-self.footer
        for i, (title, image, missing) in enumerate(panels(raw, result, params, recorded_debug)):
            row, col = divmod(i, self.cols)
            x, y = width*col//self.cols + 3, grid_height*row//self.rows + 3
            right, bottom = width*(col+1)//self.cols-3, grid_height*(row+1)//self.rows-3
            canvas[y:bottom, x:right] = (43, 35, 30)
            text(canvas, title, x+7, y+15, right-x-14)
            if image is None:
                text(canvas, missing, x+9, (y+bottom)//2, right-x-18, color=MUTED)
                continue
            fit_image(canvas, image, x, y+21, right, bottom)
        for i, line in enumerate(lines):
            text(canvas, line, 12, grid_height+16+17*i, width-24, color=WHITE if i else YELLOW)
        return canvas


def fit_image(canvas, image, x, y, right, bottom):
    height, width = image.shape[:2]
    scale = min((right-x)/width, (bottom-y)/height)
    iw, ih = max(1, round(width*scale)), max(1, round(height*scale))
    resized = cv2.resize(image, (iw, ih),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    dx, dy = x+(right-x-iw)//2, y+(bottom-y-ih)//2
    canvas[dy:dy+ih, dx:dx+iw] = resized


class FocusDashboard:
    """Lead with the road and target; use two side panels to explain what the car sees."""

    def __init__(self, size):
        self.size = width, height = size
        if width % 2 or height % 2 or width < 640 or height < 360:
            raise ValueError("Focus view requires even MP4 dimensions of at least 640x360.")

    @staticmethod
    def _status(data, result, event):
        state = data["track_state"]
        if data["started"] is False:
            return "Paused", "Recorded start signal", False
        if state == "yellow_line_detected":
            return "Yellow-line stop", "Recorded controller state", True
        if event.startswith("RECORDED"):
            return "Yellow-line trigger", "Recorded controller log", True
        if result.yellow_detected or event:
            return "Yellow line visible", "Reconstructed from camera", True
        if not result.target:
            return "No road target", "Reconstructed from camera", True
        if state:
            return state.replace("_", " ").capitalize(), "Recorded controller state", False
        return "Road target visible", "Reconstructed from camera", False

    def compose(self, raw, result, params, data, event, recorded_debug=None):
        width, height = self.size
        unit = min(width / 1280, height / 800)
        pad = max(8, round(20*unit))
        gap = max(6, round(16*unit))
        title_y = max(26, round(40*unit))
        top = title_y + pad
        bottom = height - max(66, round(100*unit))
        left_end = int(width*.675)
        right_start = left_end + gap
        right_end = width-pad
        middle = (top+bottom)//2
        background = (22, 19, 16)
        surface = (39, 33, 28)
        canvas = np.full((height, width, 3), background, np.uint8)

        text(canvas, "Vision replay", pad, title_y, width//2, 1.12*unit)
        text(canvas, f"{data['relative_s']:.2f} s   /   Frame {data['frame']:04d}",
             right_start, title_y, right_end-right_start, .73*unit, MUTED)

        def card(x, y, end_x, end_y, title, image, subtitle="", highlight=False):
            canvas[y:end_y, x:end_x] = surface
            title_height = max(25, round(38*unit))
            inset = max(5, round(12*unit))
            text(canvas, title, x+inset, y+title_height-inset, end_x-x-2*inset,
                 .68*unit, YELLOW if highlight else WHITE)
            picture_top = y + title_height
            picture_bottom = end_y - (title_height if subtitle else inset)
            if image is not None:
                fit_image(canvas, image, x+inset, picture_top, end_x-inset, picture_bottom)
            if subtitle:
                text(canvas, subtitle, x+inset, end_y-inset, end_x-x-2*inset, .56*unit, MUTED)

        if recorded_debug is not None:
            main_title = "Road and target  /  recorded debug_image"
            main_image = recorded_debug
            main_subtitle = "Recorded /line_follower/debug_image from bag"
        else:
            main_title = "Road and target  /  offline overlay (reconstructed)"
            main_image = debug_overlay(raw, result, params, annotations=True)
            main_subtitle = "Red dot: target   |   Yellow line: steering divider   |   Full debug overlay"
        card(pad, top, left_end, bottom, main_title, main_image, main_subtitle)
        card(right_start, top, right_end, middle-gap//2, "Lane sampling",
             band_image(result),
             "Green: Near   Blue: Far" + ("   |   S-turn confirmed" if result.reverse_curve else ""))
        yellow = result.yellow
        coverage = "Filter disabled"
        if yellow is not None:
            coverage = f"Yellow area {result.yellow_ratio:.2%}"
            if data["reconstructed_yellow_count"]:
                coverage += f"   |   Replay count {data['reconstructed_yellow_count']}"
            if result.yellow_detected:
                yellow = (yellow[:, :, None] > 0) * np.array(YELLOW, dtype=np.uint8)
        card(right_start, middle+gap//2, right_end, bottom, "Yellow-line mask",
             yellow, coverage, result.yellow_detected)

        display, source, highlight = self._status(data, result, event)
        status_end = int(width*.44)
        footer_top = bottom+gap
        footer_bottom = height-pad
        tone = (40, 54, 68) if highlight else surface
        canvas[footer_top:footer_bottom, pad:status_end] = tone
        caption_y = footer_top + max(14, round(20*unit))
        value_y = footer_bottom - max(8, round(13*unit))
        text(canvas, source.upper(), pad+gap, caption_y, status_end-pad-2*gap, .48*unit, MUTED)
        text(canvas, display, pad+gap, value_y, status_end-pad-2*gap, .95*unit,
             YELLOW if highlight else WHITE)
        interval = (right_end - status_end - gap)//2
        for column, (label, key) in enumerate((("MEASURED SPEED", "odom_v"),
                                                ("COMMANDED SPEED", "cmd_v"))):
            x = status_end+gap + interval*column
            end_x = x+interval-gap if column == 0 else right_end
            canvas[footer_top:footer_bottom, x:end_x] = surface
            text(canvas, label, x+gap, caption_y, end_x-x-2*gap, .48*unit, MUTED)
            speed = "Unavailable" if data[key] is None else f"{data[key]:.2f} m/s"
            text(canvas, speed, x+gap, value_y, end_x-x-2*gap, .95*unit)
        return canvas


class VideoClock:
    """Turn sampled source images into a CFR video and cut across large source gaps."""

    def __init__(self, writer, fps):
        self.writer = writer
        self.period = round(Second/fps)
        self.next_ns = None
        self.last = None
        self.count = 0
        self.duplicates = 0

    def _write(self, image):
        self.writer.write(image)
        self.count += 1

    def push(self, ns, render, new_segment=False):
        if self.next_ns is None or new_segment:
            self.last = render()
            self._write(self.last)
            self.next_ns = ns+self.period
        elif ns >= self.next_ns:
            missing = (ns - self.next_ns)//self.period
            for _ in range(missing):
                self._write(self.last)
            self.duplicates += missing
            self.last = render()
            self._write(self.last)
            self.next_ns += (missing+1)*self.period


def speed_chart(db, path, origin, end, age):
    """Draw recorded extrema per pixel without holding the complete time series."""
    width, height = 1280, 480
    left, right, top, bottom = 105, 1254, 56, 406
    start, duration = min(origin, end), max(1, abs(end-origin))
    slots = right-left+1
    image = np.full((height, width, 3), (22, 20, 17), np.uint8)
    tracks = {}
    for role in ("odom", "cmd"):
        low = np.full(slots, math.inf)
        high = np.full(slots, -math.inf)
        for sample in db.range(role, start, start+duration):
            ix = (sample.ns-start)*(slots-1)//duration
            v = sample.value[0]
            low[ix] = min(low[ix], v)
            high[ix] = max(high[ix], v)
        ok = np.isfinite(low)
        if ok.any():
            tracks[role] = (low, high, np.flatnonzero(ok))

    lower = min([0.] + [float(low[ix].min()) for low, _, ix in tracks.values()])
    upper = max([.1] + [float(high[ix].max()) for _, high, ix in tracks.values()])
    pad = (upper-lower) * .06
    lower, upper = lower-pad, upper+pad

    def y(value):
        return round(bottom - (value-lower)/(upper-lower)*(bottom-top))

    for i in range(6):
        yy = top+(bottom-top)*i//5
        xx = left+(right-left)*i//5
        cv2.line(image, (left, yy), (right, yy), (61, 55, 49), 1)
        cv2.line(image, (xx, top), (xx, bottom), (61, 55, 49), 1)
        text(image, f"{upper-(upper-lower)*i/5:+.2f}", 18, yy+4, 84)
        text(image, f"{duration/Second*i/5:.2f}", xx-12, bottom+24, 64)
    for sample in db.range("log", start, start+duration):
        if sample.value[0] == "yellow_trigger":
            xx = left+(sample.ns-start)*(slots-1)//duration
            cv2.line(image, (xx, top), (xx, bottom), (65, 138, 151), 1)
    for role, color in (("cmd", (95, 175, 255)), ("odom", (224, 215, 89))):
        if role not in tracks:
            continue
        low, high, indices = tracks[role]
        last = None
        for ix in indices:
            p = left+int(ix)
            lo, hi = y(low[ix]), y(high[ix])
            cv2.line(image, (p, lo), (p, hi), color, 1)
            if last is not None and (ix-last[0])*duration/(slots-1) <= age:
                cv2.line(image, (left+last[0], last[1]), (p, (lo+hi)//2), color, 1)
            last = (int(ix), (lo+hi)//2)
        text(image, role, 930 if role == "cmd" else 1050, 32, 110, color=color)
    text(image, "Recorded longitudinal speed (m/s) | source timeline (s)", 20, 32, 810, .64)
    if not tracks:
        text(image, "No recorded speed samples in the image interval", 134, 180, 850)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")

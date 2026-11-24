"""Parameter subset consumed by the offline replay, from the C++ declarations."""

import math
from pathlib import Path

import yaml


DEFAULTS = {
    "roi_top_ratio": .5, "roi_bottom_ratio": .05, "gaussian_kernel": 5,
    "binary_threshold": 160., "morph_kernel": 5, "morph_iterations": 2,
    "min_contour_area": 2000., "invert_binary": False, "use_white_filter": False,
    "white_h_low": 0, "white_h_high": 180, "white_s_max": 60, "white_v_min": 180,
    "use_yellow_filter": False, "yellow_h_low": 20, "yellow_h_high": 40,
    "yellow_s_min": 80, "yellow_v_min": 120, "yellow_area_ratio_threshold": .1,
    "yellow_stop_duration": 1., "yellow_permanent_count": 5, "yellow_clear_frames": 10,
    "enable_red_filter": False, "red_h_low_1": 0, "red_h_high_1": 10,
    "red_h_low_2": 160, "red_h_high_2": 180, "red_s_min": 100, "red_v_min": 80,
    "red_line_aspect_ratio_min": 3., "red_line_min_width_ratio": .30,
    "red_cone_area_ratio_threshold": .0006,
    "red_cone_min_bbox_width": 4, "red_cone_min_bbox_height": 4,
    "red_divider_angle_deg": 0., "red_line_fallback_direction": -1,
    "enable_soft_red_avoidance": False, "red_vote_frames": 8,
    "red_near_ratio": .35, "red_very_close_ratio": .72, "red_corridor_half_width_ratio": .42,
    "enable_continuous_curve_assist": False, "reverse_curve_min_error": .12,
    "reverse_curve_threshold": .35, "reverse_curve_confirm_frames": 3,
    "reverse_curve_clear_frames": 4, "enable_curve_preview_steering": False,
    "curve_preview_max_far_weight": .45,
    "enable_red_zone_cluster": False, "red_zone_close_kernel_w": 35,
    "red_zone_close_kernel_h": 13, "red_zone_min_width_ratio": .11,
    "red_zone_min_area_ratio": .0030, "red_zone_max_fill_ratio": .30,
    "red_zone_near_ratio": .28,
    "enable_tjunction_merge": False, "tjunction_wall_min_y_ratio": .23,
    "tjunction_wall_max_y_ratio": .62, "tjunction_wall_max_spread_ratio": .10,
    "tjunction_min_full_rows": 55,
}


def clamp(value, low, high):
    return max(low, min(value, high))


def parameters(**overrides):
    result = DEFAULTS.copy()
    for key, value in overrides.items():
        if key not in result:
            continue
        expected = type(result[key])
        if type(value) is not expected or (expected == float and not math.isfinite(value)):
            raise ValueError(f"{key} must be a ROS YAML {expected.__name__}, got {value!r}")
        result[key] = value

    g = max(1, result["gaussian_kernel"])
    result["gaussian_kernel"] = g if g % 2 else g + 1
    for key in ("morph_kernel", "morph_iterations", "yellow_permanent_count",
                "yellow_clear_frames", "reverse_curve_confirm_frames", "reverse_curve_clear_frames",
                "red_cone_min_bbox_width", "red_cone_min_bbox_height", "red_vote_frames"):
        result[key] = max(1, result[key])

    for color in ("white", "yellow"):
        low, high = color + "_h_low", color + "_h_high"
        result[low], result[high] = sorted(clamp(result[key], 0, 180) for key in (low, high))
    for key in ("white_s_max", "white_v_min", "yellow_s_min", "yellow_v_min"):
        result[key] = clamp(result[key], 0, 255)
    return result


def load_params(path: Path, node="/line_follower_node"):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Parameter YAML must contain a ROS node mapping")
    selected = {}

    def visit(branch, prefix=""):
        for key, item in branch.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                continue
            full = prefix + "/" + key.strip("/")
            if isinstance(item.get("ros__parameters"), dict):
                if full in ("/**", "/" + node.strip("/")):
                    # ROS applies matching sections in YAML order.
                    selected.update(item["ros__parameters"])
            else:
                visit(item, full)

    visit(document)
    if not any(key in DEFAULTS for key in selected):
        raise ValueError(f"No recognized vision parameters for {node} in {path}")
    return parameters(**selected)

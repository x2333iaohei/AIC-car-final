import re
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from vision import RedHistory, Vision, band_center
from params import DEFAULTS, parameters, load_params
from telemetry import Second, Timeline, StampClock, braking


class VisionTests(unittest.TestCase):
    def frame(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[:, 64:105] = 255
        frame[60:65, :40] = (0, 255, 255)
        return frame

    def test_defaults_are_the_cpp_declarations(self):
        here = Path(__file__).resolve()
        candidates = (
            here.parents[3] / "src/line_follow_ws/src/line_follower/src/line_follower_node.cpp",
            here.parents[2] / "ws_rdkx5_ring/src/line_follow_ws/src/line_follower/src/line_follower_node.cpp",
            here.parents[2] / "ws_rdkx5/src/line_follow_ws/src/line_follower/src/line_follower_node.cpp",
        )
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            self.skipTest("line_follower_node.cpp not present next to rosbag_visualizer")
        text = source.read_text(encoding="utf-8")
        checked = 0
        for key, value in DEFAULTS.items():
            matches = re.findall(r'declare_parameter\("' + key + r'",\s*([^);]+)\);', text)
            if not matches:
                continue
            checked += 1
            with self.subTest(parameter=key):
                literal, = matches
                actual = literal == "true" if type(value) is bool else type(value)(literal)
                self.assertEqual(actual, value)
        self.assertGreater(checked, 30)

    def test_roi_hsv_and_yellow_equality(self):
        p = parameters(roi_top_ratio=0.60, roi_bottom_ratio=0.10, gaussian_kernel=1,
                       use_white_filter=True, white_s_max=25, use_yellow_filter=True,
                       yellow_area_ratio_threshold=200 / (50 * 160))
        frame = self.frame()
        result = Vision(p).process(frame, True)
        self.assertEqual((result.top, result.bottom), (40, 90))
        self.assertEqual(result.road[21, 0], 0)
        self.assertEqual(cv2.countNonZero(result.yellow), 200)
        self.assertTrue(result.yellow_detected)
        self.assertEqual(result.target[0], 84)

    def test_gray_inversion_not_combined_with_hsv(self):
        frame = self.frame()
        common = dict(gaussian_kernel=1, binary_threshold=180., invert_binary=True)
        gray = Vision(parameters(**common)).process(frame, True)
        hsv = Vision(parameters(**common, use_white_filter=True)).process(frame, True)
        self.assertEqual((gray.road[1, 0], hsv.road[1, 0]), (255, 0))

    def test_parent_roi_border_and_morphology(self):
        frame = self.frame()
        frame[39] = 220
        p = parameters(roi_top_ratio=.6, roi_bottom_ratio=.2, gaussian_kernel=4,
                       morph_kernel=3, morph_iterations=2)
        result = Vision(p).process(frame, True)
        gauss = cv2.getGaussianKernel(5, 0, cv2.CV_32F)
        reference = cv2.sepFilter2D(frame, -1, gauss, gauss)
        np.testing.assert_array_equal(result.blur, reference[40:80])
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(result.road, cv2.MORPH_OPEN, kernel, iterations=1)
        np.testing.assert_array_equal(result.opened, opened)
        np.testing.assert_array_equal(result.closed, cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                                                                     kernel, iterations=2))

    def test_large_gaussian_matches_cpp_roi_separable_path(self):
        rng = np.random.default_rng(24)
        frame = rng.integers(0, 256, (121, 161, 3), dtype=np.uint8)
        p = parameters(gaussian_kernel=11, roi_top_ratio=.6, roi_bottom_ratio=.2)
        result = Vision(p).process(frame, True)
        gauss = cv2.getGaussianKernel(11, 0, cv2.CV_32F)
        reference = cv2.sepFilter2D(frame, -1, gauss, gauss)
        np.testing.assert_array_equal(result.blur, reference[result.top:result.bottom])

    def test_red_path_is_raw_and_only_runs_when_started_with_target(self):
        frame = self.frame()
        frame[70:76, 25:33] = (0, 0, 255)
        p = parameters(gaussian_kernel=17, enable_red_filter=True, red_v_min=230,
                       red_cone_area_ratio_threshold=0.)
        result = Vision(p).process(frame, True)
        self.assertEqual(cv2.countNonZero(result.red.mask), 48)
        self.assertEqual(result.red.kind, "cone")
        self.assertIsNone(Vision(p).process(frame, False).red.mask)

    def test_band_run_selection_and_reverse_confirmation(self):
        mask = np.zeros((10, 120), np.uint8)
        mask[:3, 15:41] = 255
        mask[:, 65:95] = 255
        self.assertEqual(band_center(mask, 0, 10, 37), 27.5)
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[34:56, 105:145] = 255
        frame[62:88, 25:65] = 255
        vision = Vision(parameters(roi_top_ratio=1., roi_bottom_ratio=0., gaussian_kernel=1,
                                   morph_kernel=1, enable_continuous_curve_assist=True,
                                   reverse_curve_confirm_frames=2))
        self.assertAlmostEqual(vision.process(frame, True).target[0], 68.5)
        result = vision.process(frame, True)
        self.assertTrue(result.reverse_curve)
        self.assertEqual(result.target[0], 44.5)
        vision.reset()
        self.assertFalse(vision.process(frame, True).reverse_curve)

    def test_contour_fallback_and_red_classification(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        frame[5:25, 40:120] = 255
        frame[37:42, 60:150] = (0, 0, 255)
        frame[50:62, 15:28] = (10, 0, 255)
        p = parameters(gaussian_kernel=1, roi_top_ratio=1., roi_bottom_ratio=0.,
                       min_contour_area=400., morph_kernel=1, enable_red_filter=True)
        result = Vision(p).process(frame, True)
        self.assertEqual(result.target_source, "contour")
        self.assertEqual(result.target, (79.5, 14.5))
        self.assertEqual(result.red.kind, "cone")
        self.assertEqual(result.red.line_box, (60, 37, 90, 5))
        self.assertEqual(result.red.box, (15, 50, 13, 12))

    def test_red_votes_and_yellow_priority_in_debug_risk(self):
        frame = self.frame()
        frame[70:85, 75:90] = (0, 0, 255)
        p = parameters(gaussian_kernel=1, morph_kernel=1, enable_red_filter=True,
                       enable_soft_red_avoidance=True, red_vote_frames=2)
        vision, red = Vision(p), RedHistory(p)
        first = vision.process(frame, True)
        red.update(first, True, False)
        self.assertEqual(first.red.risk, 0.)
        second = vision.process(frame, True)
        red.update(second, True, False)
        self.assertTrue(second.red.in_corridor)
        self.assertGreater(second.red.risk, 0.)
        parked = vision.process(frame, True)
        red.update(parked, True, True)
        self.assertEqual(parked.red.risk, 0.)

    def test_yaml_node_specific_defaults_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preset.yaml"
            path.write_text("/**:\n  ros__parameters: {gaussian_kernel: 4}\n"
                            "/line_follower_node:\n  ros__parameters: {white_h_low: 200, white_h_high: 2}\n"
                            "cmd_mux_node:\n  ros__parameters: {white_h_low: 42}\n")
            p = load_params(path)
            self.assertEqual((p["gaussian_kernel"], p["white_h_low"], p["white_h_high"]), (5, 2, 180))


class SynchronizationTests(unittest.TestCase):
    def test_late_motion_command_cannot_supply_stop_confirmation(self):
        with Timeline() as db:
            db.add("cmd", 900_000_000, (.5, 0.), "bag")
            db.add("cmd", Second, (0., 0.), "bag")
            db.add("cmd", Second+250_000_000, (.5, 0.), "bag")
            for i, v in enumerate((.6, .3, 0., 0., 0.)):
                db.add("odom", Second + i * 100_000_000, (v, 0.), "header")
            result = braking(db, Second)
            self.assertIsNone(result.distance)
            self.assertEqual(result.status, "command_resumed_before_stop_confirmation")

    def test_source_stamp_and_causal_sample_age(self):
        clock = StampClock()
        self.assertEqual(clock.choose(10 * Second, 9 * Second), (9 * Second, "header"))
        self.assertEqual(clock.choose(10 * Second, 1), (10 * Second, "bag"))
        with Timeline() as db:
            db.add("odom", 200, (2., 0.), "header")
            db.add("odom", 100, (1., 0.), "header")
            self.assertEqual(db.latest("odom", 150, 60).value, (1., 0.))
            self.assertIsNone(db.latest("odom", 180, 60))
            self.assertIsNone(db.latest("odom", 90, 60))

    def test_measured_distance_and_gap_invalidates_it(self):
        with Timeline() as db:
            db.add("cmd", 900_000_000, (.5, 0.), "bag")
            db.add("cmd", Second, (0., 0.), "bag")
            for i, v in enumerate((.6, .4, .2, 0., 0., 0.)):
                db.add("odom", Second + i * 100_000_000, (v, 0.), "header")
            result = braking(db, Second, max_gap=.15, dwell=.19)
            self.assertAlmostEqual(result.distance, .09)
            self.assertAlmostEqual(result.duration, .3)
            self.assertIsNone(braking(db, Second, max_gap=.08, dwell=.19).distance)


if __name__ == "__main__":
    unittest.main()

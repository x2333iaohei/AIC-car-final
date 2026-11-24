from pathlib import Path
import csv
import importlib.util
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from render import TOPICS, main, parser, render_frames
from bag import Bag, decode_image
from params import parameters
from synthetic import make_bag
from telemetry import Second, StampClock, Timeline


class EndToEnd(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("rosbags"), "portable bag writer unavailable")
    def test_sqlite_and_mcap_with_serialized_ros_messages(self):
        for storage in ("sqlite3", "mcap"):
            with self.subTest(storage=storage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                bag, params = make_bag(root, storage)
                video = root / "analysis.mp4"
                options = ["--layout", "3x4"] if storage == "mcap" else []
                code = main(["--bag", str(bag), "--params", str(params), "--backend", "rosbags",
                             "--fps", "10", "--resolution", "960x600", "--output", str(video),
                             *options])
                self.assertEqual(code, 0)
                self.assertTrue(video.is_file())
                capture = cv2.VideoCapture(str(video))
                try:
                    ok, frame = capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape[:2], (600, 960))
                    self.assertGreater(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
                finally:
                    capture.release()
                results = root / "analysis_analysis"
                self.assertTrue(list((results / "keyframes").glob("*.png")))
                self.assertTrue((results / "speed.png").is_file())
                with (results / "events.csv").open() as events:
                    rows = list(csv.DictReader(events))
                actual = next(r for r in rows if r["event"] == "yellow_trigger")
                self.assertEqual(actual["source"], "recorded_rosout")
                self.assertAlmostEqual(float(actual["brake_duration_s"]), .4)
                self.assertGreater(float(actual["odom_forward_m"]), 0.)
                self.assertTrue(any(r["source"] == "reconstructed_vision" for r in rows))
                report = (results / "report.md").read_text()
                self.assertIn("synthetic", report)
                self.assertIn("3 x 4 tiles" if storage == "mcap" else "focus view", report)
                storage_file, = bag.glob("*.db3" if storage == "sqlite3" else "*.mcap")
                single = Bag(storage_file, "auto", "rosbags")
                images = single.messages({"/image_raw"})
                try:
                    topic, _, message = next(images)
                    self.assertEqual(topic, "/image_raw")
                    self.assertEqual(message.encoding, "bgr8")
                finally:
                    images.close()


    @unittest.skipUnless(importlib.util.find_spec("rosbag2_py") and importlib.util.find_spec("rosbags"),
                         "native ROS 2 Humble and portable fixture writer required")
    def test_native_sqlite_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bag, params = make_bag(root, "sqlite3")
            output = root / "native.mp4"
            self.assertEqual(main(["--bag", str(bag), "--params", str(params), "--output",
                                   str(output), "--backend", "rosbag2_py"]), 0)
            report = (root / "native_analysis" / "report.md").read_text()
            self.assertIn("(sqlite3, rosbag2_py)", report)


class CanvasTests(unittest.TestCase):
    def test_image_only_discontinuities_and_missing_telemetry(self):
        background = np.zeros((60, 110, 3), dtype=np.uint8)
        background[:, 20:80] = 255
        source_times = (10*Second, 10*Second, 10*Second+90_000_000, 11*Second, 10*Second+800_000_000)

        class Images:
            seen = {"/image_raw": "sensor_msgs/msg/Image"}

            def messages(self, wanted):
                self.selected = wanted
                for ns in source_times:
                    secs, nanos = divmod(ns, Second)
                    stamp = SimpleNamespace(sec=secs, nanosec=nanos)
                    yield "/image_raw", ns, SimpleNamespace(
                        encoding="bgr8", height=60, width=110, step=330,
                        data=background.reshape(-1), header=SimpleNamespace(stamp=stamp))

        with tempfile.TemporaryDirectory() as temp, Timeline() as db:
            output = Path(temp)
            args = parser().parse_args(["--bag", temp, "--params", "unused.yaml",
                                        "--output", str(output / "out.mp4"),
                                        "--resolution", "640x400", "--keyframes", "0"])
            bag = Images()
            _, _, _, counts = render_frames(bag, db, args, StampClock(), TOPICS, parameters(),
                                            args.output, output)
            self.assertEqual(bag.selected, {"/image_raw"})
            self.assertEqual(counts["frames_processed"], 4)
            self.assertEqual(counts["duplicate_image_stamp_skipped"], 1)
            self.assertEqual(counts["segments"], 3)
            with (output / "frames.csv").open() as csv_stream:
                rows = list(csv.DictReader(csv_stream))
            self.assertTrue(all(r["odom_v"] == r["cmd_v"] == r["track_state"] == "" for r in rows))

    def test_rgb_image_step_padding(self):
        data = np.array([255, 0, 0, 99, 0, 255, 0, 88], dtype=np.uint8)
        msg = SimpleNamespace(encoding="rgb8", height=2, width=1, step=4, data=data)
        np.testing.assert_array_equal(decode_image(msg), np.array([[[0, 0, 255]], [[0, 255, 0]]]))


    def test_resolution_and_layout_validation(self):
        from panels import Dashboard, FocusDashboard
        self.assertEqual(parser().get_default("layout"), "focus")
        self.assertEqual(FocusDashboard((640, 400)).size, (640, 400))
        self.assertEqual(Dashboard((960, 600), (3, 4)).size, (960, 600))
        with self.assertRaises(ValueError):
            Dashboard((960, 600), (2, 2))

    def test_default_local_output_path_and_auto_increment(self):
        import render
        with tempfile.TemporaryDirectory() as temp:
            orig_dir = render.DEFAULT_OUTPUT_DIR
            try:
                render.DEFAULT_OUTPUT_DIR = Path(temp)
                bag_dir = Path(temp) / "my_run_bag"
                bag_dir.mkdir()
                out1, side1 = render.resolve_output_path(None, bag_dir, "focus")
                self.assertEqual(out1, (Path(temp) / "my_run_bag_focus.mp4").resolve())
                self.assertEqual(side1, (Path(temp) / "my_run_bag_focus_analysis").resolve())
                out1.write_bytes(b"placeholder")
                out2, side2 = render.resolve_output_path(None, bag_dir, "focus")
                self.assertEqual(out2, (Path(temp) / "my_run_bag_focus_001.mp4").resolve())
                self.assertEqual(side2, (Path(temp) / "my_run_bag_focus_001_analysis").resolve())
            finally:
                render.DEFAULT_OUTPUT_DIR = orig_dir

    def test_recorded_debug_image_pairing_and_fallback(self):
        raw_img = np.zeros((60, 110, 3), dtype=np.uint8)
        raw_img[:, 20:80] = 255
        dbg_img = np.full((60, 110, 3), (10, 200, 10), dtype=np.uint8)

        class PairedBag:
            seen = {
                "/image_raw": "sensor_msgs/msg/Image",
                "/line_follower/debug_image": "sensor_msgs/msg/Image",
            }

            def messages(self, wanted):
                self.selected = wanted
                t1 = 10 * Second
                t2 = 10 * Second + 100_000_000
                s1 = SimpleNamespace(sec=t1 // Second, nanosec=t1 % Second)
                s2 = SimpleNamespace(sec=t2 // Second, nanosec=t2 % Second)
                yield "/image_raw", t1, SimpleNamespace(
                    encoding="bgr8", height=60, width=110, step=330,
                    data=raw_img.reshape(-1), header=SimpleNamespace(stamp=s1))
                # Debug image arrives 15ms later with identical header timestamp
                yield "/line_follower/debug_image", t1 + 15_000_000, SimpleNamespace(
                    encoding="bgr8", height=60, width=110, step=330,
                    data=dbg_img.reshape(-1), header=SimpleNamespace(stamp=s1))
                # Second frame has no matching debug_image -> falls back to reconstructed overlay
                yield "/image_raw", t2, SimpleNamespace(
                    encoding="bgr8", height=60, width=110, step=330,
                    data=raw_img.reshape(-1), header=SimpleNamespace(stamp=s2))

        with tempfile.TemporaryDirectory() as temp, Timeline() as db:
            output = Path(temp)
            args = parser().parse_args(["--bag", temp, "--params", "unused.yaml",
                                        "--output", str(output / "out.mp4"),
                                        "--resolution", "640x400", "--keyframes", "0"])
            bag = PairedBag()
            _, _, _, counts = render_frames(bag, db, args, StampClock(), TOPICS, parameters(),
                                            args.output, output)
            self.assertEqual(bag.selected, {"/image_raw", "/line_follower/debug_image"})
            self.assertEqual(counts["frames_processed"], 2)
            self.assertEqual(counts["recorded_debug_overlay_frames"], 1)
            self.assertEqual(counts["reconstructed_debug_overlay_frames"], 1)


if __name__ == "__main__":
    unittest.main()

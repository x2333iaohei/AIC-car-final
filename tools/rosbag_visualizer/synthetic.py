"""Create a small labeled ROS 2 fixture without starting ROS or generating commands on a bus."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from telemetry import Second


def make_bag(parent, storage):
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    bag = parent / f"synthetic_{storage}"
    params = parent / "synthetic.yaml"
    if params.exists():
        raise FileExistsError(params)
    store = get_typestore(Stores.ROS2_HUMBLE)

    def message(name, *args):
        return store.types[name](*args)

    def stamp(ns):
        return message("builtin_interfaces/msg/Time", *divmod(ns, Second))

    def vector(x=0.):
        return message("geometry_msgs/msg/Vector3", x, 0., 0.)

    def twist(v):
        return message("geometry_msgs/msg/Twist", vector(v), vector())

    p = message("geometry_msgs/msg/Point", 0., 0., 0.)
    q = message("geometry_msgs/msg/Quaternion", 0., 0., 0., 1.)
    covariance = np.zeros(36)
    pose = message("geometry_msgs/msg/PoseWithCovariance",
                   message("geometry_msgs/msg/Pose", p, q), covariance)
    topic_types = {
        "/image_raw": "sensor_msgs/msg/Image",
        "/odom": "nav_msgs/msg/Odometry",
        "/cmd_vel": "geometry_msgs/msg/Twist",
        "/line_follower/track_state": "std_msgs/msg/String",
        "/line_follower/start_cmd": "std_msgs/msg/Bool",
        "/line_follower/centroid": "geometry_msgs/msg/Point",
        "/rosout": "rcl_interfaces/msg/Log",
    }

    with Writer(bag, version=8, storage_plugin=StoragePlugin[storage.upper()]) as writer:
        connections = {topic: writer.add_connection(topic, kind, typestore=store)
                       for topic, kind in topic_types.items()}

        def write(topic, obj, time):
            writer.write(connections[topic], time, store.serialize_cdr(obj, topic_types[topic]))

        def log(text, time):
            write("/rosout", message("rcl_interfaces/msg/Log", stamp(time), 20,
                  "line_follower_node", text, "synthetic", "fixture", 1), time)

        for i in range(24):
            ns = 10*Second + i*100_000_000
            header = message("std_msgs/msg/Header", stamp(ns), "camera")
            velocity = .6 if i < 8 else max(0., .6-.15*(i-8))
            command = .6 if 1 <= i < 8 or i >= 16 else 0.
            wrapped = message("geometry_msgs/msg/TwistWithCovariance", twist(velocity), covariance)
            write("/odom", message("nav_msgs/msg/Odometry", header, "base_link", pose, wrapped), ns)
            write("/cmd_vel", twist(command), ns)
            if i in (0, 1):
                write("/line_follower/start_cmd", message("std_msgs/msg/Bool", bool(i)), ns)
            if i in (1, 8, 16):
                state = "yellow_line_detected" if i == 8 else "tracking"
                write("/line_follower/track_state", message("std_msgs/msg/String", state), ns)
            if i == 8:
                log("New yellow line detected! Count: 1/4", ns)
            if i == 16:
                log("Yellow line stop duration ended. Resuming line following.", ns)
            frame = np.full((240, 320, 3), (44, 35, 29), np.uint8)
            white = [(80, 240), (260, 240), (225, 75), (180, 75)]
            cv2.fillPoly(frame, [np.array(white, np.int32)], (245, 247, 250))
            cv2.rectangle(frame, (203, 165), (223, 188), (0, 0, 255), -1)
            if 8 <= i <= 12:
                cv2.rectangle(frame, (100, 174+i), (248, 181+i), (0, 245, 255), -1)
            write("/line_follower/centroid", message("geometry_msgs/msg/Point", .60, .78, 0.), ns)
            write("/image_raw", message("sensor_msgs/msg/Image", header, 240, 320, "bgr8", 0,
                                        320*3, frame.reshape(-1)), ns)
    params.write_text(
        "line_follower_node:\n  ros__parameters:\n"
        "    roi_top_ratio: 0.6\n    roi_bottom_ratio: 0.05\n"
        "    use_white_filter: true\n    use_yellow_filter: true\n"
        "    yellow_area_ratio_threshold: 0.005\n    yellow_permanent_count: 4\n"
        "    yellow_clear_frames: 3\n    yellow_stop_duration: 0.8\n"
        "    enable_red_filter: true\n    enable_continuous_curve_assist: true\n",
        encoding="utf-8",
    )
    return bag, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "synthetic_demo",
    )
    parser.add_argument("--storage", choices=("sqlite3", "mcap"), default="sqlite3")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    bag, yaml = make_bag(args.directory, args.storage)
    print(bag)
    print(yaml)

#!/bin/bash
# 辅助录包脚本（在另一个 SSH 终端执行，方便赛场复盘）
set -eo pipefail
TOP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/tros/humble/setup.bash
BAG_DIR="$TOP/recorded_bags"
mkdir -p "$BAG_DIR"
TAG="${1:-run}"
OUT="$BAG_DIR/${TAG}_$(date +%Y%m%d_%H%M%S)"
echo "Recording to: $OUT (Ctrl+C to stop)"
ros2 bag record -o "$OUT"   /image_raw   /hobot_dnn_detection   /cmd_vel   /cmd_vel_follower   /cmd_vel_avoid   /cmd_vel_detection   /cmd_vel_stop   /line_follower/start_cmd   /line_follower/track_state   /line_follower/centroid   /odom   /rosout

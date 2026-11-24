#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAMS_FILE="$ROOT/src/line_follow_ws/src/line_follower/config/line_follower_params_fork.yaml"
source /opt/tros/humble/setup.bash
if [ -f "$ROOT/src/car_ws/install/setup.bash" ]; then
  source "$ROOT/src/car_ws/install/setup.bash"
elif [ -f "/home/sunrise/ws_rdkx5/src/car_ws/install/setup.bash" ]; then
  source "/home/sunrise/ws_rdkx5/src/car_ws/install/setup.bash"
else
  echo "ERROR: car_ws not built yet. Run: bash $SCRIPT_DIR/build_dual_map.sh"
  exit 1
fi
if [ ! -f "$ROOT/src/line_follow_ws/install/setup.bash" ]; then
  echo "ERROR: line_follow_ws not built yet. Run: bash $SCRIPT_DIR/build_dual_map.sh"
  exit 1
fi
source "$ROOT/src/line_follow_ws/install/setup.bash"

LOGDIR="$ROOT/runtime_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
pids=()
cleanup() {
 trap - EXIT INT TERM
 for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
 wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
spawn() {
 local name="$1"; shift
 setsid "$@" >"$LOGDIR/$name.log" 2>&1 &
 pids+=("$!")
}

echo "========================================================"
echo " 启动地图模式: 【岔路地图 Fork Map】"
echo " 仓库路径    : $ROOT"
echo " 参数文件    : $PARAMS_FILE"
echo "========================================================"

spawn chassis ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
spawn camera python3 "$ROOT/src/camera/camera.py"
spawn yolo python3 "$ROOT/src/yolov8/bpu_yolov8_node.py"
spawn follower ros2 launch line_follower line_follower_with_detection.launch.py params_file:="$PARAMS_FILE"
sleep 3
for pid in "${pids[@]}"; do
 if ! kill -0 "$pid" 2>/dev/null; then
  echo "Startup failed. Logs: $LOGDIR"
  tail -n 20 "$LOGDIR"/*.log
  exit 1
 fi
done
echo "Logs: $LOGDIR"
echo '按【空格键 Space】发车/暂停；按【Ctrl+C】退出全部节点。'
ros2 run line_follower start_toggle_node

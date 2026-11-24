#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source /opt/tros/humble/setup.bash
if [ -d "$ROOT/src/car_ws/src" ]; then
  cd "$ROOT/src/car_ws"
  colcon build --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release
  source install/setup.bash
fi
cd "$ROOT/src/line_follow_ws"
colcon build --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release
echo "BUILD_SUCCESS: $ROOT"

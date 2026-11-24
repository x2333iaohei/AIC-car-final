# ROS 2 Bag 视觉分析

在开发电脑上读取既有录包，离线重建比赛 C++ 的视觉处理。默认视频以巡线叠加图为主，右侧保留 Near/Far 采样带和黄线 Mask。工具直接读取 Bag 文件，在进程内反序列化消息，无需启动 ROS 节点、播放 Bag 或连接小车。视觉步骤和参数对应见 [SOURCE_MAP.md](SOURCE_MAP.md)。

## 环境

- **Ubuntu 22.04 / ROS 2 Humble / TROS** 使用系统 Python 3.10、ROS 的 `rosbag2_py`、`rclpy`、`rosidl_runtime_py` 及标准消息包。先 source 本机 ROS 环境，TROS 使用其对应的 setup 脚本。使用与 ROS Python ABI 一致的解释器。使用发行版提供的 OpenCV、NumPy、PyYAML 即可。
- **macOS / Windows 或未安装 ROS 的开发机** 用 Python 3.10+ 建虚拟环境并安装 `requirements-portable.txt`。此选项提供标准 CDR 消息的便携式读取。建议复制整个 Bag 目录，包括 `metadata.yaml` 和所有分片。也可以指定单个 `.db3` / `.mcap`，此时仅分析该分片。

Ubuntu 使用 Humble 常规二进制发行版时，可用以下命令准备工具。RDK 的 TROS 若已提供对应包，使用板端或开发环境现有依赖即可。

```bash
sudo apt install ros-humble-rosbag2 python3-opencv python3-numpy python3-yaml
# 录包为 MCAP 时安装原生存储插件
sudo apt install ros-humble-rosbag2-storage-mcap
source /opt/ros/humble/setup.bash
```

### macOS / Linux 便携环境

在仓库根目录运行。请使用 Python 3.10 或更新版本。

```bash
python3 -m venv tools/rosbag_visualizer/.venv
source tools/rosbag_visualizer/.venv/bin/activate
python -m pip install -r tools/rosbag_visualizer/requirements-portable.txt
```

Windows PowerShell 可直接调用虚拟环境解释器：

```powershell
py -3 -m venv tools/rosbag_visualizer/.venv
& tools/rosbag_visualizer/.venv/Scripts/python.exe -m pip install -r tools/rosbag_visualizer/requirements-portable.txt
```

Windows 上执行下文的渲染命令时，将 `python3` 换成上述解释器路径，并把命令写在一行。读取器接口参考 [Humble rosbag2_py](https://index.ros.org/p/rosbag2_py/) 和 [Rosbags 存储支持](https://ternaris.gitlab.io/rosbags/topics/rosbag2.html)。

## 运行（本地默认路径）

本工具已适配本地目录 `E:\desktop\AIC\rosbag_visualizer`：

- `--output` 可省略，默认自动保存到本地目录 `E:\desktop\AIC\rosbag_visualizer\output\<bag名>_<布局>.mp4`（同名时自动递增 `_001`、`_002`，不会覆盖历史结果）。
- `--params` 可省略，默认自动读取本地 `E:\desktop\AIC\ws_rdkx5\src\line_follow_ws\src\line_follower\config\line_follower_params.yaml`；若使用其他 YAML，仍可通过 `--params` 指定。
- **关于 `/line_follower/debug_image`**：
  - 默认 `--overlay-source auto`：若 Bag 中录制了 `/line_follower/debug_image`，会自动按时间戳配对并在主叠加面板中直接显示车端调试图；
  - 若 Bag 中**没有**录制 `/line_follower/debug_image`，会自动使用 `/image_raw` + 参数 YAML 离线重建完整带巡线算法叠加标记（ROI 融合、白路框、红点目标、分界线、红锥/红线风险标注、连续弯与黄线提示）的调试画面；
  - 若想强制按新 YAML 离线重算叠加图，可加 `--overlay-source reconstructed`。

### 日常回看（Windows PowerShell 一行命令）

```powershell
& E:\desktop\AIC\rosbag_visualizer\.venv\Scripts\python.exe E:\desktop\AIC\rosbag_visualizer\render.py --bag E:\desktop\AIC\<你的bag目录> --layout focus --fps 15 --resolution 1280x800
```

### 排查某一步图像处理（12 格详细视图）

```powershell
& E:\desktop\AIC\rosbag_visualizer\.venv\Scripts\python.exe E:\desktop\AIC\rosbag_visualizer\render.py --bag E:\desktop\AIC\<你的bag目录> --layout 3x4
```

默认自动识别 SQLite3/MCAP，并优先用 `rosbag2_py`。用 `--backend rosbags` 显式选择便携式读取器。用 `--storage sqlite3` 或 `--storage mcap` 指定原生存储。ROS 录制器的 file 压缩模式请用便携式后端或单独解压到一个副本再读。输出路径必须是新的 `.mp4`，避免覆盖已有分析结果。

图像默认来自 `/image_raw`。可用 `--image-topic /camera/image_raw/compressed` 或相应的 `--odom-topic`、`--cmd-topic`、`--state-topic`、`--start-topic`、`--centroid-topic`、`--log-topic` 改用实际话题。原始图像支持 BGR8、RGB8、MONO8、BGRA8、RGBA8，另支持普通 JPEG/PNG 压缩彩色图。YAML 支持全局 `/**` 和匹配指定 `--node` 的具名 ROS 参数段。用 `--layout 3x4` 查看完整的 12 步处理细节。自定义网格采用行×列，至少留 12 个格子。分辨率采用宽×高，二者需为偶数。

## 输出和口径

先从 MP4 或 `keyframes/` 定位问题时刻，再用 `events.csv` 核对黄线与停车事件，用 `speed.png` 看前后速度是否下降。需要定位到单帧时查 `frames.csv`。

| 文件 | 内容 |
|---|---|
| `analysis.mp4` | 重点视图显示巡线叠加图、采样带、黄线 Mask、停车状态与速度。用 `--layout 3x4` 查看原图、ROI、高斯、HSV、白路 Mask、开、闭、黄线 Mask、红色 Mask、轮廓、Near/Far 与完整调试叠加图。 |
| `analysis_analysis/keyframes/*.png` | 分段起始、录制黄线触发和离线候选等关键画面，默认最多 48 张。 |
| `analysis_analysis/frames.csv` | 每张实际处理的源图像一行，含原始时间、对齐时间、录制控制量、采样带误差和重建目标。目标像素为整图坐标。 |
| `analysis_analysis/speed.png`、`speed.csv` | 录制的指令和实测速率曲线，以及完整线速度、角速度样本。 |
| `analysis_analysis/events.csv` | ROS 日志黄线触发、跟踪状态切换、视觉候选及可计算的制动指标。 |
| `analysis_analysis/report.md` | 话题覆盖、时间策略、数据来源、参数 SHA-256、解码错误计数和指标限制。 |

图像和有时间戳的遥测默认优先采用非零 header。header 与记录时间差大于 `--max-header-skew`（默认 2 秒）时回退到 Bag 记录时间，可用 `--clock bag` 让所有话题采用后者。无 header 的普通 Twist、启停、状态和质心采用记录时间。每张图使用该时刻及之前的最新遥测，速度和质心超过 `--sync-max-age` 就留空。跟踪状态只在变化等情况下发布，因此显示最后录制状态和年龄。没有启停记录时，画面仍可分析，但无法还原控制器黄线计数，红色路径会标注启停未知。

图像逐帧处理，MP4 再按指定 FPS 选帧。重复时间戳丢弃，同一段内出现间隙会按输出 FPS 持有最多若干帧。时间倒退或大于 `--frame-gap` 的空洞会分段，并重置重建的连续弯和黄线锁存。分析时间从 CSV 读取。Bag 中途开始、画面丢失、实际参数不同及 OpenCV 构建差异可能造成目标点与车端调试图不同。

制动时长以录制的前进→零速或负速指令为起点，到 Odom 首个满足持续低线速、低角速度样本的时刻为止。报告同时给出有符号位移和仅前向行程，由 Odom 线速度积分估算。只有录制日志事件或缺少对应日志的状态跳变能关联该指标。若缺少前置指令、Odom 断采、证据窗口过短、指令提前恢复，则留空并给出原因。积分结果是**刹车期间车体行程估计**，没有相机标定和物理参照时无法得到车头距黄线的间距。

## 生成合成 Bag 和测试

安装便携式依赖后，在仓库根目录运行：

```bash
python3 tools/rosbag_visualizer/synthetic.py --directory /tmp/aic-bag-demo --storage sqlite3
python3 tools/rosbag_visualizer/render.py \
  --bag /tmp/aic-bag-demo/synthetic_sqlite3 \
  --params /tmp/aic-bag-demo/synthetic.yaml \
  --output /tmp/aic-bag-demo/demo.mp4
cd tools/rosbag_visualizer
python3 -B -m unittest discover -s tests -v
```

合成 Bag 使用 ROS 2 CDR 序列化。自动化测试覆盖 ROI 高斯边界、HSV 与灰度互斥、开闭运算、黄线面积、红色路径、连续反向弯、时间对齐、制动积分及 SQLite3/MCAP 的 MP4 和 CSV 输出。ROS 原生后端测试需要配置 Humble/TROS。

## 实车录包验证

已使用一份 1,368 帧的实车 SQLite3 Bag 和随包保存的参数快照，在 macOS 便携环境渲染重点视图及 12 格视图。按录制顺序与 `/line_follower/centroid` 配对时，该样本的重建目标在 X、Y 两轴都与录制值相差不超过 1 px。这个结果适用于该录包与参数快照，其他版本的录包需要重新对照。

## 文件

- `vision.py`、`params.py` 对应视觉处理和 ROS 参数载入。
- `bag.py`、`telemetry.py` 负责读取、磁盘索引、对齐以及事件指标。
- `render.py`、`panels.py` 负责视频、面板、曲线、CSV 和报告。
- `synthetic.py`、`tests/` 提供合成数据和回归测试。

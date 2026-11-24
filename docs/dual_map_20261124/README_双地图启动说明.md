# 全国总决赛 RDK X5 双地图（环岛地图 / 岔路地图）运行与调参说明

> **安全隔离说明**：本目录（默认部署于 `/home/sunrise/ws_aic_dual_20261124`）与车上原有 `/home/sunrise/ws_rdkx5` 完全独立，**不覆盖、不修改原有代码**。如需跑回旧版代码，仍执行原来的 `/home/sunrise/ws_rdkx5/start-original.sh` 即可。

---

## 一、目录结构

```text
ws_aic_dual_20261124/
├── README.md                  # 本说明文件
├── build_all.sh               # 一键编译环岛+岔路两套工作区（首次在新车部署时运行一次）
├── start_ring.sh              # 【环岛地图】一键比赛启动脚本
├── start_fork.sh              # 【岔路地图】一键比赛启动脚本
├── record_bag.sh              # （可选）新开终端一键录包脚本
├── ws_rdkx5_ring/             # 环岛地图独立工程（默认已内置环岛参数）
├── ws_rdkx5_fork/             # 岔路地图独立工程（默认已内置岔路参数）
└── docs/                      # 两份地图修改计划与参数备份
```

---

## 二、快速启动步骤（严格区分两张地图！）

> 启动前请确保没有其他终端正在运行旧的底盘、相机或巡线进程。

### 1. 跑【环岛地图（Ring Map）】

SSH 登录小车后直接执行：

```bash
bash /home/sunrise/ws_aic_dual_20261124/start_ring.sh
```

- 等待约 3 秒看到提示：`按【空格键 Space】发车/暂停；按【Ctrl+C】退出全部节点。`
- 把车摆在起点黄线前，**按一下 `空格键` 发车**；
- 跑完第 4 条黄线（`END`）小车会自动永久刹停；再按 `Ctrl+C` 干净退出所有节点。

---

### 2. 跑【岔路地图（Fork Map）】

SSH 登录小车后直接执行：

```bash
bash /home/sunrise/ws_aic_dual_20261124/start_fork.sh
```

- 同样按 **`空格键`** 发车/暂停，按 **`Ctrl+C`** 一键退出全部节点。

---

## 三、发给队友部署到其他小车的方法

将压缩包 `AIC_双地图定稿代码_20261124.zip` 上传到目标小车 `/home/sunrise/` 后执行：

```bash
cd /home/sunrise
unzip AIC_双地图定稿代码_20261124.zip
cd /home/sunrise/ws_aic_dual_20261124
bash build_all.sh
```

编译完成后即可直接使用 `bash start_ring.sh` 或 `bash start_fork.sh`（小车本地若由脚本预编译完成，则无需再次编译）。

---

## 四、两套地图核心区别与现场微调速查

> **注意**：`start_ring.sh` 和 `start_fork.sh` 已配置为**直接读取 `src` 下的 YAML 参数文件**，现场修改 YAML 参数后**不需要重新 `colcon build`**，直接重启 `start_ring.sh` / `start_fork.sh` 即可立即生效！（只有修改 `.cpp/.hpp` 源码才需运行 `build.sh`。）

| 地图 | 参数文件位置 | 已开启特性 | 现场微调速查项 |
|---|---|---|---|
| **环岛地图** | `ws_rdkx5_ring/src/line_follow_ws/src/line_follower/config/line_follower_params.yaml` | 四级仲裁 `stop > avoid > detection > follower`、`35x13` 镂空红色禁行区聚类、`kd=0.28` 阻尼、关闭激进远端拉扯与丁字并线 | 1. 分岔转角深浅：`turn_duration: 0.44`（`±0.05`）<br>2. 禁行区避让幅度：`red_line_soft_yaw: 0.28`（`0.24~0.34`） |
| **岔路地图** | `ws_rdkx5_fork/src/line_follow_ws/src/line_follower/config/line_follower_params.yaml` | 在环岛全部修复基础上，额外开启：**单行道急弯远端预瞄**（`enable_curve_preview_steering: true`）+ **内支路末端 `┤` 丁字口顺行闭环右转并线**（`enable_tjunction_merge: true`） | 1. 丁字口右转出弯角度：若略欠调大 `tjunction_merge_yaw_rate: 1.20 -> 1.30` 或 `tjunction_exit_upper_road_ratio: 0.12 -> 0.16`；若略多调小至 `1.10`<br>2. 分岔口切入时长：`turn_duration: 0.52`（`±0.05`） |

### （可选）现场试跑录包方法
保持 `start_ring.sh` 或 `start_fork.sh` 终端运行，**新开第二个 SSH 终端**执行：
```bash
bash /home/sunrise/ws_aic_dual_20261124/record_bag.sh ring_test
```
录制的包会保存在 `/home/sunrise/ws_aic_dual_20261124/recorded_bags/` 下。

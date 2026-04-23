# LeRobotDataset 格式与数据采集规范

> 本项目统一使用 LeRobotDataset v2 格式，三种策略（ACT / Diffusion Policy / π0）共享同一数据管线。

---

## 1. 数据格式

### 1.1 目录结构

```
dataset/
├── README.md               # 本文件
└── <task_name>/
    ├── meta/
    │   ├── info.json        # 数据集元信息
    │   ├── episodes.jsonl   # Episode 索引
    │   └── stats.json       # 归一化统计量
    ├── data/
    │   ├── chunk-000/
    │   │   ├── episode_000000.parquet
    │   │   ├── episode_000001.parquet
    │   │   └── ...
    │   └── chunk-001/
    └── videos/
        └── chunk-000/
            ├── episode_000000.mp4
            └── ...
```

### 1.2 Parquet 列定义

| 列名 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `timestamp` | float64 | (1,) | 时间戳（秒） |
| `observation.state` | float32 | (14,) | 关节位置 (7) + 关节速度 (7) |
| `observation.images.top` | uint8 | (480, 640, 3) | 顶部相机 RGB |
| `observation.images.wrist` | uint8 | (480, 640, 3) | 腕部相机 RGB（可选） |
| `action` | float32 | (7,) | 6-DoF 末端位姿增量 + 夹爪开合 |
| `episode_index` | int64 | (1,) | Episode 编号 |
| `frame_index` | int64 | (1,) | 帧编号（Episode 内） |

### 1.3 info.json 关键字段

```json
{
  "codebase_version": "v2.1",
  "robot_type": "ur5",
  "fps": 10,
  "total_episodes": 100,
  "total_frames": 50000,
  "features": {
    "observation.state": {"dtype": "float32", "shape": [14]},
    "action": {"dtype": "float32", "shape": [7]}
  }
}
```

---

## 2. Action Chunk Size 标准化

三种策略对 action chunk size 有不同的最优设置，数据采集时**按单步记录**，训练时由 DataLoader 按策略配置组装 chunk：

| 策略 | action_chunk_size | 说明 |
|------|-------------------|------|
| ACT | 20 | CVAE 预测 20 步，temporal ensemble 平滑 |
| Diffusion Policy | 16 | 预测 16 步，执行前 8 步后重规划 |
| π0 | 50 | VLA 长视野预测，一次执行全部 |

**关键约定**：
- 数据集中 `action` 列始终为**单步动作** (shape = [7])
- chunk 组装在训练/推理时通过滑动窗口完成
- 确保 episode 长度 >= max(action_chunk_size) = 50 帧

---

## 3. 数据采集流程

### 3.1 硬件要求

- 机械臂：UR5 / Franka Emika（配 Robotiq 2F-85 夹爪）
- 相机：Intel RealSense D435（顶部 + 腕部）
- 采集频率：10 Hz
- 遥操作：3DConnexion SpaceMouse 或示教器

### 3.2 采集步骤

1. **校准**：相机外参标定，手眼标定
2. **遥操作采集**：操作员完成任务演示，系统同步记录关节状态、末端位姿、相机图像
3. **质量检查**：
   - 过滤失败轨迹（任务未完成）
   - 检查时间戳连续性（帧间隔应为 100ms ± 10ms）
   - 确认图像无遮挡/模糊
4. **格式转换**：原始 ROS bag / HDF5 → LeRobotDataset v2 Parquet
5. **统计量计算**：运行归一化脚本生成 `stats.json`

### 3.3 数据量建议

| 任务复杂度 | 最少 Episodes | 推荐 Episodes | 备注 |
|-----------|---------------|---------------|------|
| 简单（reach, push） | 25 | 50 | π0 可用 25 条 |
| 中等（pick-place） | 50 | 100 | Diffusion Policy 推荐 50+ |
| 复杂（peg-insert） | 100 | 200 | ACT 需要较多数据 |

---

## 4. 归一化

### 4.1 stats.json 格式

```json
{
  "observation.state": {
    "mean": [0.1, ...],
    "std": [0.05, ...],
    "min": [-1.0, ...],
    "max": [1.0, ...]
  },
  "action": {
    "mean": [0.0, ...],
    "std": [0.02, ...],
    "min": [-0.1, ...],
    "max": [0.1, ...]
  }
}
```

### 4.2 归一化策略

- **观测**：z-score 标准化（减均值除标准差）
- **动作**：min-max 归一化到 [-1, 1]
- 统计量基于训练集计算，评估时复用

---

## 5. 数据验证清单

- [ ] Parquet 文件可被 `pandas.read_parquet()` 正常读取
- [ ] `action` shape 一致为 (7,)
- [ ] `observation.state` shape 一致为 (14,)
- [ ] 每个 episode 帧数 >= 50
- [ ] `timestamp` 单调递增，间隔约 0.1s
- [ ] `stats.json` 中无 NaN 或 Inf
- [ ] 图像分辨率与配置文件一致

---

*最后更新：2026-04-23*

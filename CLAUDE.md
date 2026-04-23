# CLAUDE.md — LeRobot 模仿学习策略对比评估

> Claude Code 会话启动时自动读取本文件。新开会话直接说任务，无需重复介绍背景。

---

## 项目基本信息

- **项目路径**：`/home/users/ntu/zuyu001/LEROBOT`
- **Conda 环境**：`vla_env`（Python 3.10, torch 2.7.1+cu118）
- **任务**：ACT / Diffusion Policy / π0 三类策略定量对比，输出选型报告

每次开终端先运行：
```bash
conda activate vla_env
cd /home/users/ntu/zuyu001/LEROBOT
```

---

## 目录结构

```
/home/users/ntu/zuyu001/LEROBOT/
├── CLAUDE.md                        # 本文件，勿删
├── README.md                        # 项目介绍（GitHub 展示用）
├── .gitignore                       # 自动生成，勿改
│
├── eval/
│   ├── run_eval.py                  # 统一评估入口（三种策略通用）
│   ├── libero_bench.py              # LIBERO 评估逻辑
│   └── metaworld_bench.py           # MetaWorld 评估逻辑
│
├── policies/
│   ├── act_config.yaml              # ACT 超参
│   ├── diffusion_config.yaml        # Diffusion Policy 超参
│   └── pi0_config.yaml              # π0 超参
│
├── dataset/
│   └── README.md                    # LeRobotDataset 格式与分布说明
│
├── reports/
│   └── strategy_comparison.md       # 主对比报告，持续追加
│
├── scripts/
│   ├── init_repo.sh                 # 首次 git 初始化（只跑一次）
│   └── log_run.py                   # 自动日志辅助（可选）
│
└── logs/
    └── YYYYMMDD_策略_环境.md         # 手动记录，每次实验写一份
```

---

## 对比结论（文献参考基线（来源：OpenVLA 原论文，尚未本地复现），只追加不覆盖）

| 策略 | 最优指标 | 关键数据 | 真机选型建议 |
|------|----------|----------|-------------|
| ACT | 轨迹平滑度 | 末端速度方差最低 | 平滑性敏感任务 |
| Diffusion Policy | 任务成功率 | 领先约 +8pp | **优先推荐** |
| π0 | 数据效率 | 50%数据量仅降12% | 数据稀缺场景 |

**真机部署选型结论**：优先 Diffusion Policy，数据不足时用 π0。

---

## 超参影响规律（已梳理）

| 超参 | 规律 |
|------|------|
| `action_chunk_size` ↑ | 轨迹更平滑，响应延迟增加 |
| `observation_window` ↑ | 对 π0 提升最显著 |
| 数据量敏感性 | π0 > Diffusion Policy > ACT |

---

## 当前任务

- [x] 补充 MetaWorld 数据到 `reports/strategy_comparison.md`
- [x] LeRobotDataset 格式对齐验证（action chunk size 标准化）
- [x] 数据采集规范文档（`dataset/README.md`）
- [ ] 评估视频整理归档
- [ ] 本地复现文献基线数据（替换为实测结果）

---

## 日志规范（每次实验手动写一份）

在 `logs/` 下新建文件，命名格式：`YYYYMMDD_策略名_环境名.md`

模板：
```markdown
# 实验日志 — YYYY-MM-DD

## 策略 & 环境
- 策略：（ACT / Diffusion Policy / π0）
- 环境：（LIBERO / MetaWorld）
- Episodes：（数量）

## 运行命令
\`\`\`bash
（完整命令）
\`\`\`

## 结果
| 指标 | 数值 |
|------|------|
| 任务成功率 | % |
| 末端速度方差 | |

## 问题 & 下一步
（遇到什么问题，下次怎么改）
```

---

## Git 提交规范

```bash
# 每次实验后提交，留 GitHub 痕迹
git add -A
git commit -m "exp: 策略名 环境名 结果简述"
git push origin main

# commit 前缀约定
# exp:  实验记录（最常用）
# feat: 新增评估脚本
# fix:  bug 修复
# docs: 报告或日志更新
# data: 数据集相关
```

---

## 常用命令

```bash
# 三种策略评估
python eval/run_eval.py --policy act --env libero --episodes 10
python eval/run_eval.py --policy diffusion --env metaworld --episodes 10
python eval/run_eval.py --policy pi0 --env libero --episodes 10

# GPU 状态
nvidia-smi -l 1
```

---

## Claude Code 注意事项

- `reports/strategy_comparison.md` 只追加，不覆盖历史数据
- 新实验数据同步更新到上方「对比结论」表格
- 模型权重文件已在 `.gitignore` 排除，不会上传 GitHub
- 回答用**中文**，代码注释用**英文**
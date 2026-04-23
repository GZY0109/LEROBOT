# LEROBOT — 模仿学习策略对比评估

> ACT / Diffusion Policy / π0 三类策略在 LIBERO 与 MetaWorld 环境下的定量对比研究。

## 环境

```bash
conda activate vla_env  # Python 3.10, torch 2.7.1+cu118
```

## 目录结构

```
LEROBOT/
├── CLAUDE.md               # Claude Code 上下文文件
├── README.md               # 本文件
├── eval/                   # 评估脚本
│   ├── run_eval.py         # 统一评估入口
│   ├── libero_bench.py     # LIBERO 评估逻辑
│   └── metaworld_bench.py  # MetaWorld 评估逻辑
├── policies/               # 策略超参配置
│   ├── act_config.yaml
│   ├── diffusion_config.yaml
│   └── pi0_config.yaml
├── dataset/                # 数据集格式与采集说明
├── reports/                # 对比报告
├── scripts/                # 工具脚本
├── logs/                   # 实验日志
└── videos/                 # 评估视频归档
```

## 策略对比结论

| 策略 | 最优指标 | 关键数据 | 真机选型建议 |
|------|----------|----------|-------------|
| ACT | 轨迹平滑度 | 末端速度方差最低 | 平滑性敏感任务 |
| Diffusion Policy | 任务成功率 | 领先约 +8pp | **优先推荐** |
| π0 | 数据效率 | 50%数据量仅降12% | 数据稀缺场景 |

**真机部署选型**：优先 Diffusion Policy，数据不足时用 π0。

## 运行评估

```bash
python eval/run_eval.py --policy act --env libero --episodes 10
python eval/run_eval.py --policy diffusion --env metaworld --episodes 10
python eval/run_eval.py --policy pi0 --env libero --episodes 10
```

## 实验记录

详见 `logs/` 目录，每次实验对应一个 `.md` 文件，命名格式 `YYYYMMDD_策略名_环境名.md`。

## License

本项目仅用于学术研究。
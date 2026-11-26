# 实验产物的保存与整理

Git 中保留什么，需要同时考虑复现、证据和仓库体积。当前仓库已跟踪许多历史 run、MLflow 文件、会话数据库和较大的预测数据；它们可能被报告引用，不能简单按目录全部删除。

| 类别 | 建议 | 当前处理 |
| --- | --- | --- |
| 源码、配置、协议、测试 | 保留在 Git | 保留 |
| 支撑结论的汇总表、图、检查点、少量原始轨迹和哈希 | 核对报告引用后保留 | 已有文件保留 |
| 大型逐窗口预测、中间数组、可重建的重复结果 | 先制作清单、核对引用和恢复方式，再考虑 GitHub Release、Git LFS 或其他数据存储 | 本次未迁移、未删除 |
| 新生成的本地 run、MLflow、Agent session 和 SQLite 状态 | 默认留在本机；需分享时选择性检查并强制添加 | 新增 `.gitignore` 规则；已跟踪文件不自动移除 |
| `*:Zone.Identifier` | 不作为项目数据 | 本次仅将已跟踪标记移出 Git 索引，保留本地文件 |

以后若某次 run 必须公开作为证据，先检查是否包含凭据、敏感路径或无关信息，再选择必要文件。例如可用 `git add -f reports/runs/<run-id>/config.yaml` 强制添加被忽略的文件；同时检查相关报告是否需要其原始轨迹与哈希清单，避免只上传空壳。不要通过删除原始数据来伪造实验完整性。

新增 `.gitignore` **不会取消已跟踪文件，也不会压缩 Git 历史**。即使以后新提交不再包含某个大文件，它仍可能存在于以前的提交中。历史重写或强制推送会影响其他克隆和现有引用，本次没有执行；需要时先独立评估并备份。

`external/` 中的第三方源码、BOPTEST、下载的模型权重和构建缓存不自动上传；它们的版本、来源与获取方式见 [复现指南](setup_reproduction.md) 和 [第三方来源](third_party.md)。

## 当前需要优先评估的大文件

以下大小为本次整理时的近似值，文件仍在 Git 中；移动前应检查对应报告、复现脚本和校验清单。不要只移动文件而不更新引用、下载方式和 SHA256。

| 文件 | 约 | 建议 |
| --- | ---: | --- |
| `reports/week5/results/projection/rollout/metrics_by_episode.csv` | 80 MiB | 逐 episode 明细；确认审核需求后可考虑外置，汇总表继续保留 |
| `reports/week5/results/rollout/metrics_by_episode.csv` | 58 MiB | 同上，不能假设有汇总表就可以丢掉 |
| `reports/week5/results/projection/rollout/metrics_by_episode_across_seeds.csv` | 40 MiB | 先核查被哪些脚本和报告引用 |
| `reports/week5/results/rollout/metrics_by_episode_across_seeds.csv` | 32 MiB | 先核查被哪些脚本和报告引用 |
| `reports/week1/week1_results/predictions.csv` | 30 MiB | 评估是否需要整份预测作为可追溯证据，再决定是否外置 |
| `reports/week6/results/open_loop/raw_predictions.npz` | 25 MiB | 原始开环预测，若外置需保留版本、下载方法和校验值 |

GitHub 对普通 Git 中超过 50 MiB 的文件给出警告，超过 100 MiB 的文件会被阻止；具体限制与外置方式见 [GitHub 官方文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。本次保留这些文件，避免影响已有结果复核。

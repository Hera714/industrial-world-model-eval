# 数据、模型与第三方代码来源

项目自身尚未声明代码许可证；此文件不替项目作者作出授权选择。以下第三方资产的条款分别适用，引用前请查看对应上游文件和实际使用的版本。

| 资产 | 用途和位置 | 来源与许可线索 |
| --- | --- | --- |
| Appliances Energy Prediction | Week 1 真实楼宇数据；`data/raw/energydata_complete.csv` | [UCI 数据集页面](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) 标注 CC BY 4.0。引用：Candanedo, L. (2017), *Appliances Energy Prediction*, UCI Machine Learning Repository, [DOI:10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G)。页面说明该数据集包含来自 Reliable Prognosis 的天气数据。 |
| BOPTEST | Week 3、6–8 仿真平台；源码和 FMU 不由项目 Git 仓库分发 | [上游 v0.9.0](https://github.com/ibpsa/project1-boptest/tree/v0.9.0)，[许可原文](https://github.com/ibpsa/project1-boptest/blob/v0.9.0/license.md)。固定 commit 和案例文件哈希见 `reports/week6/results/assets/manifest.json`。本地构建补丁包含对上游文件的修改，需连同上游许可条件审阅。 |
| Qwen2.5-1.5B-Instruct | Week 8 本地 Agent；权重不纳入项目 Git 仓库 | [上游模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)标注 Apache 2.0。模型 revision 和文件哈希见 `configs/agent/local_model_manifest.json`。 |
| PatchTST | Week 1 对照；上游源码目录不纳入本仓库 | [官方代码与许可证](https://github.com/yuqinie98/PatchTST)。仓库中的相关模型代码、检查点和结果应分别核对其来源及适用条款。 |
| MinIO 与 MinIO Client | BOPTEST 历史本地构建所需对象存储工具；源码目录和镜像不纳入本仓库 | 记录的上游 commit 见 `deployment/boptest/minio_sources.json`；具体源码、镜像及其许可证请以所使用版本的上游声明为准。 |

Week 2 自建热工数据及 Week 6 BOPTEST 采集结果是本项目的实验产物；它们的生成方式、版本与局限见各周报告。上面的模型或数据许可不自动成为本项目其他代码的许可证，也不替代引用、版权或其他上游要求。

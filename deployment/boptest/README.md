# BOPTEST 构建修改归档

固定上游版本和 FMU 摘要见 reports/week6/results/assets/manifest.json。

在相同上游 commit 的干净 checkout 中运行 `git apply deployment/boptest/local-build.patch`（补丁路径须相对于当前目录正确解析）。补丁包含本地 APT 镜像、非交互时区和本地 Miniconda 校验改动。

补丁引用的 apt-seed 与 installers 文件未纳入 Git；manifest 列出路径、大小和 SHA256，重建前必须准备相同文件。现有 Docker 镜像并非该补丁本身，MinIO 自建镜像也需单独保留来源和镜像检查结果。禁止声称只靠此补丁即可离线重建全部镜像。

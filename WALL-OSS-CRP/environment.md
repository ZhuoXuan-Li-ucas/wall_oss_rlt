# WALL-OSS-CRP：Docker 环境验证与迁移

> **当前状态入口：[PROJECT_STATUS.md](PROJECT_STATUS.md)**。2026-10-29，D0 基础包、训练候选包和 RTX 5070 Ti 推理源码工具包均已在 My Passport 完成打包与校验。当前优先由用户在 Ubuntu `crp-5070ti-01` 本地离线构建派生镜像，再验证官方 WALL 完整推理；本机不连接 H100 或目标推理主机。完整 WALL＋RLT 尚未最终封版。

推理主机已确认：RTX 5070 Ti，16303 MiB 显存，驱动 595.91.07，工作目录 `/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt`。`nvidia-smi` 的 CUDA 13.2 是驱动兼容上限，不是容器工具链版本。方案为 D0 派生的 CUDA 12.8／Torch 2.10.0+cu128 镜像，离线重编 FlashAttention 与 WALL 的 `sm_120` 扩展；目标镜像构建、5070 Ti GPU 内核和完整推理记录为 `NOT_RUN`，尚无目标机验收报告。尚未记录用户拷贝／上传完成回执。操作见 [5070 Ti 本地推理部署](WALL_RLT_Inference_Deployment.md)。

本轮实现检查 **377 项 CPU 测试 PASS**，依据 [CPU 报告](logs/rlt-inference-20261029/cpu-check/report.json) 与 [推理交付记录](metadata/rlt-inference-deployment-20261029.json)。官方 WALL 权重已在 D0 提供；已训练 WALL＋RLT 与 RL Actor 权重仍缺失。训练候选冻结时的 **218 项 CPU 测试**是该旧包的历史证据，不等同于本轮 377 项，也不表示 H100 训练通过。双卡 FSDP 候选配置仍待实测。数据已经采集完成，不安排试采或扩采。

H100 后续交付链路（2026-10-29 用户确认）为 **本机打包 → My Passport → 用户自己的电脑 → 用户连接专用网线 → SSH/SCP 上传 H100**；当前 5070 Ti 分支由用户直接将移动硬盘文件拷到推理主机并本地执行。本机无需连接 H100，既有连接超时记录不阻塞硬盘交付。用户确认服务器资源完备，共 **8 张 H100**；此记录来自用户说明，未远程探测，也未指定本次作业使用全部 8 卡或每卡显存。H100 承担训练与后续复验，不是当前优先部署的推理端。

My Passport 上的 D0 基础环境包 `WALL-OSS-CRP-transfer-20261029` 和训练候选包 `WALL-RLT-training-candidate-20261029-v1` 合计 **22,568,731,444 字节（约 22.57GB）**；这个大小及硬盘根目录 `WALL-H100-SHA256SUMS-20261029.txt` 的范围仅为这两个既有包。训练候选提供目标数据、固定源码和 44 个离线依赖 wheel，排除 mouse／tape；训练覆盖层与 GPU 训练仍需在 H100 验证。

新增的 `WALL-RLT-inference5070-20261029-v1` 已校验，共 **8 个文件、10,979,923 字节**。它包含离线编译源码和推理工具，不含基础镜像、模型权重或录制数据；自身校验入口为 `package_wall_rlt_inference.py verify`。移动硬盘上的 `WALL-5070Ti-开始部署-20261029.txt` 是本次操作入口，旁置部署说明和实现审阅文档供用户阅读。三个包都是不可变交付快照，本仓库后续状态文档更新不回写这些包。

## D0 基础包既有证据与未完成项（2026-10-29）

以下记录对应 D0 构建与导出阶段，保留其真实验证范围和失败历史；不表示新增 5070 Ti 派生镜像已构建或已在目标 GPU 运行。

- Docker 镜像全离线构建 PASS；在本机 RTX 3060 Ti 的断网容器中，CUDA BF16、FlashAttention、Wall-X 扩展导入、processor 与归一化检查 PASS；24 个模型文件重新校验 PASS。报告：[setup-verification.json](logs/docker-validation-delivery-20261029/setup-verification.json)。
- 硬盘验证集 `H100_lerobot/Datasets/stack_the_base_llx_04251`，32 条、15376 帧、337408840 字节；LeRobot＋PyAV 读取首／中／尾样本（索引 0、7688、15375）PASS。两路图像均可读取；这不是全部视频逐帧质检，也不是 WALL 训练 batch 或 AR5 数据验收。报告：[dataset-verification.json](logs/docker-validation-delivery-20261029/dataset-verification.json)。
- FlashAttention 含 `sm_80`／`sm_90`，Wall-X 扩展含 `sm_86`／`sm_90`；两项 `sm_90` 静态架构覆盖检查 PASS。此项不执行 H100 内核，不能代替目标服务器运行验收。报告：[native-architectures.json](logs/docker-validation-delivery-20261029/native-architectures.json)。
- 容器与原环境的 `pip check` 仍 FAIL：10 项 LeRobot 缺失声明依赖和 1 项 Hub 版本约束冲突。新容器已执行依赖检查，`unexpected=[]`，仅有这 11 项已知例外；允许基础验证继续，不表示完整 LeRobot 或训练依赖通过。报告：[dependencies.json](logs/docker-validation-delivery-20261029/dependencies.json)。
- Docker Engine 29.9.2 与 NVIDIA Container Toolkit 1.20.1-1 已安装，断网容器 GPU 透传通过（E1）；记录为 `logs/docker-host-install-offline-retry-20261028.log`。
- 已从移动硬盘旧归档中恢复并导入 CUDA 12.8.1／Ubuntu 22.04 的 11 个 NVIDIA 官方原始基础层，逐层与官方 manifest／config 摘要匹配；未引入旧项目层。`nvcc 12.8.93` 与 GPU 容器透传检查 PASS。
- 离线 wheel 缓存共 126 个、4,588,605,200 字节；完整离线依赖解析及 PyTorch 2.10.0+cu128 等依赖安装已完成，原始 wheelhouse 仍保留在 My Passport。
- 原 APT 系统阶段因 HTTP 502 失败，现已改用从 25 个本地 Conda 基础包新建的 Python 3.10.22 bootstrap，重定位到 `/opt/walloss/env`；归档与重定位检查通过。官方 CUDA devel 基础镜像提供编译工具，当前 Dockerfile 不再调用 APT。
- FlashAttention 2.8.3／Wall-X 编译、基础运行检查、归档身份、gzip 与整包 SHA256 检查已完成；新交付目录已发布 `BUNDLE_READY.json`，任务状态为 `BUNDLE_READY`。
- Windows 校验为 `NOT_REQUIRED`；D0 交付记录中的 SSH 上传为 `server_transfer=NOT_STARTED`，目标 H100 验证、完整模型／WALL-RLT 推理与训练为 `NOT_RUN`。该基础包验收不覆盖 AR5 数据适配、闭环或在线 RLT；后续适配代码的 CPU 验证见顶部本轮报告，不能追记为 D0 的 GPU 验收。
- 机器可读结论：[server-environment-delivery-20261029.json](logs/server-environment-delivery-20261029.json)；迁移状态：[migration-preflight.json](metadata/migration-preflight.json)。原始预检日志目录：`logs/migration-preflight-20261028T064203Z/`。

**已就绪的完整基础包**：[WALL-OSS-CRP-transfer-20261029](</media/pangk/My Passport/WALL-OSS-CRP-transfer-20261029>)。目录统计为 **16,787,299,846 字节（约 16.79 GB／15.63 GiB）**；镜像归档 9,879,516,707 字节、模型归档 6,570,912,728 字节、历史数据归档 336,602,786 字节。完成标记于 2026-10-29 06:14:55 UTC 发布，记录 27 个清单文件和 24 项必需文件；`SHA256SUMS` 摘要为 `9dbf56cb50f464339ae1053f91934e8a98c3857b901b8d8d81f7061b7b537f10`。目录内另有清单与完成标记，共 29 个文件。实际交付身份以 [BUNDLE_READY.json](</media/pangk/My Passport/WALL-OSS-CRP-transfer-20261029/BUNDLE_READY.json>) 为准。

源镜像为 `sha256:8f067a0eb220228f444811532cc8ec0a328e4d46b7983637c1c4ea7743ef9ee8`，迁移标签为 `docker.io/library/wall-oss-crp-transfer:validated-8f067a0eb220`。本次成功报告目录为 [docker-validation-delivery-20261029](logs/docker-validation-delivery-20261029)，[任务 JSON](logs/docker-bundle-job-20261028.json) 为 `BUNDLE_READY`；`walloss-bundle-20261029-export.service` 已成功退出（inactive，exit 0），见 [导出流水线日志](logs/docker-pipeline-export-20261029.log) 和 [完整导出日志](logs/docker-bundle-job-20261028.log)。

两次交付阻塞均已解决：镜像构建成功后数据读取因根盘空间不足失败，精准回收 BuildKit 的 `wall_wheels` 输入缓存 `gr9locm6pqyhaksphubtmh2d8`（4.595 GB），保留外盘原始 wheelhouse，再复用上述精确镜像 ID 重跑全部运行验证，未重建镜像。导出随后因两份 31 行资产清单的 locale 排序不同失败；路径、哈希与重复次数一致，现改为 `LC_ALL=C` 排序后比较，并固定后续清单生成顺序。负例检查见 [asset-manifest-order-tests-20261029.json](logs/asset-manifest-order-tests-20261029.json)。临时 swap 已移除，核查时 `/proc/swaps` 无启用条目。

旧 `/media/pangk/My Passport/WALL-OSS-CRP-transfer-20261028` 保留为 **INCOMPLETE，不可部署**，没有完成标记；本次使用新目录导出成功，没有覆盖失败目录。原始构建及磁盘失败报告保留在 `logs/docker-validation-final-20261029T011859Z/`，后续复用来源绑定见 [build-reuse.json](logs/docker-validation-delivery-20261029/build-reuse.json)；更早的 `logs/docker-validation-local-20261028T094112Z/` 和 `logs/docker-system-20261028.log` 分别保留中断与 APT 失败证据。先行资产目录 `/media/pangk/My Passport/WALL-OSS-CRP-assets-20261028` 只含模型／历史数据，已复核并用于本次完整包，不能单独作为环境部署包。

## D0 与 H100 迁移参考

本节及下面第 1～5 节保留 D0 复现和 H100 后续验收命令。当前 5070 Ti 部署使用顶部链接的独立说明；不要在 5070 Ti 上运行要求 `sm_90` 的 H100 验收脚本。

交付 Docker 镜像归档，模型与数据单独归档并在容器运行时只读挂载。HF datasets／Torch／Numba 等临时缓存位于容器可写的 `/tmp/walloss-*`，支持以服务器普通用户 UID 运行。用户的电脑负责从 My Passport 拷贝、再经专用网线上传原样归档，无需解包或安装 Docker；电脑系统未指定，下面的相对路径上传命令可在提供 ssh／scp 的终端使用。环境部署与运行验证在 Ubuntu 服务器进行。

原本地环境是 PyTorch 2.10.0+cu130；H100 历史记录为驱动 570.86.10。NVIDIA 文档中 CUDA 13.x 的常规最低驱动为 580，不能默认套用原环境。因此本次镜像使用 CUDA 12.8／Ubuntu 22.04／Python 3.10.22，保持 PyTorch 2.10.0、Transformers 5.2.0 等主要版本；实际基础镜像及依赖以构建记录为准。它是已通过本机基础验证的兼容性重建，目标 H100 仍需实测。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[PyTorch 2.10 官方 cu128 安装组合](https://pytorch.org/get-started/previous-versions/#v2100)

FlashAttention 从现有 2.8.3 源码构建 sm80／sm90，Wall-X 构建 sm86／sm90；源码构建使用单任务以控制内存。实查原环境 FlashAttention 仅含 sm80、Wall-X 仅含 sm86，均没有 PTX；不能直接用于 H100，详见 `logs/native-architecture-audit-20261028.json`。原 `.so`、cu130 wheels、模型、数据、缓存凭据及账号说明不进入 Docker 构建上下文。D0 基础镜像只覆盖 WALL 组件与数据验证，不包含 RLinf 训练覆盖层。项目的 RLinf 适配代码与离线依赖已随训练候选交付；该覆盖层在 H100 的安装、完整模型训练仍待验收，不能从基础包的组件结果推断通过。完整分工和代码基准见 [人类审核版](WALL_RLT_Human_Review.md) 与 [实施任务书](WALL_RLT_Agent_Implementation_Plan.md)。

最终 WALL-RLT 交付仍需：已确认的 Ubuntu／RTX 5070 Ti 先完成环境与官方 WALL 推理验收；H100 完成训练并导出目标 Stage 1 权重后，在该 Ubuntu 上完成真实观测端到端推理与重启验证；随后冻结通过验证的代码版本、依赖与构建输入、权重／processor 摘要、配置及验收报告，再封版打包并到 H100 复验。H100 的 `sm_90` 与 5070 Ti 的 `sm_120` 构建输入须分别绑定同一冻结源码，不能直接混用原生二进制。服务器验收记录单独留存；训练、在线 RL 和机器人闭环分别验收，不新增数据采集任务。

## 1. 本机安装 Docker 并验证 GPU 透传

本机安装与 GPU 透传已完成，以下命令保留用于新构建机复现。安装宿主机软件需要 sudo；密码只在本机终端输入。脚本使用官方 apt 源，安装 Docker Engine 和 NVIDIA Container Toolkit、配置 NVIDIA runtime 并重启 Docker；遇到冲突容器包会停止，不自动卸载。它不安装显卡驱动，也不修改原项目环境。

```bash
cd /home/pangk/workspaces/WALL-OSS-CRP
sudo bash scripts/install_docker_host.sh
```

安装脚本最后执行单卡容器 `nvidia-smi`。这一步只验证 Docker GPU 透传。依据：[Docker Ubuntu 安装](https://docs.docker.com/engine/install/ubuntu/)、[NVIDIA Container Toolkit 安装](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

## 2. 构建候选镜像并在断网容器验证

本次构建与基础验收已完成。以下命令仅用于后续从源码重建，默认创建新的报告目录；三个输入目录参数均为必填。已有交付包可直接按第 4～5 节上传和验证。

```bash
cd /home/pangk/workspaces/WALL-OSS-CRP
sudo env \
  WALL_DATASET='/media/pangk/My Passport/H100_lerobot/Datasets/stack_the_base_llx_04251' \
  WALL_WHEELS='/media/pangk/My Passport/WALL-OSS-CRP-build-cache-20261028/wheelhouse' \
  WALL_PYTHON_BOOTSTRAP='/media/pangk/My Passport/WALL-OSS-CRP-build-cache-20261028/python-bootstrap' \
  WALL_GPU=0 \
  bash scripts/docker_build_validate.sh
```

构建使用本地已恢复的官方 CUDA 基础镜像、`WALL_WHEELS` 指定的 wheel 缓存，以及 `WALL_PYTHON_BOOTSTRAP` 指定的 Python bootstrap **目录**。该目录包含 `python310-bootstrap.tar.gz`、`SHA256SUMS` 和来源／重定位 JSON；不能把参数填为单个 tar 文件。bootstrap 仅来自新建的最小 Conda 基础环境，未打包原项目环境；默认 `conda-pack` 检查、归档路径和重定位检查通过。

构建命令设置 `--pull=false --network none`，所有构建步骤均使用本地输入。Dockerfile 校验 bootstrap 与 wheels 的 SHA256，再通过只读 named context 挂载；Python 安装使用 `--no-index`，源码安装禁用依赖下载与构建隔离，不调用 APT。`docker/runtime-pins.txt` 固定当前环境的 102 个非 CUDA 包；Dockerfile 单独指定 torch cu128 和构建工具，重新安装本地固定源码，避免可编辑安装引用原 `/home/pangk` 路径。基础镜像及实际安装版本保存在镜像构建记录和 inspect 输出；导出只使用本机构建验收通过的 image ID，跨 Docker 版本迁移时按归档内的配置摘要和层摘要核验。

126 个缓存 wheel 包括官方原始 wheel 和从本地已安装包按原始 `RECORD` 校验后重组的 wheel。重组包保留包版本、元数据、平台标签和已验证的 payload，但 wheel 归档 SHA256 是本次重组文件的摘要，不等于上游原始 wheel 摘要。各自的来源与 SHA256 清单保存在缓存及 `logs/cu128-wheel-cache-20261028/`、`logs/runtime-wheel-cache-20261028/`、`logs/cu128-wheel-download-20261028/`；不能把重组包描述为未经改变的官方下载文件。

构建完成后继续以 `--network none` 执行：GPU 组件、双路 processor／归一化、历史数据的两路视频读取和依赖声明检查；默认报告写入 `logs/docker-validation-<UTC时间>/`，也可用 `WALL_REPORT` 指定新目录。本次成功目录为 `logs/docker-validation-delivery-20261029/`。`docker/expected-pip-check.txt` 只记录已知 11 条历史声明问题；新问题会失败，已有问题仍在 JSON 中标 FAIL，不代表完整 LeRobot 或训练依赖已通过。完整 WALL-RLT 推理遵循上述第二阶段流程，不作为当前基础包已通过的项目。

本次已生成 `validated-image-id.txt` 和资产校验清单。运行验证复用已完成构建的精确镜像 ID：`WALL_REUSE_BUILD_REPORT` 与 `WALL_REUSE_IMAGE_ID` 必须成对提供，脚本核对旧 `build.log`、旧／当前镜像身份，在新报告目录重跑组件、数据、依赖和资产哈希，跳过 build 与输入缓存重新导入；watcher 对应参数为 `--reuse-build-report`、`--reuse-image-id`。复用路径检查见 [reuse-built-image-mock-20261029.json](logs/reuse-built-image-mock-20261029.json)。迁移身份工具的历史 Docker 29 归档检查证据仍保留在 `logs/docker29-identity-smoke-20261028.json`、`logs/docker29-identity-loaded-20261028.json`；本次真实完整镜像身份记录已随新包交付，H100 部署记录为 `NOT_RUN`，尚无目标机验收报告。

## 3. 验证通过后导出到 My Passport

本次已导出到 `WALL-OSS-CRP-transfer-20261029`。以下为后续重新导出的命令，必须为 `WALL_BUNDLE` 选择尚不存在的新目录；不要覆盖已 READY 的交付包。

```bash
cd /home/pangk/workspaces/WALL-OSS-CRP
sudo env \
  WALL_REPORT='/home/pangk/workspaces/WALL-OSS-CRP/logs/docker-validation-delivery-20261029' \
  WALL_DATASET='/media/pangk/My Passport/H100_lerobot/Datasets/stack_the_base_llx_04251' \
  WALL_BUNDLE='/media/pangk/My Passport/WALL-OSS-CRP-transfer-新的唯一目录名' \
  bash scripts/docker_export_bundle.sh
```

若已提前生成独立资产包，可向导出命令增加 `WALL_ASSETS`，脚本会比对归档 SHA256 与容器验证时的源文件清单，匹配后复用模型／数据归档。当前独立资产准备目录为 `/media/pangk/My Passport/WALL-OSS-CRP-assets-20261028`；它不含 Docker 镜像，不代表完整交付包。

导出前还会在已验证的精确镜像上，以只读挂载的 `verify_native_architectures.py` 调用容器内 `cuobjdump`，要求 FlashAttention 和 Wall-X 两个实际扩展均含 `sm_90` cubin，记录二进制 SHA256、ELF／PTX 架构及检查结果。本机 GPU 检查通过不能替代这一项。架构检查失败时，脚本在打迁移标签、`docker save` 和创建输出目录之前退出；隔离负测已确认这一行为，记录为 `logs/export-native-gate-20261028.json`。架构覆盖检查本身不运行模型，也不代表 H100 内核实测通过。

本次已生成的输出结构：

```text
WALL-OSS-CRP-transfer-20261029/
├── environment-image.tar.gz   # docker image save 的完整镜像
├── models.tar.gz              # WALL 权重与 Qwen processor
├── smoke-data.tar.gz          # 本次选定的历史小数据集
├── image-id.txt               # 构建机 ID，仅供追溯
├── image-tag.txt              # 导出／加载使用的明确迁移标签
├── image-identity.json        # 配置摘要、有序层摘要及可达的允许 ID
├── SHA256SUMS
├── BUNDLE_READY.json          # 最后发布的完整交付标记，绑定清单摘要与文件集合
├── OFFLINE-DEPLOY.md          # SSH/SCP 上传与 Ubuntu 服务器部署说明，无密码
├── verify_h100_bundle.sh      # 服务器离线组件与数据验证入口
├── verify_native_architectures.py # 检查两项原生扩展包含 sm_90 cubin
├── image_archive_identity.py  # Python 标准库镜像身份校验工具
├── verify_bundle.py           # Python 标准库完整包校验工具
├── model_files.manifest.json
├── sources.lock.json
└── reports/                   # 镜像 ID、组件、数据、依赖及构建记录
```

脚本检查已验证的 image ID 和模型／数据校验和，建立由源 ID 前 12 位确定的迁移标签，再以 `docker image save --platform linux/amd64` 保存标签对应镜像。本机 Docker 29 支持该导出选项；服务器使用普通 `docker load`，兼容旧版 Docker。归档以 `.partial` 写入，确认配置 payload 的 SHA256、Linux/amd64 平台和有序层摘要后重命名，最后检查 gzip 完整性及全包 SHA256。迁移标签、身份 JSON、部署脚本和报告全部纳入 `SHA256SUMS`；不把历史含密码说明文档放入交付包。[Docker 镜像保存说明](https://docs.docker.com/reference/cli/docker/image/save/)

`SHA256SUMS` 先写临时文件，再原子改名；全部文件复核成功后，才原子发布 `BUNDLE_READY.json`。标记采用 `walloss-offline-bundle` schema version 1，完整迁移包的 `package_kind` 为 `transfer`，记录清单 SHA256、文件数、24 项必需文件、归档摘要和镜像身份。清单覆盖交付目录内所有文件，仅排除清单自身与完成标记，避免自引用。构建机导出检查与 Ubuntu 服务器校验器均核对实际文件集合，拒绝空／子集清单、缺失必需文件、重复或逃逸路径、链接、大小写冲突及残留 `.partial`。

导出脚本完成整包复核并发布完成标记后，生成可交付目录。本次 `WALL-OSS-CRP-transfer-20261029` 已完成该流程，清单覆盖 27 个文件；服务器仍须在接收后复验。导出失败时保留失败目录和日志，将 `WALL_BUNDLE` 改为另一个尚不存在的新目录再执行；脚本拒绝覆盖现有目录，不原地续写半成品。完整包正例、空／子集清单与中断拒绝等历史隔离测试记录见 `logs/bundle-completeness-tests-20261029.json`、`logs/bundle-export-integration-20261029.json`；实际完成依据为新包的 `BUNDLE_READY.json` 与对应导出日志。

Docker 29 的 containerd 存储可能以 manifest／index 摘要作为 image ID，旧版 Docker 则使用配置摘要，因此不要求两台机器显示同一个 ID。身份工具只接受归档中所选标签的配置摘要，以及实际可达且指向同一配置的 manifest／index 摘要；同时核对有序 rootfs diff IDs。不会把归档内任意 JSON 的摘要都列为有效镜像 ID。构建机 ID 保留作审计，服务器运行命令使用加载核验后取得的目标机 ID。

My Passport 为 exFAT，能够承载这些大文件；中转时保持归档原样。上述 D0 快照只包含明确的 32 条历史小数据集。后续训练候选另行包含选定的 350 条 `top_error` 轨迹；新推理源码工具包不包含录制数据。

## 4. SSH/SCP 上传压缩后的交付目录

将上述两个目录及旁置说明／清单从 My Passport 拷贝到用户电脑同一目录，保留全部文件和子目录。用户连接专用网线后，在该共同父目录打开提供 ssh／scp 的终端，执行：

```text
ssh aiuser@10.241.90.194 "mkdir -p ~/WALL-OSS-CRP-incoming"
scp -r "WALL-OSS-CRP-transfer-20261029" "WALL-RLT-training-candidate-20261029-v1" "WALL-H100-中转说明-20261029.txt" "WALL-H100-SHA256SUMS-20261029.txt" aiuser@10.241.90.194:~/WALL-OSS-CRP-incoming/
```

上传后，在 Ubuntu 服务器的共同父目录执行 `sha256sum -c WALL-H100-SHA256SUMS-20261029.txt`，再按基础包内 `OFFLINE-DEPLOY.md` 执行完整性校验与环境验证；训练步骤见候选中的实施方案第 9.2 节。中转电脑只承担文件传输。说明、上传与验证日志均存放到两包目录之外。服务器访问网关历史记录为 `10.241.245.6`，上传端 IP 由现场分配，掩码 `/26`；本轮不修改网络配置。

## 5. Ubuntu H100 基础验证

服务器需要已有 Python 3、可工作的 NVIDIA 驱动、Docker Engine 与 GPU 容器运行支持，并允许当前用户运行 Docker。身份工具只使用 Python 标准库。本包不包含宿主机离线安装包；脚本遇到缺失项会明确失败，不更新驱动、重启服务或修改权限。无需在离线服务器构建镜像。

随包提供的 [离线验证脚本](docker/verify_h100_bundle.sh) 先使用 `verify_bundle.py` 核对完成标记、必需文件、完整目录集合及逐文件 SHA256，再检查 gzip、归档身份和宿主机。加载后按迁移标签核对目标机 ID 与有序层摘要，再解压资产并执行断网验收；不完整包会在 Docker 加载与解压前被拒绝。容器首先运行随包只读挂载的架构检查脚本，确认两个扩展含 `sm_90` cubin，随后运行 GPU 组件与历史数据读取检查。选择现场分配的 GPU，以下 `WALL_RUN` 必须是不存在的新目录，父目录需要存在；失败时保留原目录并换新目录重试。

```bash
cd ~/WALL-OSS-CRP-incoming/WALL-OSS-CRP-transfer-20261029
export WALL_RUN="$HOME/WALL-OSS-CRP-container-validation-$(date -u +%Y%m%dT%H%M%SZ)"
# GPU 0 仅为示例，改为服务器实际分配的卡。
export WALL_GPU=0
bash verify_h100_bundle.sh
```

成功时 `$WALL_RUN/reports/components-passed-image-id.txt` 记录架构、组件与样本读取检查通过的服务器镜像 ID；`native-architectures.json` 保留两项扩展的架构及哈希证据，`loaded-image-identity.json` 保留身份验证结果，源 ID 另存为 `source-image-id.txt`。已知依赖声明 FAIL 仍单独保存在 `dependencies.json`。这些 H100 步骤记录为 `NOT_RUN`，尚无目标机验收报告。以下为可单独执行的 H100 完整 WALL 模型合成 smoke 入口，可用于后续目标服务器复验；它既不代表已执行，也不能替代另一台 Ubuntu 的端到端 WALL-RLT 推理：

```bash
WALL_IMAGE_ID="$(cat "$WALL_RUN/reports/components-passed-image-id.txt")"
docker run --rm --pull=never --network none --gpus "device=$WALL_GPU" \
  --user "$(id -u):$(id -g)" --workdir /opt/walloss \
  -v "$WALL_RUN/models:/opt/walloss/models:ro" \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_IMAGE_ID" python docker/verify_full_model.py \
  --output /reports/full-model-verification.json
```

包装脚本使用合成观测，检查动作形状 `32×26` 和有限值，保存加载／推理时延及峰值显存。它仍需 H100 实测；不能把组件 PASS 或脚本存在当成完整模型验收。将输出文件名改为 `full-model-restart-verification.json` 后重跑，可验证新容器离线重启并保留两份报告。服务往返、训练 batch、RLinf／RLT、AR5 示范和机器人闭环保持单独验收。交付包内的 [部署说明源文件](docker/OFFLINE-DEPLOY.md) 包含相同步骤。

---

# 本地独立环境历史说明（2026-10-26）

本目录为 WALL-OSS-0.5 的独立准备目录；本轮范围为克隆、安装与模型下载。完整动作推理和 WebSocket 验收尚未执行。

- 环境：`/home/pangk/workspaces/WALL-OSS-CRP/env`，Python 3.10。
- 主机原始记录：`environment.host.md`。
- 使用官方 conda-forge URL 创建 prefix 环境，没有修改全局 Conda 配置。
- 使用系统现有 `/usr/local/cuda-13.0` 构建扩展，没有安装或修改系统驱动、系统 CUDA。
- PyTorch 目标为 2.10.0+cu130，torchvision 0.25.0+cu130，torchaudio 2.10.0+cu130。其余依赖遵循固定 Wall-X 的 requirements.txt。
- LeRobot v0.4.4 来自固定 Wall-X 部署脚本，使用 `--no-deps -e` 安装。
- FlashAttention 2.8.3 对 torch2.10 没有官方预编译 wheel，需源码构建。该版本枚举 80/90/100/120，不接受 86，因此使用兼容 RTX 3060 Ti 的 sm80 内核；Wall-X 扩展使用 sm86。实际 GPU 内核检查见 `setup-verification.json`。
- 构建使用 MAX_JOBS=1、NVCC_THREADS=1。三任务构建曾触发系统 OOM，编译进程退出 137；原始日志保留在 logs/flash-attn-build-parallel3.log。继续使用单任务完整构建，不删减前向或反向内核。
- DMuon 的可选训练 fast-clip 扩展通过其公开选项 DMUON_BUILD_FAST_CLIP=0 禁用；本轮不验证训练。
- 现有 LeRobot、ACT、π0.5 等环境未被用于安装，也未修改其依赖。
- 系统 shell 原先通过 PYTHONPATH 注入 ROS Humble 的两个 Python 路径；本项目激活脚本清除 PYTHONPATH/PYTHONHOME 并设置 PYTHONNOUSERSITE=1，避免读取这些外部包。没有改动 ROS 配置。

## 使用

```bash
source /home/pangk/workspaces/WALL-OSS-CRP/scripts/activate.sh
python --version
```

## 复现安装

```bash
conda create -y -p /home/pangk/workspaces/WALL-OSS-CRP/env --override-channels -c https://conda.anaconda.org/conda-forge python=3.10 pip
bash /home/pangk/workspaces/WALL-OSS-CRP/scripts/install_runtime.sh
```

源码版本和路径见 `sources.lock.json`；实际安装后的精确 Python 包版本见 `environment.freeze.txt`，Conda 包来源见 `environment.conda-explicit.txt`。`configs/requirements-bootstrap.txt` 仅为下载期间先安装独立依赖的分步清单，最终以官方 requirements 和实际 freeze 为准。

## 安装检查

```bash
source /home/pangk/workspaces/WALL-OSS-CRP/scripts/activate.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /home/pangk/workspaces/WALL-OSS-CRP/scripts/verify_setup.py
python -m pip check
```

`pip check` 的原始结果会保存在 `logs/pip-check.log`。Wall-X 官方要求 LeRobot 使用 --no-deps，且 LeRobot v0.4.4 的 Hub 约束与 Transformers 5.2.0 不同；不能将该声明冲突写成全量依赖一致。实际导入验证与完整模型推理验收分别记录。

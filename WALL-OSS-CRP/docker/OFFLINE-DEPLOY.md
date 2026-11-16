# WALL 基础环境离线部署（D0／H100 范围）

> 当前状态入口：[PROJECT_STATUS.md](../PROJECT_STATUS.md)。D0 基础包、训练候选包和 `WALL-RLT-inference5070-20261029-v1` 推理源码工具包均已在 My Passport 完成打包与校验；新增工具包为 8 个文件、10,979,923 字节。本轮 377 项 CPU 测试 PASS；目标镜像构建、5070 Ti GPU／完整推理记录为 `NOT_RUN`，尚无目标机验收报告，也尚未记录用户拷贝／上传完成回执。详见 [交付记录](../metadata/rlt-inference-deployment-20261029.json) 和 [CPU 报告](../logs/rlt-inference-20261029/cpu-check/report.json)。这是仓库中的现行说明；已交付包内的文档快照保持原样。

**当前优先部署 RTX 5070 Ti 推理主机，按 [5070 Ti 本地离线部署说明](../WALL_RLT_Inference_Deployment.md) 执行，不运行本文要求 `sm_90` 的 H100 脚本。** 目标为 Ubuntu `crp-5070ti-01`、16303 MiB 显存、驱动 595.91.07，工作目录 `/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt`。`nvidia-smi` CUDA 13.2 不是 toolkit；使用 D0 派生的 CUDA 12.8／Torch 2.10.0+cu128 环境，重编 `sm_120` 扩展。官方 WALL 权重已有，训练后的 WALL＋RLT／RL Actor 权重缺失，最终封版尚未完成。

本文保留 D0 与 H100 的后续使用步骤。H100 由用户确认共 8 卡，承担训练和后续复验；双卡 FSDP 候选仍待实测。中转链路为 My Passport → 用户电脑 → 用户连接专用网线 → H100；本机不连接服务器。Windows 只负责文件中转，不做校验、解包或部署。训练候选冻结时的 218 项 CPU 测试属于旧包历史证据，不能改记成本轮 377 项。数据已经采集完成，不安排新采集。

本包包括环境镜像、WALL 权重与 Qwen processor、历史验证小数据集、构建验收报告和部署脚本。数据集为 32 条／15376 帧的历史 smoke 数据，不是 AR5 示范。本基础包的镜像验收范围为 WALL 组件与数据读取；不能据此认定 RLinf／RLT 集成、训练或机器人闭环已通过。

构建机导出前会检查 FlashAttention 与 Wall-X 两个实际原生扩展均包含 H100 所需的 `sm_90` cubin，并记录二进制 SHA256 与架构列表。检查失败会在创建交付目录、打标签或保存镜像前停止。随包提供 `verify_native_architectures.py`，服务器也会复查；该检查不等同于在 H100 运行内核或完整模型。

完整包必须同时具备 `SHA256SUMS` 和最后发布的 `BUNDLE_READY.json`，并通过整包校验。清单先写 `.partial` 再原子改名；所有文件复核成功后才原子发布完成标记。标记记录 schema version 1、`package_kind: transfer`、清单 SHA256／文件数、24 项必需文件、归档摘要和镜像身份。清单覆盖目录中除其自身与完成标记之外的全部文件。构建机导出检查与 Ubuntu 服务器校验器均核对实际文件集合，拒绝空／子集清单、缺失文件、重复／逃逸路径、链接、大小写冲突及残留 `.partial`。服务器部署脚本会自动检查完成标记和整个目录的完整性。

D0 交付包是否就绪，以目录内的 `BUNDLE_READY.json` 和 `verify_bundle.py` 整包校验结果为准；目标服务器是否验收通过，以对应 `$WALL_RUN/reports/` 结果为准。项目现行进展以 [PROJECT_STATUS.md](../PROJECT_STATUS.md) 及其引用证据为准；下文命令不表示服务器已经执行。

如果构建机导出中断，保留失败目录与日志，使用尚不存在的新 `WALL_BUNDLE` 目录重新导出；脚本拒绝覆盖现有目录，不原地续写半成品。下面的示例目录须改为实际成功输出目录。

## 1. 后续由用户中转至 H100

导出完成后，镜像、模型和数据均为 `.tar.gz` 压缩归档。保持目录结构与归档原样，连同脚本、清单和完成标记一起上传；无需解包或再次压缩。以下目录名替换为实际导出目录。

在用户电脑连接专用网线、并已将完整目录从 My Passport 拷到共同父目录后，可用提供 ssh／scp 的终端执行：

```bash
ssh aiuser@10.241.90.194 "mkdir -p ~/WALL-OSS-CRP-incoming"
scp -r "WALL-OSS-CRP-transfer-20261029" aiuser@10.241.90.194:~/WALL-OSS-CRP-incoming/
```

历史构建机直传命令保留如下，仅用于追溯原方案；当前由用户电脑中转，本机不执行该命令：

```bash
ssh aiuser@10.241.90.194 "mkdir -p ~/WALL-OSS-CRP-incoming"
scp -r "/media/pangk/My Passport/WALL-OSS-CRP-transfer-20261029" aiuser@10.241.90.194:~/WALL-OSS-CRP-incoming/
```

下面保留从 Windows 直接读取移动硬盘的历史传输命令，仅作路径示例；现行流程以上述用户电脑中转为准，不要求 Windows 校验。`E:` 仅为示例盘符：

```bat
ssh aiuser@10.241.90.194 "mkdir -p ~/WALL-OSS-CRP-incoming"
scp -r "E:\WALL-OSS-CRP-transfer-20261029" aiuser@10.241.90.194:~/WALL-OSS-CRP-incoming/
```

上传后，在 Ubuntu 服务器执行下一节的完整性校验与环境验证；Windows 不承担部署或实跑验收。保持整个交付目录不变，上传／验证日志存放到目录之外。网关固定为 `10.241.245.6`，1～5 不可用；上传端 IP 由现场分配，掩码 `/26`。服务器地址为 `10.241.90.194`，账号 `aiuser`；登录时使用已有凭据，本包不包含密码。

## 2. H100 离线基础组件验证

服务器必须已有 Python 3、可工作的 NVIDIA 驱动、Docker Engine 和 GPU 容器运行支持，并允许当前用户运行 Docker。随包身份校验工具只使用 Python 标准库。此包不包含宿主机安装包，也不会更新驱动、重启 Docker 或修改用户权限。历史驱动记录为 570.86.10，镜像使用 CUDA 12.8；以服务器实际检查为准。

部署前须检查磁盘可用空间。本基础镜像记录的 `docker image inspect` 大小为 **28,657,257,229 字节，约 28.7GB**；模型展开后为 **8,353,158,306 字节，约 8.35GB**，历史验证数据另约 0.34GB。还需同时保存上传的压缩包，并为 Docker 内容存储、镜像解包临时文件、容器缓存和报告留出空间。镜像大小不能直接当作首次加载的磁盘峰值。

首次部署建议按 **100GB 可用空间**安排保守预算；这是预留建议，**不是精确最低值**。实际需求受已有共享层、Docker 存储方式和压缩包大小影响。如果上传目录、Docker 数据目录、临时目录和 `WALL_RUN` 分属不同文件系统，应按各自承载内容分别检查，不能只看总剩余空间。Docker 数据目录以 `docker info` 输出为准；若配置了独立内容存储或临时目录，也应检查其所在文件系统。

登录服务器并进入上传的完整目录后，可先执行以下只读检查。`df` 的 `Avail` 是可用空间；若 `WALL_RUN` 不在主目录下，将示例中的 `$HOME` 换成其实际父目录。后续版本应使用对应镜像和归档的实际尺寸重新估算。

```bash
WALL_DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
printf 'Docker data directory: %s\n' "$WALL_DOCKER_ROOT"
df -h "$WALL_DOCKER_ROOT" . "$HOME" /tmp
du -sh .
```

完成空间检查后，选择已分配的 GPU；示例 GPU 0 需按现场资源调整。`WALL_RUN` 必须是尚不存在的新目录，父目录需要存在；失败后保留报告，重新执行时换新目录。

```bash
cd ~/WALL-OSS-CRP-incoming/WALL-OSS-CRP-transfer-20261029
export WALL_RUN="$HOME/WALL-OSS-CRP-container-validation-$(date -u +%Y%m%dT%H%M%SZ)"
export WALL_GPU=0
bash verify_h100_bundle.sh
```

脚本先用 `verify_bundle.py` 检查完成标记、必需文件、实际目录的完整文件集合和逐文件 SHA256，再检查 gzip、核对镜像归档身份、检查宿主机并加载镜像；不完整包会在加载与解压前被拒绝。脚本按 `image-tag.txt` 找到服务器上的镜像，只有归档中该标签对应的 Linux/amd64 配置摘要、可达 manifest／index 摘要以及有序层摘要全部符合要求，才解压到新目录并以只读模型／数据挂载执行 `--network none` 验证。容器先运行只读挂载的架构检查脚本，要求两个扩展均含 `sm_90` cubin，再进行 GPU 组件与视频样本检查。

Docker 29 的 containerd 存储和旧版 Docker 可能给同一个镜像显示不同 ID。`image-id.txt` 保留构建机 ID，仅供追溯；`image-identity.json` 记录实际配置摘要与允许的归档 ID。服务器使用核验后得到的本机 ID 运行，报告位于 `$WALL_RUN/reports/`。`loaded-image-identity.json` 记录身份核验结果，`native-architectures.json` 记录扩展架构与哈希；`components-passed-image-id.txt` 仅在架构、组件与样本读取检查通过后写入服务器 ID。

构建机的归档格式测试不能替代目标 H100 的部署验证；服务器部署结论以实际生成、且绑定本包镜像身份的验收报告为准。

`dependencies.json` 会保留已知的依赖声明 FAIL：10 项 LeRobot 缺失声明依赖及一项 Hub 版本约束冲突。只有已知问题被允许记录后继续；新增问题使验证失败。组件通过不能证明完整 LeRobot、训练或完整模型推理可用。

## 3. 后续完整推理与最终封版

基础包已完成其限定范围的交付准备，完成标记不等于完整 WALL-RLT 验收。当前先在已确认的 RTX 5070 Ti 上构建环境并检查官方 WALL；H100 训练并导出目标 Stage 1 权重后，才可继续真实 AR5 观测的 WALL 参考动作＋RLT 特征服务验证。目标 Ubuntu 端到端推理与重启 PASS 后，再冻结代码、依赖与构建输入、权重／processor 摘要、配置及报告，完成最终封版，并到 H100 复验。两类 GPU 原生扩展分别要求 `sm_120`／`sm_90`，验收记录须绑定各自镜像及同一冻结源码，不能将另一台机器通过当成服务器通过。

镜像内 `docker/verify_full_model.py` 是可单独执行的完整 WALL 模型合成 smoke 入口，可用于后续 H100 复验；它不是完整 WALL-RLT 端到端验证。以下命令在上一节成功后使用同一 `WALL_RUN` 和分配的 `WALL_GPU`，执行结果以输出报告为准。

```bash
WALL_IMAGE_ID="$(cat "$WALL_RUN/reports/components-passed-image-id.txt")"
docker run --rm --pull=never --network none --gpus "device=$WALL_GPU" \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  --workdir /opt/walloss \
  -v "$WALL_RUN/models:/opt/walloss/models:ro" \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_IMAGE_ID" python docker/verify_full_model.py \
  --output /reports/full-model-verification.json
```

包装脚本默认使用 `/opt/walloss/models/wall-oss-0.5`、`x2_normal` 与 seed 0，检查合成观测下动作形状为 `32×26` 且数值有限，记录模型加载／推理时延和峰值显存。将输出文件名改为 `full-model-restart-verification.json` 再运行同一命令，可用新容器复核离线重启并保留两份证据。该入口须在目标 H100 实测并保留报告，不能把脚本存在当成验收完成。

完整模型 smoke 只验证合成观测上的单次推理。服务往返、WALL 训练 batch、RLinf／RLT 适配、AR5 数据与机器人闭环各自单独验收；上述脚本不会启动机器人或训练任务。

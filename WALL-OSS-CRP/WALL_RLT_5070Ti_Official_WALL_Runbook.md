# RTX 5070 Ti Ubuntu 环境构建与官方 WALL 推理操作指南

更新：2026-10-31。执行主机：`crp-5070ti-01`；工作目录：`/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt`。

**本阶段已完成：5070 Ti 镜像构建、GPU 小算子及官方 WALL 合成观测完整推理通过，用户于 2026-10-31 确认已在目标机复验。** 下一步为 H100 Stage 1 训练及训练后 WALL＋Encoder 推理。本文保留最初的部署命令供参考，不要求为继续训练而重复本阶段。

现场运行补充了 `USER`、可写 `HOME` 和缓存路径，并使用核对实际绑权与重复归一化张量的受限键兼容检查器，替代第 7–8 节原先“任何缺失／额外键均失败”的检查。两轮输出均为 `32×26`，峰值 allocated/reserved 约 7.94/8.07 GiB。结果见 [现场验证记录](metadata/rlt-5070ti-validation-20261031.json)。

本次官方 WALL 基础验证使用随包官方权重，未启动 RLT 服务、Stage 2 训练或机械臂执行，也未使用 `top_error` 或新增采集。接下来的 Stage 1 使用已有目标数据训练 WALL＋RLT。整体状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)；后续权重导出与服务流程见 [完整部署说明](WALL_RLT_Inference_Deployment.md)。

## 执行思路与完成标准

D0 基础镜像已有 Python、PyTorch、CUDA 编译器及模型运行依赖，但 FlashAttention 与 WALL 的旧 CUDA 扩展没有覆盖 5070 Ti 架构。我们保留 D0，用源码工具包构建新镜像，重编这两个扩展。

`sm_120` 是 5070 Ti 对应的 GPU 编译目标，其 compute capability 为 12.0，见 [NVIDIA 架构表](https://developer.nvidia.com/cuda/gpus)。容器继续使用 **Python 3.10、Torch 2.10.0＋cu128、CUDA 12.8、FlashAttention 2.8.3**。宿主沿用现有驱动；`nvidia-smi` 显示 CUDA 13.2 不能当作容器 toolkit 版本。宿主无需另装 CUDA toolkit 或 PyTorch。

| 顺序 | 执行内容 | 通过后说明什么 |
|---|---|---|
| 1–3 | 检查主机，拷贝并核验两份包 | Docker 可访问，输入文件完整 |
| 4 | 加载 D0，在容器运行 `nvidia-smi` | Docker 可以访问目标 GPU |
| 5 | 构建 `sm_120` 镜像 | 扩展编译成功，静态架构检查通过 |
| 6 | 执行 BF16 和 FlashAttention 小算子 | 基础 GPU 计算实际运行通过 |
| 7–8 | 加载官方 WALL，两次分别启动新容器推理 | 完整权重可加载，合成观测下输出有效动作 |
| 9 | 保存并回传报告 | 有证据记录本次部署结果 |

**每一步成功后再进入下一步；报错就在该步停止，保留日志。** 官方模型输出应为 `32×26`，即 32 步、每步 26 个原始动作槽位。通过后可记为“5070 Ti 官方 WALL 合成观测推理通过”；AR5 joint8 动作、RLT 特征和抓取成功率另行验收。

## 1 准备工作目录并检查主机

以下命令在 **5070 Ti 主机的同一个 Bash 终端**逐段执行。宿主的 `(base)` 提示符可以保留，模型在容器内运行。`set -euo pipefail` 让失败停止后续执行，也能发现被 `tee` 保存日志的命令失败；错误可能导致终端退出，恢复方法见文末。

插入 My Passport，查看真实挂载点：

```bash
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS
```

按输出调整 `WALL_MEDIA`。路径中的空格必须保留引号。

```bash
set -euo pipefail
export WALL_ROOT='/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt'
export WALL_MEDIA='/media/crp-5070ti-01/My Passport'
test -d "$WALL_MEDIA/WALL-OSS-CRP-transfer-20261029"
test -d "$WALL_MEDIA/WALL-RLT-inference5070-20261029-v1"
mkdir -p "$WALL_ROOT/incoming" "$WALL_ROOT/runs"

WALL_RUN="$(mktemp -d "$WALL_ROOT/runs/official-wall5070-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
export WALL_RUN
export WALL_BASE_BUNDLE="$WALL_ROOT/incoming/WALL-OSS-CRP-transfer-20261029"
export WALL_TOOL_BUNDLE="$WALL_ROOT/incoming/WALL-RLT-inference5070-20261029-v1"
mkdir "$WALL_RUN/reports"
declare -p WALL_ROOT WALL_MEDIA WALL_RUN WALL_BASE_BUNDLE WALL_TOOL_BUNDLE > "$WALL_RUN/session.env"
printf '本轮工作目录：%s\n' "$WALL_RUN"

cat /etc/os-release | tee "$WALL_RUN/reports/host-os.txt"
uname -m | tee "$WALL_RUN/reports/host-architecture.txt"
test "$(uname -m)" = x86_64
python3 -c 'import sys; print(sys.version); assert sys.version_info >= (3, 9), "宿主 Python 需至少 3.9"' \
  | tee "$WALL_RUN/reports/host-python.txt"
nvidia-smi 2>&1 | tee "$WALL_RUN/reports/host-nvidia-smi.txt"
docker version 2>&1 | tee "$WALL_RUN/reports/host-docker-version.txt"
docker context show | tee "$WALL_RUN/reports/host-docker-context.txt"
docker info --format '{{json .Runtimes}}' | tee "$WALL_RUN/reports/host-docker-runtimes.json"
WALL_DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
df -h "$WALL_ROOT" /tmp | tee "$WALL_RUN/reports/workspace-space.txt"
sudo df -h "$WALL_DOCKER_ROOT" | tee "$WALL_RUN/reports/docker-space.txt"
free -h | tee "$WALL_RUN/reports/host-memory.txt"
```

应看到 RTX 5070 Ti、约 16GB 显存，以及 Docker 的 Client 和 Server 信息。Docker 找不到命令、权限不足或无法连接 daemon 时，先处理附录 A，再恢复本轮会话。

首次部署按 **至少 100GB 可用磁盘空间**安排预算：两份传输包约 16.80GB、D0 镜像记录大小约 28.7GB、展开模型约 8.35GB，另有派生镜像和编译临时文件。100GB 是预留预算，实际峰值尚未测量。工作目录、Docker 存储和 `/tmp` 若位于不同分区，须分别检查。编译会使用宿主 RAM，后续固定单任务编译，不提高并行数。

## 2 拷贝两份包到本机磁盘

```bash
for WALL_PACKAGE in \
  WALL-OSS-CRP-transfer-20261029 \
  WALL-RLT-inference5070-20261029-v1
do
  if [[ -e "$WALL_ROOT/incoming/$WALL_PACKAGE" ]]; then
    printf '已有目录，交由下一步完整性核验：%s\n' "$WALL_ROOT/incoming/$WALL_PACKAGE"
  else
    cp -a -- "$WALL_MEDIA/$WALL_PACKAGE" "$WALL_ROOT/incoming/"
  fi
done
```

已有副本直接核验，不合并覆盖。如果拷贝中断，保留失败目录，改名移开后再完整拷贝。**不要在包目录添加日志、说明或缓存**，校验会检查文件集合。所有新文件写入 `WALL_RUN`。本次使用 `20261029` 包，旧 `20261028` 目录不用于部署。

## 3 校验文件与镜像归档

```bash
python3 "$WALL_BASE_BUNDLE/verify_bundle.py" verify --bundle "$WALL_BASE_BUNDLE" \
  2>&1 | tee "$WALL_RUN/reports/d0-bundle-verification.log"

python3 "$WALL_TOOL_BUNDLE/package_wall_rlt_inference.py" verify --bundle "$WALL_TOOL_BUNDLE" \
  2>&1 | tee "$WALL_RUN/reports/inference-source-bundle-verification.log"

python3 "$WALL_BASE_BUNDLE/image_archive_identity.py" verify \
  --archive "$WALL_BASE_BUNDLE/environment-image.tar.gz" \
  --identity "$WALL_BASE_BUNDLE/image-identity.json" \
  2>&1 | tee "$WALL_RUN/reports/d0-archive-identity.log"
```

预期结果：

- D0 输出 `Complete transfer bundle verified: 27 files`。27 指清单成员，加清单和完成标记共 29 个文件。
- 源码包输出 `status: INTEGRITY_PASS`、`source_members: 6888`。`gpu_validated: false` 是交付时的正常状态。
- 镜像归档输出 `Archive image identity matches`。

校验会读取大文件，可能持续一段时间。任何校验失败都先解决包完整性问题，不继续加载镜像。

## 4 加载 D0 并验证 GPU 容器访问

```bash
docker load --input "$WALL_BASE_BUNDLE/environment-image.tar.gz" \
  2>&1 | tee "$WALL_RUN/reports/docker-load.log"

WALL_BASE_TAG="$(cat "$WALL_BASE_BUNDLE/image-tag.txt")"
WALL_BASE_IMAGE_ID="$(docker image inspect "$WALL_BASE_TAG" \
  | python3 "$WALL_BASE_BUNDLE/image_archive_identity.py" check-loaded \
      --identity "$WALL_BASE_BUNDLE/image-identity.json" \
      --report "$WALL_RUN/reports/loaded-base-image-identity.json")"
export WALL_BASE_IMAGE_ID
printf '%s\n' "$WALL_BASE_IMAGE_ID" > "$WALL_RUN/reports/base-image-id.txt"
declare -p WALL_BASE_IMAGE_ID >> "$WALL_RUN/session.env"

docker run --rm --pull=never --network none --gpus 'device=0' \
  "$WALL_BASE_IMAGE_ID" nvidia-smi \
  2>&1 | tee "$WALL_RUN/reports/container-nvidia-smi.txt"
```

容器应显示同一张 5070 Ti。出现 `could not select device driver ... gpu` 或 NVIDIA runtime 错误时，先处理附录 B。本步只查询驱动，不运行 D0 的旧 CUDA 扩展。

镜像 ID 按本机 Docker 的实际结果核对，不手抄另一台主机的 ID。不要运行基础包的 `verify_h100_bundle.sh`，它要求 H100 的 `sm_90`。

容器访问通过后，展开官方模型：

```bash
tar --no-same-owner --no-same-permissions --keep-old-files \
  -xzf "$WALL_BASE_BUNDLE/models.tar.gz" -C "$WALL_RUN"
test -f "$WALL_RUN/models/wall-oss-0.5/model.safetensors"
test -f "$WALL_RUN/models/wall-oss-0.5/config.json"
test -d "$WALL_RUN/models/Qwen2.5-VL-3B-Instruct"
```

归档自身含顶层 `models/`，因此解压到 `WALL_RUN`。已经成功解压过就跳过这段，避免重复覆盖。无需解压 `smoke-data.tar.gz`。

## 5 离线构建 5070 Ti 镜像

```bash
WALL_BUILD_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export WALL_INFERENCE_IMAGE_TAG="wall-rlt-inference:5070ti-$WALL_BUILD_STAMP"
export WALL_BUILD_CONTEXT="$WALL_RUN/build-context-$WALL_BUILD_STAMP"
export WALL_BUILD_LOG="$WALL_RUN/reports/inference-build-$WALL_BUILD_STAMP.log"
declare -p WALL_INFERENCE_IMAGE_TAG WALL_BUILD_CONTEXT WALL_BUILD_LOG >> "$WALL_RUN/session.env"

bash "$WALL_TOOL_BUNDLE/build_inference5070.sh" \
  --base-image "$WALL_BASE_IMAGE_ID" \
  --tag "$WALL_INFERENCE_IMAGE_TAG" \
  --context-dir "$WALL_BUILD_CONTEXT" \
  2>&1 | tee "$WALL_BUILD_LOG"

WALL_INFERENCE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$WALL_INFERENCE_IMAGE_TAG")"
export WALL_INFERENCE_IMAGE_ID
printf '%s\n' "$WALL_INFERENCE_IMAGE_ID" > "$WALL_RUN/reports/inference-image-id.txt"
printf '%s\n' "$WALL_INFERENCE_IMAGE_TAG" > "$WALL_RUN/reports/inference-image-tag.txt"
declare -p WALL_INFERENCE_IMAGE_ID >> "$WALL_RUN/session.env"
docker image inspect "$WALL_INFERENCE_IMAGE_ID" > "$WALL_RUN/reports/inference-image-inspect.json"

docker run --rm --pull=never --network none \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_INFERENCE_IMAGE_ID" \
  sh -c 'cp /opt/wall-rlt/build-record/native-sm120.json /opt/wall-rlt/build-record/environment.freeze.txt /opt/wall-rlt/build-record/source-manifest.json /reports/'
```

脚本会核验 D0 和源码，使用 CUDA 12.8 重编 FlashAttention 与 WALL 扩展，最后检查两者都含 `sm_120`。`MAX_JOBS=1`、`NVCC_THREADS=1` 已设置。构建禁用网络、禁止拉取镜像，pip 不下载依赖。

编译耗时以现场为准；短时间没有新输出不能单独判断失败。可在另一终端用 `free -h`、`df -h` 查看资源，或查看已显示路径对应的构建日志。成功时脚本输出 `Image built: ... Static sm_120 coverage passed`；这还不是 GPU 实跑通过。

**构建失败重试：** 处理错误后，重新执行本节，从生成新 `WALL_BUILD_STAMP` 开始。新标签、新目录和新日志保留旧证据；不要复用已有的 `--context-dir`。无需重新拷贝包、加载 D0 或解压已准备好的模型。

## 6 执行 GPU 小算子检查

```bash
docker run --rm --pull=never --network none --gpus 'device=0' \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  --workdir /opt/wall-rlt \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_INFERENCE_IMAGE_ID" \
  python scripts/check_wall_rlt_inference_host.py \
    --device-index 0 --run-kernels \
    --output /reports/host-kernel-check.json \
  2>&1 | tee "$WALL_RUN/reports/host-kernel-check.log"

python3 - "$WALL_RUN/reports/host-kernel-check.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    report = json.load(f)
assert report['status'] == 'PASS', report.get('failed_checks')
assert report['static_checks_status'] == 'PASS'
assert report['kernels']['status'] == 'PASS'
print('GPU 小算子检查通过，可以进入官方 WALL 完整推理。')
PY
```

检查包括 GPU 型号和 UUID、Torch／nvcc 版本匹配、扩展架构，以及实际执行 BF16 矩阵乘法和 FlashAttention。必须保留 `--run-kernels`；仅静态检查通过不能进入下一步。

整体为 `BLOCKED`，或 `kernels` 为 `FAIL`／`NOT_RUN` 时停止。`wall_native_kernel_executed: false` 符合本检查范围：它没有单独执行 WALL 自定义算子，完整模型路径在下一节运行。

## 7 运行官方 WALL 完整推理并用新容器复跑

先记录 GPU 占用，确认 WALL 来自新镜像安装位置：

```bash
nvidia-smi | tee "$WALL_RUN/reports/before-full-model-nvidia-smi.txt"
docker run --rm --pull=never --network none \
  "$WALL_INFERENCE_IMAGE_ID" \
  python -c 'import wall_x; print(wall_x.__file__)' \
  2>&1 | tee "$WALL_RUN/reports/installed-wall-path.txt"
```

WALL 路径应位于 `/opt/walloss/env/.../site-packages/wall_x/`。GPU 若已被其他计算任务占用，待其结束后再执行；16GB 是否足够以实测为准。

下面的循环运行两次，每次创建并退出独立容器。第一轮失败会停止，不执行第二轮：

```bash
for WALL_CHECK_NAME in base-wall-full-model base-wall-full-model-restart
do
  test ! -e "$WALL_RUN/reports/$WALL_CHECK_NAME.json"
  test ! -e "$WALL_RUN/reports/$WALL_CHECK_NAME.log"
  docker run --rm --pull=never --network none --gpus 'device=0' \
    --user "$(id -u):$(id -g)" \
    -e USER="$(id -un)" -e HOME=/tmp \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
    --workdir /opt/walloss \
    -v "$WALL_RUN/models:/opt/walloss/models:ro" \
    -v "$WALL_RUN/reports:/reports" \
    "$WALL_INFERENCE_IMAGE_ID" \
    python /opt/walloss/docker/verify_full_model.py \
      --checkpoint-path /opt/walloss/models/wall-oss-0.5 \
      --norm-key x2_normal --seed 0 \
      --output "/reports/$WALL_CHECK_NAME.json" \
    2>&1 | tee "$WALL_RUN/reports/$WALL_CHECK_NAME.log"
done
```

固定使用 `/opt/walloss/docker/verify_full_model.py`，它需要 D0 中的 `wall-x/scripts/fake_inference.py` 提供合成输入辅助函数。不要换成 `/opt/wall-rlt/docker/verify_full_model.py`。脚本旧说明中的 `H100-only` 是历史名称；实现只要求 CUDA GPU，不限制 H100 型号。

输入是合成相机图像、状态和指令 `pick up the object`，模型通过官方适配器生成动作。两次通过证明新容器中均可离线运行；检查器未保存完整动作或摘要，**不能据此声称两次动作完全一致**。

## 8 检查报告并记录结果

```bash
python3 - "$WALL_RUN/reports" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ('base-wall-full-model', 'base-wall-full-model-restart'):
    report = json.loads((root / (name + '.json')).read_text())
    assert report['status'] == 'PASS', report.get('error')
    assert report['action_shape'] == [32, 26], report['action_shape']
    loads = report['checkpoint_loads']
    assert loads, '未记录到权重加载'
    assert all(not x['missing_keys'] and not x['unexpected_keys'] for x in loads)
    print(name)
    print('  GPU:', report['gpu'])
    print('  动作形状:', report['action_shape'])
    print('  模型加载秒数:', round(report['model_load_seconds'], 3))
    print('  首次推理秒数:', round(report['first_inference_seconds'], 3))
    print('  峰值 allocated GiB:', round(report['peak_allocated_bytes'] / 2**30, 3))
    print('  峰值 reserved GiB:', round(report['peak_reserved_bytes'] / 2**30, 3))
print('两轮官方 WALL 完整模型检查均通过。')
PY
```

完整模型脚本只有在权重键完全匹配、动作形状正确且数值全部有限时才写 `PASS`。`allocated` 是 PyTorch 分配给张量的显存，`reserved` 是缓存分配器保留的显存；两者都不等于整张卡的全部占用。“首次推理秒数”含首次运行开销，不作为稳定推理或机械臂控制频率。

D0 的 11 项依赖声明问题仍保留。本阶段推理通过不等于 `pip check`、完整 LeRobot 或训练依赖全部通过；不要为消除这些声明临时升级 Torch／Transformers。

## 9 打包需要回传的报告

成功或中途失败，都可在恢复本轮变量后执行本节，回传报告包及停止在哪一步。包中只有日志、环境信息和报告，不含权重或数据集。

```bash
WALL_REPORT_ARCHIVE="$WALL_ROOT/$(basename "$WALL_RUN")-reports-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
test ! -e "$WALL_REPORT_ARCHIVE"
tar -czf "$WALL_REPORT_ARCHIVE" -C "$WALL_RUN" reports session.env
sha256sum "$WALL_REPORT_ARCHIVE"
printf '请回传报告包：%s\n' "$WALL_REPORT_ARCHIVE"
```

关键结果是 `host-kernel-check.json`、两份 `base-wall-full-model*.json`、相应日志、构建日志及 `inference-image-id.txt`。进程若被系统直接终止，可能没有 JSON，仍需保留日志。收到现场报告后再更新项目验收状态。

## 中断恢复与常见故障

新开终端先找本轮目录：

```bash
ls -dt /home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt/runs/official-wall5070-*
```

将以下示例换成刚才打印的实际路径：

```bash
set -euo pipefail
source '/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt/runs/这里替换为本轮目录名/session.env'
printf '恢复本轮目录：%s\n' "$WALL_RUN"
```

从失败处恢复，不重复已成功的拷贝、解压和构建。第 7 节会拒绝覆盖已有报告；只重试某一轮时，将循环名称列表换成一个新名称，如 `base-wall-full-model-retry1`。第 8 节读取两次实际成功的报告名称：保留已成功的那轮，仅将失败那轮的名称换成重试名称，失败报告原样保留。

| 现象 | 处理方法 |
|---|---|
| 找不到移动硬盘 | 第 1 节用 `lsblk` 查看挂载点，修改 `WALL_MEDIA` |
| 校验缺文件／哈希不符／多余文件 | 检查拷贝；移开失败副本再完整拷贝，不编辑清单 |
| Docker 权限／daemon 问题 | 附录 A；先使当前用户的 `docker version` 显示 Server |
| 容器看不到 GPU | 附录 B；先使 D0 容器的 `nvidia-smi` 通过 |
| 编译 `Killed`／退出 137 | 检查 RAM、磁盘及系统日志，保留完整编译日志；137 本身不充分证明 OOM |
| `no kernel image`／缺少 `sm_120` | 核对新镜像 ID 和扩展，不能直接运行 D0 的旧扩展 |
| 小算子 `BLOCKED` | 查看 `failed_checks` 和 `kernels`，解决后再加载完整模型 |
| 完整模型 CUDA OOM | 保留报告、日志和 `nvidia-smi`，确认空闲显存；当前验收记为失败 |
| 权重键不匹配／动作出现 NaN | 保存原始错误并停止，不放宽检查来宣告通过 |

## 附录 A Docker 不可用时

已有可用 Docker 就跳过。以下针对 Ubuntu 系统级 Docker Engine；已有 rootless Docker、Docker Desktop 或其他容器栈时，按对应官方配置处理。

**已安装但服务未启动：**

```bash
sudo systemctl start docker
sudo docker version
```

**sudo 可用而普通用户无权限：** 构建脚本内部也调用 Docker，需要当前用户能够访问。为工作账号启用 Docker 组权限后，注销 Ubuntu 会话并重新登录，再恢复 `session.env`；Docker 组允许管理宿主容器。依据 [Docker Linux 安装后配置](https://docs.docker.com/engine/install/linux-postinstall/)。

```bash
sudo usermod -aG docker "$(id -un)"
```

重新登录后检查：

```bash
docker version
```

**尚未安装 Docker：** 两份包不提供宿主安装包。以下在线命令适用于尚未安装容器栈的 Ubuntu amd64 主机，依据 [Docker 官方 Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/)。已有 `docker.io`、containerd、runc 或其他容器栈时，先检查冲突并沿用可工作的现有环境，不运行这段新安装命令。

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -d -m 0755 /etc/apt/keyrings
sudo curl -fsSL -o /etc/apt/keyrings/docker.asc https://download.docker.com/linux/ubuntu/gpg
sudo chmod a+r /etc/apt/keyrings/docker.asc
WALL_UBUNTU_CODENAME="$(. /etc/os-release && printf '%s' "${UBUNTU_CODENAME:-$VERSION_CODENAME}")"
WALL_APT_ARCH="$(dpkg --print-architecture)"
test "$WALL_APT_ARCH" = amd64
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $WALL_UBUNTU_CODENAME
Components: stable
Architectures: $WALL_APT_ARCH
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl start docker
sudo docker version
```

安装完成后按本附录启用当前用户访问，再配置附录 B。完全离线时，先在联网机器按实际 Ubuntu 版本和 amd64 架构准备宿主安装包及依赖，再拷贝安装；本项目的模型环境仍可离线构建。

## 附录 B 配置 NVIDIA GPU 容器支持

宿主 `nvidia-smi` 应先正常。若已有 `nvidia-ctk`，但 D0 容器 GPU 检查失败，可按 [NVIDIA Container Toolkit 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 配置：

```bash
nvidia-ctk --version
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

该操作修改 Docker runtime 配置并重启 Docker；在这台工作机没有其他需保留运行的容器任务时执行。随后恢复 `session.env`，重跑第 4 节容器 `nvidia-smi`。

若缺少 `nvidia-ctk` 且网络可用，首次添加官方仓库并安装：

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
```

已有该仓库和 keyring 则跳过添加步骤，直接更新软件索引并安装。安装后执行本附录前面的 `nvidia-ctk runtime configure` 和 Docker 重启，再重试容器 GPU 检查。完全离线且缺少组件时，先补齐匹配宿主的安装包及依赖，再配置。NVIDIA Container Toolkit 用于将 GPU 提供给容器，与镜像内编译扩展所用的 CUDA toolkit 是两种组件。


# WALL＋RLT：RTX 5070 Ti 本地离线推理部署

**当前进度（2026-10-31）：5070 Ti 环境构建、GPU 小算子和官方 WALL 合成观测推理已通过，用户已确认现场复验。** 下一步在 H100 进行 Stage 1 训练，随后从本文第 7 节导出 WALL＋Encoder 并验证服务。[官方 WALL 操作指南](WALL_RLT_5070Ti_Official_WALL_Runbook.md) 保留首次部署流程供参考。

日期：2026-10-29。目标主机为 Ubuntu `crp-5070ti-01`，工作目录为 `/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt`。用户提供的 GPU 为 RTX 5070 Ti、显存 16303 MiB，驱动为 595.91.07，`nvidia-smi` 显示 CUDA 13.2。本流程由用户通过 My Passport 拷贝后，在目标主机本地执行，不要求 SSH 或连接 H100。

项目当前状态与各包范围见 [统一状态表](PROJECT_STATUS.md)。三份交付包及 377 项 CPU 测试的历史结果不变；本次 [5070 Ti 现场验证](metadata/rlt-5070ti-validation-20261031.json) 新增官方模型两轮 `32×26` 有限数值输出，峰值 allocated/reserved 约 7.94/8.07 GiB。

真实 AR5 WALL＋RLT 服务需要接下来的 Stage 1 训练产物。第 2–6 节保留初次部署命令；现场已补充 `USER`、可写 `HOME` 和缓存路径，并使用严格核对绑权及归一化张量的键兼容检查器完成官方模型验证。不要求回传原始日志或再复跑该阶段才开展训练。

## 1. 文件与兼容性

官方 WALL 基础推理使用三份已交付候选中的以下两个目录，完整拷贝并保留其文件集合与文件内容：

| 目录 | 用途 | 不包含的内容 |
|---|---|---|
| `WALL-OSS-CRP-transfer-20261029` | 约 16.79 GB 的 D0 基础包：环境镜像、官方 WALL 模型、Qwen processor、32 条历史验证数据及校验文件 | 训练后的 AR5 WALL＋RLT 权重 |
| `WALL-RLT-inference5070-20261029-v1` | 已就绪的源码工具包，10,979,923 字节（约 10.98 MB）：本项目代码、精确 patched WALL、完整 FlashAttention 2.8.3 源码、离线构建与核验入口 | 基础镜像、模型权重、原始录制数据、训练依赖覆盖层 |

后续真实观测回放，还需以下两项：

- 经 Stage 1 训练并使用本项目导出脚本生成的独立推理 artifact。
- 目标机上的原始 `top_error` 数据，或者移动硬盘中 `WALL-RLT-training-candidate-20261029-v1` 的 `target-data.tar`。训练候选包已有 350 条选定轨迹，排除了 `mouse`、`tape`。D0 的 32 条历史验证数据不用于 AR5 回放。

RTX 5070 Ti 的 compute capability 为 12.0，对应本流程要求的 `sm_120` 原生机器码；H100 则为 9.0。这来自 [NVIDIA 官方 GPU 架构表](https://developer.nvidia.com/cuda/gpus)。D0 中 FlashAttention 的机器码覆盖 `sm_80/sm_90`、WALL 覆盖 `sm_86/sm_90`，因此需要在 D0 上派生新镜像，重新编译这两个扩展。

派生镜像继续使用 D0 的 Python 3.10、Torch 2.10.0＋CUDA 12.8、Transformers 5.2.0 和 CUDA 12.8 开发工具链。`nvidia-smi` 的 CUDA 13.2 表示驱动所报告的 CUDA 兼容上限，不代表需要给容器安装 CUDA 13.2。新驱动支持较早 CUDA runtime 的规则见 [NVIDIA CUDA 向后兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html)。本流程不修改宿主驱动。

目标机已配置 Docker 29.1.3 与 NVIDIA runtime，当前用户可以运行 GPU 容器，官方 WALL 已在 16GB 显存下完成本次推理。后续训练权重的 WALL＋Encoder 推理继续使用匹配模型与预处理源码的环境。

## 2. 拷贝到目标工作目录，建立新报告目录

以下命令在目标 Ubuntu 的同一个 Bash 终端执行。`WALL_MEDIA` 按移动硬盘实际挂载点调整。若目录已存在，保留它，核对是否为已拷贝的完整包；不要将再次拷贝变成对旧包的覆盖或合并。重复验证时新建 `WALL_RUN`。

```bash
set -euo pipefail
export WALL_ROOT='/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt'
WALL_MEDIA='/media/crp-5070ti-01/My Passport'
mkdir -p "$WALL_ROOT/incoming" "$WALL_ROOT/runs" "$WALL_ROOT/artifacts"

test ! -e "$WALL_ROOT/incoming/WALL-OSS-CRP-transfer-20261029"
test ! -e "$WALL_ROOT/incoming/WALL-RLT-inference5070-20261029-v1"
cp -a -- "$WALL_MEDIA/WALL-OSS-CRP-transfer-20261029" "$WALL_ROOT/incoming/"
cp -a -- "$WALL_MEDIA/WALL-RLT-inference5070-20261029-v1" "$WALL_ROOT/incoming/"

export WALL_BASE_BUNDLE="$WALL_ROOT/incoming/WALL-OSS-CRP-transfer-20261029"
export WALL_TOOL_BUNDLE="$WALL_ROOT/incoming/WALL-RLT-inference5070-20261029-v1"
export WALL_RUN="$WALL_ROOT/runs/inference5070-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -- "$WALL_RUN"
mkdir -- "$WALL_RUN/reports"
declare -p WALL_ROOT WALL_BASE_BUNDLE WALL_TOOL_BUNDLE WALL_RUN > "$WALL_RUN/session.env"
```

后续新开终端时，先 `source` 本轮实际的 `session.env` 路径。所有日志、解包目录和构建上下文都放在 `WALL_RUN`；不要往两个 `incoming` 包目录增加日志、缓存或说明文件，否则完整文件集合校验会失败。

先检查现有宿主环境与空间：

```bash
nvidia-smi | tee "$WALL_RUN/reports/host-nvidia-smi.txt"
docker version > "$WALL_RUN/reports/host-docker-version.txt"
docker info --format '{{json .Runtimes}}' > "$WALL_RUN/reports/host-docker-runtimes.json"
WALL_DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
df -h "$WALL_ROOT" "$WALL_DOCKER_ROOT" /tmp
free -h
```

D0 镜像记录大小约 28.7 GB，模型展开约 8.35 GB；还有 16.79 GB 压缩包、派生镜像新增层与 CUDA 编译临时文件。首次部署可先按至少 100 GB 可用空间安排预算，但这不是已测得的峰值保证；如果 Docker 存储、`/tmp` 和工作目录位于不同文件系统，应分别核对。CPU 内存也影响编译；本构建固定 `MAX_JOBS=1`、`NVCC_THREADS=1`。

## 3. 验证 D0、加载基础镜像、核对本机身份

这一节只加载基础镜像和解包官方模型。**不要运行 D0 的 `verify_h100_bundle.sh`**：它要求 `sm_90`，属于 H100 的检查流程。

```bash
python3 "$WALL_BASE_BUNDLE/verify_bundle.py" verify --bundle "$WALL_BASE_BUNDLE" \
  | tee "$WALL_RUN/reports/d0-bundle-verification.log"
python3 "$WALL_BASE_BUNDLE/image_archive_identity.py" verify \
  --archive "$WALL_BASE_BUNDLE/environment-image.tar.gz" \
  --identity "$WALL_BASE_BUNDLE/image-identity.json" \
  | tee "$WALL_RUN/reports/d0-archive-identity.log"

docker load --input "$WALL_BASE_BUNDLE/environment-image.tar.gz" \
  2>&1 | tee "$WALL_RUN/reports/docker-load.log"
WALL_BASE_TAG="$(cat "$WALL_BASE_BUNDLE/image-tag.txt")"
WALL_BASE_IMAGE_ID="$(docker image inspect "$WALL_BASE_TAG" \
  | python3 "$WALL_BASE_BUNDLE/image_archive_identity.py" check-loaded \
      --identity "$WALL_BASE_BUNDLE/image-identity.json" \
      --report "$WALL_RUN/reports/loaded-base-image-identity.json")"
export WALL_BASE_IMAGE_ID
printf '%s\n' "$WALL_BASE_IMAGE_ID" > "$WALL_RUN/reports/base-image-id.txt"

tar --no-same-owner --no-same-permissions --keep-old-files \
  -xzf "$WALL_BASE_BUNDLE/models.tar.gz" -C "$WALL_RUN"
test -f "$WALL_RUN/models/wall-oss-0.5/model.safetensors"
test -f "$WALL_RUN/models/wall-oss-0.5/config.json"
test -d "$WALL_RUN/models/Qwen2.5-VL-3B-Instruct"
```

`models.tar.gz` 的顶层已经是 `models/`，所以解包到 `WALL_RUN`，不要再解包进 `WALL_RUN/models`。本流程无需展开 `smoke-data.tar.gz`。

Docker 版本及存储后端不同，镜像本机 ID 可以不同；以上 helper 核验归档允许的 config／manifest ID 与有序文件系统层。不要把构建机的 Docker 29 ID 写成另一台机器必然可用的 ID。D0 的固定 config digest 是 `sha256:2eaee50dfe782ceec5f3c29ade76fdb4b641d92229a18404f1cb5001bba353c3`。

确认 GPU 容器访问。此处仅在 D0 中执行 `nvidia-smi`，不使用其旧架构的 CUDA 扩展：

```bash
docker run --rm --pull=never --network none --gpus 'device=0' \
  "$WALL_BASE_IMAGE_ID" nvidia-smi \
  | tee "$WALL_RUN/reports/container-nvidia-smi.txt"
```

如果 Docker 不可访问或该命令不能显示目标 GPU，先处理目标机现有的 Docker/GPU runtime 配置并保留错误输出；不能据此认定模型或 FlashAttention 失败。

## 4. 验证新工具包，离线构建 sm_120 派生镜像

```bash
python3 "$WALL_TOOL_BUNDLE/package_wall_rlt_inference.py" verify \
  --bundle "$WALL_TOOL_BUNDLE" \
  | tee "$WALL_RUN/reports/inference-source-bundle-verification.log"

export WALL_INFERENCE_IMAGE_TAG="wall-rlt-inference:5070ti-$(date -u +%Y%m%dT%H%M%SZ)"
bash "$WALL_TOOL_BUNDLE/build_inference5070.sh" \
  --base-image "$WALL_BASE_IMAGE_ID" \
  --tag "$WALL_INFERENCE_IMAGE_TAG" \
  --context-dir "$WALL_RUN/build-context" \
  2>&1 | tee "$WALL_RUN/reports/inference-build.log"

WALL_INFERENCE_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$WALL_INFERENCE_IMAGE_TAG")"
export WALL_INFERENCE_IMAGE_ID
printf '%s\n' "$WALL_INFERENCE_IMAGE_ID" > "$WALL_RUN/reports/inference-image-id.txt"
printf '%s\n' "$WALL_INFERENCE_IMAGE_TAG" > "$WALL_RUN/reports/inference-image-tag.txt"
declare -p WALL_BASE_IMAGE_ID WALL_INFERENCE_IMAGE_TAG WALL_INFERENCE_IMAGE_ID >> "$WALL_RUN/session.env"
```

构建入口再次核对基础镜像身份，然后将源码提取到新的 `build-context`。目标标签已存在会被拒绝，旧 D0 不会被覆盖。Docker 构建设置 `--pull=false --network=none`；pip 使用 `--no-index --no-deps --no-build-isolation --force-reinstall`，不会下载 wheel 或额外训练依赖。

FlashAttention 使用完整 2.8.3 源码，包括 CUTLASS；包内没有 `.git`，所以其 `setup.py` 走完整源代码分支，不会执行 `git submodule update`。构建依赖的 setuptools、wheel、ninja、packaging、psutil、Torch 和 CUDA 编译器来自 D0。WALL 源码已校验为固定 commit 加精确补丁，重新编译 `_cuda_ext_bin`。运行时使用新安装的 `wall_x`，项目本身放在 `/opt/wall-rlt`。

镜像构建末尾会实际调用 `verify_native_architectures.py --required-arch sm_120`，检查两个已安装扩展中的 cubin，并检查推理依赖导入与 prefix 补丁接口。**构建阶段不运行 GPU 内核、不加载模型权重**。构建成功只说明离线编译与静态检查成功。

可将静态构建记录另存到报告目录：

```bash
docker run --rm --pull=never --network none \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_INFERENCE_IMAGE_ID" bash -euc '
    cp /opt/wall-rlt/build-record/native-sm120.json /reports/build-native-sm120.json
    cp /opt/wall-rlt/build-record/environment.freeze.txt /reports/inference-environment.freeze.txt
    cp /opt/wall-rlt/build-record/source-manifest.json /reports/inference-source-manifest.json
  '
```

编译可能较久。若出现 `Killed`／退出 137，先保留 `inference-build.log` 和 `build-context.build.log`，检查主机内存，而非增加并行编译数。若失败，重新运行 wrapper 时使用新的上下文目录和标签；Docker 可复用已经成功的构建层。

## 5. 在新镜像内运行目标 GPU 小算子检查

```bash
docker run --rm --pull=never --network none --gpus 'device=0' \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  --workdir /opt/wall-rlt \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_INFERENCE_IMAGE_ID" \
  python scripts/check_wall_rlt_inference_host.py \
    --device-index 0 --run-kernels --output /reports/inference-host-kernels.json \
  2>&1 | tee "$WALL_RUN/reports/inference-host-kernels.log"
```

这里要求 Python／Torch／nvcc 版本匹配、实际选中 RTX 5070 Ti、两个已加载扩展包含 `sm_120`；随后执行小型 BF16 矩阵乘法与 FlashAttention 数值检查。成功报告应包含 `status: PASS`、`kernels.status: PASS`。该脚本不执行完整 WALL 模型，也不专门调用 WALL 自定义 CUDA 内核；报告中的 `wall_native_kernel_executed: false` 是明确范围。

## 6. 先验证官方 WALL 模型完整前向

以下命令使用**新派生镜像**，调用 D0 内已有的完整模型检查入口。必须使用 `/opt/walloss/docker/verify_full_model.py`：它会加载 `/opt/walloss/wall-x/scripts/fake_inference.py` 作为合成输入辅助函数；不要把脚本单独复制到另一个目录运行。

```bash
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
    --output /reports/base-wall-full-model.json \
  2>&1 | tee "$WALL_RUN/reports/base-wall-full-model.log"
```

该旧入口的说明字符串保留了历史的 “H100-only” 字样，但实现检查的是 CUDA 可用性，没有 H100 型号或 `sm_90` 断言。我们先完成上节针对 5070 Ti 的架构与内核检查，再用它做完整模型 smoke。

成功条件包括：checkpoint 没有 missing/unexpected keys、输出形状 **`32 × 26`**、数值有限。报告记录 GPU 型号、首次模型加载／推理时间和峰值显存。这里的 26 维是官方 WALL checkpoint 的原始动作槽位；输入也是合成的官方适配器观测。**这不是 AR5 的 7 个关节＋夹爪动作，不是训练后的 RLT 服务，也不代表机械臂任务成功。** 16 GB 显存是否足够以本机报告为准。

需要验证官方模型离线重启时，以新容器重跑同一命令，分别把输出与日志改为 `base-wall-full-model-restart.json`、`base-wall-full-model-restart.log`，保留两轮证据。

## 7. 训练后才能执行：导出与验证 Stage 1 推理 artifact

**目前没有可用于本节的已训练目标 artifact。** 旧的 `WALL-RLT-training-candidate-20261029-v1` 不包含这次新增的导出入口。H100 训练完成、checkpoint 已保存后，需要由用户把本次完整 `WALL-RLT-inference5070-20261029-v1` 工具包也带到 H100 主机。这里只提供后续本地操作命令，本轮不连接 H100。

在 H100 上，先进入已有的训练 Python 环境，使用工具包的标准库提取命令展开到一个新的源码目录，然后从新目录执行导出。**无需在 H100 构建 5070 Ti 镜像**，也不覆盖旧训练包或已有训练工程：

```bash
WALL_EXPORT_TOOLS='/H100上完整工具包路径/WALL-RLT-inference5070-20261029-v1'
WALL_EXPORT_SOURCE='/H100上尚不存在的新目录/wall-rlt-export-sources'
python3 "$WALL_EXPORT_TOOLS/package_wall_rlt_inference.py" extract \
  --bundle "$WALL_EXPORT_TOOLS" --output "$WALL_EXPORT_SOURCE"
python "$WALL_EXPORT_SOURCE/project/scripts/export_wall_rlt_inference.py" \
  --checkpoint /实际已完成checkpoint路径/actor \
  --output /新的导出路径/wall-rlt-stage1-inference
```

输入目录应包含项目 checkpoint sidecar 和 RLinf 已合并的 `model_state_dict/full_weights.pt`；导出器不会替代 FSDP 合并，不会读取优化器来恢复训练。导出目录包含 `wall.safetensors`、`encoder.safetensors`、`assets/` 与 `manifest.json`。输出的用途是 WALL 参考动作＋RLT 特征；没有 decoder，也没有训练好的 Stage 2 RL Actor。不要将 D0 官方权重或随机初始化的 encoder 作为替代品传入服务。

将完整导出目录人工拷贝到目标主机例如：

```text
/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt/artifacts/wall-rlt-stage1-inference/
```

在目标机核验，尚不加载模型和 GPU：

```bash
export WALL_ARTIFACT="$WALL_ROOT/artifacts/wall-rlt-stage1-inference"
test -f "$WALL_ARTIFACT/manifest.json"
docker run --rm --pull=never --network none \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  --workdir /opt/wall-rlt \
  -v "$WALL_ARTIFACT:/artifact:ro" \
  "$WALL_INFERENCE_IMAGE_ID" \
  python scripts/serve_wall_rlt.py --artifact /artifact --check-only \
  | tee "$WALL_RUN/reports/stage1-artifact-integrity.log"
```

校验成功只表示 manifest、文件与目标数据契约一致；模型训练效果和真实推理仍需后续步骤。

## 8. 为真实观测回放准备已采集数据

只选择下面一种数据来源。不会重新采集，不使用 `mouse`、`tape`。

**方案 A：目标机已有同一批原始 `top_error`。** 设置其实际目录，后续只读挂载：

```bash
export WALL_TARGET_DATA='/目标机上现有top_error的实际绝对路径'
test -d "$WALL_TARGET_DATA"
```

保留原始目录结构，例如 `banana/dataset.json`、`banana/episode_0000.npz`、`banana/episode_0000_cam_front.mp4`、`banana/episode_0000_cam_high.mp4`。契约的固定 episode 索引决定实际读取项；原始目录即使仍有废数据目录，也不会选择它们。

**方案 B：从训练候选包提取已经选好的数据。** 先完整拷贝训练候选目录，再调用随包标准库校验器。此命令只展开 `target-data.tar`，不会安装训练 wheel、创建 RLinf 工作树或运行训练：

```bash
test ! -e "$WALL_ROOT/incoming/WALL-RLT-training-candidate-20261029-v1"
cp -a -- "$WALL_MEDIA/WALL-RLT-training-candidate-20261029-v1" "$WALL_ROOT/incoming/"
WALL_TRAIN_BUNDLE="$WALL_ROOT/incoming/WALL-RLT-training-candidate-20261029-v1"
export WALL_TARGET_DATA="$WALL_RUN/data/top_error"

python3 -B - "$WALL_TRAIN_BUNDLE" "$WALL_TARGET_DATA" <<'PY'
from pathlib import Path
import sys
import tempfile
bundle, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).absolute()
if output.exists() or output.is_symlink():
    raise SystemExit('数据提取目标必须为新目录')
sys.path.insert(0, str(bundle))
from unpack_wall_rlt_training import verify_bundle, extract_checked
record = verify_bundle(bundle)
if not record['included_data']:
    raise SystemExit('这个训练候选包没有录制数据')
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='.target-data-', dir=output.parent) as temporary:
    stage = Path(temporary) / 'top_error'
    stage.mkdir()
    extract_checked(bundle / 'target-data.tar', stage, record['payloads']['target-data.tar'])
    stage.rename(output)
print('目标数据逐文件校验并解包完成：', output)
PY
```

两个方案的后续容器路径统一为 `/target-data`。在报告目录生成一份回放专用契约，只调整存储路径，不改 artifact 中的原件：

```bash
docker run --rm -i --pull=never --network none \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  --workdir /opt/wall-rlt \
  -v "$WALL_ARTIFACT:/artifact:ro" \
  -v "$WALL_TARGET_DATA:/target-data:ro" \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_INFERENCE_IMAGE_ID" python - <<'PY'
from dataclasses import asdict
from pathlib import Path
import json
from wall_rlt.data.contract import DataContract
original = DataContract.load('/artifact/assets/data_contract.json')
value = asdict(original)
value.update(root='/target-data', normalization_path='/artifact/assets/normalization.json',
             manifest_path='/artifact/assets/episodes.json')
destination = Path('/reports/replay-contract.json')
with destination.open('x') as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2)
relocated = DataContract.load(destination)
relocated.require_trainable()
assert relocated.fingerprint() == original.fingerprint()
print('Replay contract unchanged:', relocated.fingerprint())
print('Suggested dev episode:', relocated.splits['dev'][0])
PY
```

当前 `check_wall_rlt_inference.py` 没有 `--data-root` 参数，所以必须用这份路径重定位后的 `--contract`。该命令会验证契约与统计文件，具体录制内容则在回放读取时检查。当前固定划分中首个 dev episode 为 `5`。

## 9. 训练 artifact 的服务、真实观测往返和重启对比

先在当前终端启动服务。它只监听同一台主机的回环地址。为了让另一容器的回放客户端访问相同 `127.0.0.1`，此处采用 Ubuntu 的 host network；没有对外监听、没有发布到其他网卡，模型下载仍被容器离线环境变量禁用。

```bash
export WALL_SERVICE_NAME="wall-rlt-5070ti-$(date -u +%Y%m%dT%H%M%SZ)"
docker run -d --pull=never --network host --gpus 'device=0' \
  --name "$WALL_SERVICE_NAME" \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  --workdir /opt/wall-rlt \
  -v "$WALL_ARTIFACT:/artifact:ro" \
  "$WALL_INFERENCE_IMAGE_ID" \
  python -u scripts/serve_wall_rlt.py \
    --artifact /artifact --host 127.0.0.1 --port 8080 \
    --device cuda:0 --vision-attention flash_attention_2 --inference-steps 10
docker logs -f "$WALL_SERVICE_NAME" || test "$?" -eq 130
```

看到 `LISTENING` 后，用 Ctrl-C 退出日志跟随；容器服务仍运行。若服务因 OOM 或其他异常退出，保留 `docker logs`，不继续把后续连接错误解释为模型通过。`LISTENING` 只表明模型加载和监听完成，尚未证明一次推理成功。

运行真实观测回放。客户端容器不分配 GPU，仅负责读取已有 NPZ／视频与发 HTTP 请求：

```bash
docker run --rm --pull=never --network host \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  --workdir /opt/wall-rlt \
  -v "$WALL_ARTIFACT:/artifact:ro" \
  -v "$WALL_TARGET_DATA:/target-data:ro" \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_INFERENCE_IMAGE_ID" \
  python scripts/check_wall_rlt_inference.py \
    --endpoint http://127.0.0.1:8080 \
    --contract /reports/replay-contract.json \
    --episode-id 5 --frame-index 0 --seed 0 --repeats 2 \
    --timeout 180 --atol 0.0001 \
    --output /reports/stage1-replay-first.json \
  2>&1 | tee "$WALL_RUN/reports/stage1-replay-first.log"
docker logs "$WALL_SERVICE_NAME" > "$WALL_RUN/reports/stage1-service-first.log" 2>&1
```

这会确认服务使用相同目标契约与 Stage 1 checkpoint，读取现有同帧双相机、state 和任务文本，再发送两次相同 seed 的请求。预计动作形状为 `32 × 8`、`z_rl` 维数为 2048，以服务 metadata 与 artifact 配置为准。动作来源明确为 `WALL_reference_prediction`；最后一维是针对二值夹爪目标的连续预测，服务不会替你取阈值或下发机械臂命令。

首次报告 PASS 后，重启当前任务自己的服务容器，形成新进程。等待新一条 `LISTENING` 后再回放：

```bash
WALL_RESTART_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker restart "$WALL_SERVICE_NAME"
docker logs --since "$WALL_RESTART_AT" -f "$WALL_SERVICE_NAME" || test "$?" -eq 130
```

退出日志跟随后运行：

```bash
docker run --rm --pull=never --network host \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e HOME=/tmp \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  --workdir /opt/wall-rlt \
  -v "$WALL_ARTIFACT:/artifact:ro" \
  -v "$WALL_TARGET_DATA:/target-data:ro" \
  -v "$WALL_RUN/reports:/reports" \
  "$WALL_INFERENCE_IMAGE_ID" \
  python scripts/check_wall_rlt_inference.py \
    --endpoint http://127.0.0.1:8080 \
    --contract /reports/replay-contract.json \
    --episode-id 5 --frame-index 0 --seed 0 --repeats 2 \
    --timeout 180 --atol 0.0001 \
    --compare-report /reports/stage1-replay-first.json \
    --output /reports/stage1-replay-restart.json \
  2>&1 | tee "$WALL_RUN/reports/stage1-replay-restart.log"
docker logs "$WALL_SERVICE_NAME" > "$WALL_RUN/reports/stage1-service-after-restart.log" 2>&1
docker stop "$WALL_SERVICE_NAME"
```

第一次与重启报告必须使用相同 artifact、契约、episode、frame、seed、实际观测内容 SHA256（`observation_sha256`）、推理设置（`runtime_settings`）和 `atol`。服务实例 UUID（`server_instance_id`）必须不同，否则脚本会拒绝把同一个服务进程的重复请求记成重启验证。两轮动作／特征的最大绝对差还需不超过明示容差。

报告会保存请求前后健康状态；`metadata_after` 中包含实际推理计数、时延和 GPU `peak_allocated_bytes`／`peak_reserved_bytes`。这些是本次运行测量值，不是由显存容量推算的通过结论。失败时保留报告，不通过放宽阈值覆盖原证据。输出文件已存在时回放脚本会拒绝覆盖；重新测试请换新文件名。

## 10. 如何汇报验收结果

按实际完成的层次提交报告：

| 层次 | 证据 | 可以说明什么 |
|---|---|---|
| 包完整性 | 两个包的验证日志、D0 加载身份 | 拷贝完整，基础镜像身份一致 |
| 离线编译 | `inference-build.log`、`build-native-sm120.json`、新镜像 ID | 目标架构原生扩展编译与静态覆盖通过 |
| GPU 小算子 | `inference-host-kernels.json` | BF16／FlashAttention 在目标 GPU 上执行通过 |
| 官方 WALL 完整前向 | `base-wall-full-model.json` | 原始官方 checkpoint 在合成观测上输出有限的 `32×26` 动作，并记录显存／耗时 |
| 训练后 Stage 1 服务 | artifact ID、`stage1-replay-first.json`、`stage1-replay-restart.json` | 已训练 WALL 参考动作＋RLT 特征完成真实观测往返及固定 seed 重启对比 |

D0 仍有 11 项已记录的依赖声明问题；候选 Dockerfile 不安装训练覆盖层，不能据此宣称这些问题已消除。当前工作的本地 CPU 测试、打包校验和脚本存在，都不能替代上表的目标 GPU 实跑。任何一层 PASS 都不意味着 Stage 2 RL Actor 已训练、AR5 任务成功率已提升或机械臂闭环已验收。

实现说明见 [详细实现审阅文档](WALL_RLT_Implementation_Review.md)；基础包的历史 `OFFLINE-DEPLOY.md` 保留 H100 交付背景。本次 5070 Ti 本地部署以本文命令为准，所有实际生成证据保存在本轮 `WALL_RUN`。本次状态统一只更新工作区文档，已有包内文档和旁置交付文档仍是打包时的快照。

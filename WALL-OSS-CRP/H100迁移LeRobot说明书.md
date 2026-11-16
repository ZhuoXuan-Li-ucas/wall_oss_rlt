# H100 迁移 LeRobot 0.5.2／π₀.₅ 历史说明书

文档性质：历史 LeRobot／π0.5 迁移与验证记录；原正文未统一标注记录日期，保留其原有日期与版本。项目适用范围更新：2026-10-29。**本文不是当前 WALL＋RLT 推理部署入口。** 当前项目状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。

本轮推理验证由用户把 My Passport 带到 Ubuntu RTX 5070 Ti（16303 MiB，驱动 595.91.07）主机，按 [推理部署说明](WALL_RLT_Inference_Deployment.md) 自行离线构建和测试。本项目的导出、服务和回放入口已实现并通过 CPU 检查，源码工具包已就绪；交付记录中 sm_120 派生镜像构建、目标 GPU、完整模型推理与完整训练为 `NOT_RUN`，尚无目标机验收报告；当前交付不含已训练 RLT／Actor 权重。H100 的 8 卡资源来自用户确认，保留训练职责；本轮不连接 H100，也不安排远程推理或新采集。已有目标数据为 `top_error` 的 7 类、350 条 episode，排除 `mouse`、`tape`。

## D0 基础交付记录（2026-10-29，独立于 π0.5 历史验证）

`/media/pangk/My Passport/WALL-OSS-CRP-transfer-20261029` 的 WALL 基础包已完成，状态 `BUNDLE_READY`，包内文件合计 16,787,299,846 字节（约 16.79 GB）。当次断网容器组件、历史数据读取、`sm_90` 原生扩展及整包校验 PASS，证据见 [服务器环境交付记录](logs/server-environment-delivery-20261029.json) 和 [当次检查目录](logs/docker-validation-delivery-20261029/)。该包的 `pip check` 保留 11 项声明例外（10 项缺包、1 项 Hub 版本冲突），不能据此声称完整训练环境通过。

D0 未改写，它为新的 5070 Ti 派生镜像提供基础环境；D0 内的原生扩展不能直接视为 sm_120 可用。`WALL-OSS-CRP-transfer-20261028` 是保留的未完成目录，不可部署。此前“先上传 H100”的安排不是本轮推理任务；未来如需迁移训练环境，使用对应包内的 `OFFLINE-DEPLOY.md`。服务器地址仍为 `10.241.90.194`，网关只用 `10.241.245.6`；Windows 只是可选的文件中转，不承担 Linux 环境验收。

## 历史 π0.5 正文（技术参考，不执行）

下面的路径、镜像、安装、SSH/SCP、冒烟和训练命令属于原 LeRobot／π0.5 迁移方案。其 `5/5` 结果与版本快照不属于 WALL＋RLT，也不代表目前的 5070 Ti 或 H100 已完成本项目验收。保留历史命令用于追溯，不按它们更新当前环境或触发新的上传、训练。

<details>
<summary>展开历史 LeRobot／π0.5 迁移说明与原验证结果</summary>

> **原示例场景**：本机有网打包 → 移动硬盘 → Windows 中转 → 远程 Ubuntu 8×H100（无外网）
> **镜像**：`lerobot:v052-cu128`
> **远程根目录**：`~/H100_lerobot`
> **视频后端**：使用 `--dataset.video_backend=pyav`
> **历史记录中的已验证结果**：远程 pi05 冒烟 `5/5` 通过（Torch `2.11.0+cu128`，LeRobot `0.5.2`）

---

## 目录

- [0. 历史 π0.5 迁移总流程](#historical-pi05-flow)
- [1. 最终目录结构](#1-最终目录结构)
- [2. 本机：创建 conda 环境并安装依赖](#2-本机创建-conda-环境并安装依赖)
- [3. 本机：准备 U 盘上的数据与模型](#3-本机准备-u-盘上的数据与模型冒烟要用)
- [4. 本机：conda 内 pi05 冒烟](#4-本机conda-内-pi05-冒烟)
- [5. 本机：conda-pack + 构建 Docker 镜像](#5-本机conda-pack--构建-docker-镜像)
- [6. 本机：Docker 环境检测 + 冒烟](#6-本机docker-环境检测--docker-pi05-冒烟必做后再-save)
- [7. 传输到远程](#7-传输到远程)
- [8. 远程：加载镜像与 hf_offline](#8-远程加载镜像与-hf_offline)
- [9. 远程：环境检测](#9-远程环境检测)
- [10. 远程：pi05 冒烟](#10-远程pi05-冒烟)
- [11. 历史远程 π0.5 训练模板](#historical-pi05-training)
- [12. 关键注意事项](#12-关键注意事项)
- [13. 历史 π0.5 成功时版本快照](#historical-pi05-versions)

---

<a id="historical-pi05-flow"></a>

## 0. 历史 π0.5 迁移总流程

```text
① 本机 conda 建环境 + 装依赖 + 自检
② 本机 conda pi05 冒烟
③ conda-pack + Docker build
④ 本机 Docker 环境检测 + Docker pi05 冒烟
⑤ docker save 写入 U 盘 Environments
⑥ 组装 H100_lerobot（Datasets / Models / lerobot 源码）
⑦ U 盘 → Windows → 远程 ~/H100_lerobot
⑧ 远程 docker load + hf_offline + 权限
⑨ 远程环境检测 + pi05 冒烟
⑩ 远程正式训练
```

---

## 1. 最终目录结构

移动硬盘与远程保持一致：

```text
H100_lerobot/
├── Environments/
│   ├── lerobot_v052_cu128.tar.gz   # Docker 镜像（必传）
│   └── lerobot052_env.tar.gz       # conda-pack 备份（建议保留）
├── Datasets/
│   └── stack_the_base_llx_04251/   # 冒烟集（~322MB）
├── Models/
│   ├── pi05_base/
│   ├── paligemma-3b-pt-224/
│   ├── bart-large/
│   └── （其它可选权重）
├── lerobot/                        # 源码（排除 outputs 等）
├── hf_offline/                     # 远程生成：离线 HF 缓存
├── smoke_out/                      # 远程冒烟输出（可选）
└── outputs/                        # 正式训练输出
```

---

## 2. 本机：创建 conda 环境并安装依赖

```bash
conda create -n lerobot052_cu128 python=3.12 -y
conda activate lerobot052_cu128
unset PYTHONPATH

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd /path/to/lerobot   # 例：/home/dingyuxuan/lerobot
pip install '.[dataset,pi,training]'   # 不要用 -e
```

若 `packaging` 无法 import（仅有 dist-info）：

```bash
pip uninstall -y packaging
rm -rf "$CONDA_PREFIX/lib/python3.12/site-packages/packaging" \
       "$CONDA_PREFIX/lib/python3.12/site-packages/packaging-"*.dist-info
pip install --force-reinstall --no-cache-dir 'packaging==25.0' 'setuptools==78.1.0'
```

依赖自检：

```bash
unset PYTHONPATH
python <<'PY'
mods = [
    "packaging","torch","transformers","datasets","scipy","accelerate",
    "pandas","pyarrow","av","safetensors","einops","draccus","lerobot",
]
for m in mods:
    __import__(m); print("ok", m)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print("pi05+dataset import ok")
print("torch", __import__("torch").__version__)
print("lerobot", __import__("lerobot").__version__)
PY
```

---

## 3. 本机：准备 U 盘上的数据与模型（冒烟要用）

```bash
ROOT="/media/dingyuxuan/My Passport/H100_lerobot"
mkdir -p "$ROOT"/{Environments,Datasets,Models}

# 冒烟数据集（从 3floor 选小集，勿拷整包 100G）
cp -a "$HOME/.cache/huggingface/lerobot/dataset/3floor/stack_the_base_llx_04251" \
  "$ROOT/Datasets/"

# 预训练权重拷入 Models/：pi05_base、paligemma-3b-pt-224、bart-large
# （来源可以是内网 SMB 等）

# 源码（可选，排除大目录；U 盘若为非 Linux 文件系统用 -L）
rsync -a -L \
  --exclude='outputs/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='build/' \
  --exclude='.venv/' \
  --exclude='license.key' \
  /path/to/lerobot/ "$ROOT/lerobot/"
```

本机离线 HF 缓存（conda / Docker 冒烟共用）：

```bash
ROOT="/media/dingyuxuan/My Passport/H100_lerobot"
HF="$HOME/hf_offline_local"
mkdir -p "$HF/hub/models--google--paligemma-3b-pt-224/snapshots/local" \
         "$HF/hub/models--google--paligemma-3b-pt-224/refs" \
         "$HF/hub/models--facebook--bart-large/snapshots/local" \
         "$HF/hub/models--facebook--bart-large/refs"

cp -a "$ROOT/Models/paligemma-3b-pt-224/." \
  "$HF/hub/models--google--paligemma-3b-pt-224/snapshots/local/"
echo -n local > "$HF/hub/models--google--paligemma-3b-pt-224/refs/main"

cp -a "$ROOT/Models/bart-large/." \
  "$HF/hub/models--facebook--bart-large/snapshots/local/"
echo -n local > "$HF/hub/models--facebook--bart-large/refs/main"
```

---

## 4. 本机：conda 内 pi05 冒烟

```bash
conda activate lerobot052_cu128
unset PYTHONPATH
ROOT="/media/dingyuxuan/My Passport/H100_lerobot"
export HF_HOME="$HOME/hf_offline_local"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

OUT="$HOME/pi05_smoke_conda_$(date +%H%M%S)"
lerobot-train \
  --policy.type=pi05 \
  --policy.pretrained_path="$ROOT/Models/pi05_base" \
  --dataset.repo_id=stack_the_base_llx_04251 \
  --dataset.root="$ROOT/Datasets/stack_the_base_llx_04251" \
  --dataset.video_backend=pyav \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --batch_size=1 --steps=5 --save_freq=5 --eval_freq=100000 --num_workers=2 \
  --output_dir="$OUT" \
  --job_name=pi05_smoke_conda \
  --wandb.enable=false --policy.push_to_hub=false
```

期望：`Training: 100% 5/5`，`End of training`。

---

## 5. 本机：conda-pack + 构建 Docker 镜像

```bash
conda activate lerobot052_cu128
unset PYTHONPATH
USB="/media/dingyuxuan/My Passport/H100_lerobot/Environments"
mkdir -p "$USB" ~/lerobot052_docker

rm -f "$USB/lerobot052_env.tar.gz"
"$HOME/miniforge3/bin/conda-pack" -n lerobot052_cu128 \
  -o "$USB/lerobot052_env.tar.gz" --ignore-missing-files

cd ~/lerobot052_docker
cp -f "$USB/lerobot052_env.tar.gz" .
```

`Dockerfile`：

```dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
COPY lerobot052_env.tar.gz /tmp/env.tar.gz
RUN mkdir -p /opt/conda/envs/lerobot052_cu128 \
 && tar -xzf /tmp/env.tar.gz -C /opt/conda/envs/lerobot052_cu128 \
 && rm /tmp/env.tar.gz \
 && /opt/conda/envs/lerobot052_cu128/bin/python \
      /opt/conda/envs/lerobot052_cu128/bin/conda-unpack
ENV PATH=/opt/conda/envs/lerobot052_cu128/bin:$PATH
WORKDIR /workspace
CMD ["bash"]
```

```bash
docker build --no-cache -t lerobot:v052-cu128 .

# 快速 import 检查
docker run --rm lerobot:v052-cu128 \
  python -c "import packaging,transformers,datasets,lerobot,torch; print(packaging.__version__, transformers.__version__, datasets.__version__, lerobot.__version__, torch.__version__)"
# 期望含：25.0、5.5.x、4.8.x、0.5.2、2.11.0+cu128
```

---

## 6. 本机：Docker 环境检测 + Docker pi05 冒烟（必做后再 save）

```bash
ROOT="/media/dingyuxuan/My Passport/H100_lerobot"
HF="$HOME/hf_offline_local"
IMG=lerobot:v052-cu128

# —— 环境检测 ——
docker run --rm --gpus all \
  -v "$HF:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  "$IMG" \
  python -c "
import torch, lerobot, packaging, transformers, datasets
from transformers import AutoTokenizer
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('lerobot', lerobot.__version__)
print('packaging', packaging.__version__, 'transformers', transformers.__version__, 'datasets', datasets.__version__)
print('cuda_ok', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
tok = AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224', local_files_only=True)
print('tok_ok', len(tok))
"

docker run --rm --gpus all \
  -v "$ROOT/Datasets:/data/Datasets:ro" \
  "$IMG" \
  python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    'stack_the_base_llx_04251',
    root=Path('/data/Datasets/stack_the_base_llx_04251'),
    video_backend='pyav',
)
print('episodes', ds.num_episodes, 'frames', ds.num_frames)
print('sample_ok', list(ds[0].keys())[:8])
"

# —— 容器内 pi05 冒烟 ——
mkdir -p "$ROOT/smoke_out_local"
rm -rf "$ROOT/smoke_out_local/pi05_smoke"

docker run --rm -it --gpus all \
  -v "$ROOT/Datasets:/data/Datasets:ro" \
  -v "$ROOT/Models:/models:ro" \
  -v "$HF:/root/.cache/huggingface" \
  -v "$ROOT/smoke_out_local:/data/outputs" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  "$IMG" \
  lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=/models/pi05_base \
    --dataset.repo_id=stack_the_base_llx_04251 \
    --dataset.root=/data/Datasets/stack_the_base_llx_04251 \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.train_expert_only=true \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --batch_size=1 --steps=5 --save_freq=5 --eval_freq=100000 --num_workers=2 \
    --output_dir=/data/outputs/pi05_smoke \
    --job_name=pi05_smoke_local_docker \
    --wandb.enable=false --policy.push_to_hub=false
```

期望：`tok_ok`、`sample_ok`、`Training: 100% 5/5`、`End of training`。

**通过后再导出镜像：**

```bash
USB="/media/dingyuxuan/My Passport/H100_lerobot/Environments"
rm -f "$USB/lerobot_v052_cu128.tar.gz"
docker save lerobot:v052-cu128 | gzip > "$USB/lerobot_v052_cu128.tar.gz"
ls -lh "$USB"/lerobot*.tar.gz
# 可选：gzip -t "$USB/lerobot_v052_cu128.tar.gz" && echo gzip_ok
```

---

## 7. 传输到远程

1. 确认 U 盘 `H100_lerobot` 含 Environments / Datasets / Models / lerobot
2. 经 Windows 拷到远程：`~/H100_lerobot`
3. **只更新环境时**：覆盖远程 `Environments/` 两个 tar.gz，其它目录可不动

---

## 8. 远程：加载镜像与 hf_offline

前提：Docker、NVIDIA Container Toolkit、驱动可用（成功案例为 Driver **570.86.10** / CUDA **12.8**）。

```bash
ROOT=~/H100_lerobot
sudo chown -R "$USER:$USER" "$ROOT"

docker rmi lerobot:v052-cu128 2>/dev/null || true
gunzip -c "$ROOT/Environments/lerobot_v052_cu128.tar.gz" | docker load
docker images | grep v052

HF="$ROOT/hf_offline"
mkdir -p "$HF/hub/models--google--paligemma-3b-pt-224/snapshots/local" \
         "$HF/hub/models--google--paligemma-3b-pt-224/refs" \
         "$HF/hub/models--facebook--bart-large/snapshots/local" \
         "$HF/hub/models--facebook--bart-large/refs"

cp -a "$ROOT/Models/paligemma-3b-pt-224/." \
  "$HF/hub/models--google--paligemma-3b-pt-224/snapshots/local/"
echo -n local > "$HF/hub/models--google--paligemma-3b-pt-224/refs/main"

cp -a "$ROOT/Models/bart-large/." \
  "$HF/hub/models--facebook--bart-large/snapshots/local/"
echo -n local > "$HF/hub/models--facebook--bart-large/refs/main"
```

---

## 9. 远程：环境检测

```bash
ROOT=~/H100_lerobot
HF="$ROOT/hf_offline"
IMG=lerobot:v052-cu128

nvidia-smi

docker run --rm --gpus all \
  -v "$HF:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  "$IMG" \
  python -c "
import torch, lerobot, packaging, transformers, datasets
from transformers import AutoTokenizer
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('lerobot', lerobot.__version__)
print('packaging', packaging.__version__, 'transformers', transformers.__version__, 'datasets', datasets.__version__)
print('cuda_ok', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
print('names', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
tok = AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224', local_files_only=True)
print('tok_ok', len(tok))
"

docker run --rm --gpus all \
  -v "$ROOT/Datasets:/data/Datasets:ro" \
  "$IMG" \
  python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    'stack_the_base_llx_04251',
    root=Path('/data/Datasets/stack_the_base_llx_04251'),
    video_backend='pyav',
)
print('episodes', ds.num_episodes, 'frames', ds.num_frames)
print('sample_ok', list(ds[0].keys())[:8])
"
```

期望：8×H100、`tok_ok`、数据集可读。

---

## 10. 远程：pi05 冒烟

```bash
ROOT=~/H100_lerobot
HF="$ROOT/hf_offline"
IMG=lerobot:v052-cu128
mkdir -p "$ROOT/smoke_out"
rm -rf "$ROOT/smoke_out/pi05_smoke"

docker run --rm -it --gpus all \
  -v "$ROOT/Datasets:/data/Datasets:ro" \
  -v "$ROOT/Models:/models:ro" \
  -v "$HF:/root/.cache/huggingface" \
  -v "$ROOT/smoke_out:/data/outputs" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  "$IMG" \
  lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=/models/pi05_base \
    --dataset.repo_id=stack_the_base_llx_04251 \
    --dataset.root=/data/Datasets/stack_the_base_llx_04251 \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.train_expert_only=true \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --batch_size=1 --steps=5 --save_freq=5 --eval_freq=100000 --num_workers=2 \
    --output_dir=/data/outputs/pi05_smoke \
    --job_name=pi05_smoke \
    --wandb.enable=false --policy.push_to_hub=false
```

成功：`Training: 100% 5/5`，`End of training`。

---

<a id="historical-pi05-training"></a>

## 11. 历史远程 π0.5 训练模板（不用于本轮 WALL＋RLT）

**默认始终带** `--dataset.video_backend=pyav`。

```bash
ROOT=~/H100_lerobot
HF="$ROOT/hf_offline"
IMG=lerobot:v052-cu128
mkdir -p "$ROOT/outputs"
OUT=pi05_run_$(date +%Y%m%d_%H%M)

docker run --rm -it --gpus all \
  -v "$ROOT/Datasets:/data/Datasets:ro" \
  -v "$ROOT/Models:/models:ro" \
  -v "$HF:/root/.cache/huggingface" \
  -v "$ROOT/outputs:/data/outputs" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  "$IMG" \
  lerobot-train \
    --policy.type=pi05 \
    --policy.pretrained_path=/models/pi05_base \
    --dataset.repo_id=你的数据集名 \
    --dataset.root=/data/Datasets/你的数据集名 \
    --dataset.video_backend=pyav \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.train_expert_only=false \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --batch_size=8 \
    --steps=30000 \
    --save_freq=5000 \
    --eval_freq=100000 \
    --num_workers=8 \
    --log_freq=100 \
    --output_dir=/data/outputs/$OUT \
    --job_name=$OUT \
    --wandb.enable=false \
    --policy.push_to_hub=false
```

| 参数 | 说明 |
|------|------|
| `--dataset.video_backend=pyav` | **默认必加** |
| `--batch_size` | OOM 则降低 |
| `--policy.train_expert_only` | `true` 省显存；`false` 训全模型 |
| 单卡 | `--gpus '"device=0"'` |

检查点：`$ROOT/outputs/$OUT`。

换数据集：放入 `$ROOT/Datasets/<名>/`，改 `repo_id` 与 `root`。

---

## 12. 关键注意事项

1. 安装必须用 `pip install '.[dataset,pi,training]'`，不能只装基础包。
2. `conda-pack` 命令带连字符；目标 tar 已存在先删除。
3. Docker 构建用新 env：`cp -f` + 必要时 `--no-cache`。
4. 远程加载：`gunzip -c xxx.tar.gz | docker load`（不要 `tar xzf`）。
5. **本机 Docker 冒烟通过后再 `docker save`**，减少白跑 U 盘。
6. 远程 `H100_lerobot` 权限不对时先 `chown`。
7. 输出目录已存在会报错：换名或删除。
8. 驱动与 cu128 对齐（成功案例：570.86.10 / CUDA 12.8）。

---

<a id="historical-pi05-versions"></a>

## 13. 历史 π0.5 成功时版本快照

| 项 | 值 |
|----|-----|
| 镜像 | `lerobot:v052-cu128` |
| Python | 3.12 |
| Torch | 2.11.0+cu128 |
| LeRobot | 0.5.2 |
| packaging / transformers / datasets | 25.0 / 5.5.4 / 4.8.5 |
| 冒烟数据 | `stack_the_base_llx_04251`（32 ep / 15376 frames） |
| 底座 | `/models/pi05_base` |
| 视频 | `pyav` |

</details>

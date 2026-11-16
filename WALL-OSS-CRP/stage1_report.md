# WALL-OSS-CRP：环境与模型准备历史报告（2026-10-26）

安装与下载：2026-10-22；本报告最终离线检查：2026-10-26。适用范围标注：2026-10-29。

> 本文是 RTX 3060 Ti 开发机的历史检查记录，下面的 PASS／FAIL、包版本、显存测量和“尚未执行”均以 2026-10-26 为截止点，不代表当前项目总状态。原始日志与测量未改写。当前入口见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，最新部署步骤见 [推理部署说明](WALL_RLT_Inference_Deployment.md)。
>
> 截至 2026-10-29，独立的 WALL＋RLT 导出、localhost 服务和回放检查入口已实现并通过 CPU 检查；离线源码工具包已就绪，等待用户在 Ubuntu RTX 5070 Ti 主机自行构建／验证。交付记录中 5070 Ti 派生镜像构建、目标 GPU 与完整训练为 `NOT_RUN`，尚无目标机验收报告；当前交付不含已训练 RLT 权重。下文旧报告的“未启动服务”等表述不能用来否定这些新增实现。

2026-10-22～26 的工作按用户“首先完成克隆仓库并创建独立环境，下载模型”的范围执行。当时仓库、独立环境和本地模型资源已准备完成，该轮推理组件的离线环境检查通过。完整模型参数加载、动作推理、WebSocket、性能测量和新进程完整模型离线重启尚未执行，不能写“第一阶段通过”。

## 截至 2026-10-26 的已完成结果

| 项目 | 结果 | 证据 |
|---|---|---|
| 主机与资源检查 | PASS | [environment.host.md](environment.host.md)；[checkpoint-summary.json](metadata/checkpoint-summary.json) |
| 官方代码克隆、固定版本 | PASS | [sources.lock.json](sources.lock.json)；[clone.log](logs/clone.log) |
| 独立 Python 3.10 环境与该轮推理依赖安装 | PASS | [environment.freeze.txt](environment.freeze.txt)；[安装日志](logs/runtime-install.log) |
| FlashAttention 完整源码构建 | PASS | [flash-attn-build.log](logs/flash-attn-build.log) |
| Wall-X 自定义 CUDA 扩展构建与导入 | PASS | [wall-x-build.log](logs/wall-x-build.log)；[setup-verification.json](setup-verification.json) |
| 官方权重、配置、归一化与处理器文件校验 | PASS，24 个文件 | [model_files.manifest.json](model_files.manifest.json)；[model-verification.log](logs/model-verification.log) |
| 离线 BF16 运算、FlashAttention、模块导入、处理器与归一化检查 | PASS | [setup-verification.json](setup-verification.json)；[检查日志](logs/setup-verification.log) |
| 所有已安装包的依赖声明一致性 | FAIL | [pip-check.log](logs/pip-check.log)，见下文限制 |

上述 PASS 仅覆盖表中具体检查，不表示模型已在 GPU 上完整加载或输出过动作。

## 主机与隔离方式

- Ubuntu 24.04.4，Intel i7-12700F，约 16GB RAM，无 swap。
- GPU 0：NVIDIA GeForce RTX 3060 Ti，8192MiB 显存；驱动 580.178.04。
- 使用现有 CUDA 13.0 工具链；没有修改系统驱动、系统 CUDA 或其他任务环境。
- 项目：`/home/pangk/workspaces/WALL-OSS-CRP`。
- Conda prefix：`/home/pangk/workspaces/WALL-OSS-CRP/env`，Python 3.10.22。
- 激活脚本清除系统 shell 注入的 ROS Python 路径，并禁用用户 site-packages，避免读取外部环境包。
- 初始可用磁盘约 68GB；最终检查时约 50GB，详见主机原始记录及终端检查。

## 固定版本与本地资源

| 资源 | 固定版本 / revision | 本地路径 |
|---|---|---|
| Wall-X | `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`，包版本 1.1.0 | `wall-x/` |
| LeRobot | v0.4.4，`8fff0fde7c79f23a93d845d1a50e985de01f8b8a` | `lerobot/` |
| DMuon | `f802a282207984da248defa9d5ade0e57dbbfb67`，0.2.0 | `dmuon/` |
| Wall-OSS-0.5 | `b7dc23a82dbf70a65859c9fdd596703234597028` | `models/wall-oss-0.5/` |
| Qwen2.5-VL-3B-Instruct processor | `66285546d2b821cf421d4f5eb2576359d3770cd3` | `models/Qwen2.5-VL-3B-Instruct/` |

核心运行包为 PyTorch 2.10.0+cu130、torchvision 0.25.0+cu130、torchaudio 2.10.0+cu130、Transformers 5.2.0、Accelerate 1.13.0、PEFT 0.18.1、FlashAttention 2.8.3。完整清单及本地 wheel 来源见 [environment.freeze.txt](environment.freeze.txt)、[environment.conda-explicit.txt](environment.conda-explicit.txt) 和 [sources.lock.json](sources.lock.json)。

来源为[官方 Wall-X 仓库](https://github.com/X-Square-Robot/wall-x)、[官方 Wall-OSS-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5) 和 [Qwen processor](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)，实际使用上述固定版本。Qwen 仅下载 processor/tokenizer、配置和说明文件，没有额外下载其 VLM 权重。没有下载完整 LIBERO 数据集。

模型权重文件为 8,331,486,672 字节，SHA256：

```text
37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a
```

LFS 文件与官方 metadata 的 SHA256 对照，普通 Git 文件与官方 Git blob SHA1 对照；另保存全部文件的实际 SHA256。最终校验在离线设置下完成。

## 环境与资源检查的实际内容

- CUDA BF16 矩阵运算通过；FlashAttention BF16 前向结果有限，且与 PyTorch SDPA 对照通过。
- Wall-X 原生扩展、Qwen2.5 模型类、harrix adapter、LeRobot 数据集模块均导入成功。
- 两路合成 RGB 图像均为 448×448；processor 的两行 `image_grid_thw` 均为 `[1,32,32]`，`pixel_values` 形状为 `[2048,1176]`。
- 使用官方 `load_wallx_processors` 离线加载本地 checkpoint processor；flow 所需 `<|propri|>`、`<|action|>` token ID 分别为 151665、151666，加载后 tokenizer 大小为 151667。
- 官方 action/proprio 归一化文件中的 `x2_normal` 均为 26 维，全部数值有限；没有使用伪造归一化统计。官方配置的 flow horizon 为 32。
- safetensors 文件头可读：1061 个张量，4,162,224,832 个元素；其中 4,158,769,152 个为 BF16，3,455,680 个为 F32。这是文件元数据检查，没有将完整预训练参数加载到模型实例。

权重张量存储共 8,331,361,024 字节，约 **7.759GiB**。最终记录的 GPU 可用显存为 **7299MiB，约 7.13GiB**；该数值随桌面进程占用变化。按完整权重驻 GPU 的普通路径，预计在激活、缓存等开销之前就会面临显存不足。此为资源预判，未通过完整模型加载来实测 OOM，也没有以量化或较小目标配置替代验证。

## 安装问题及保留的限制

1. 初始 Conda 镜像返回 403，改用本次命令的官方 conda-forge URL；没有改全局配置，原始日志保留。
2. 下载发生连接提前结束及临时网络故障，使用固定 revision、可续传分段下载，最终完整文件哈希已通过。失败和恢复日志均保留。
3. FlashAttention 没有与本次 torch2.10 匹配的官方 v2.8.3 wheel，故从官方 2.8.3 源码构建。三并发曾触发 OOM、退出 137，见 [原始错误](logs/flash-attn-build-parallel3.log)；最终以 `MAX_JOBS=1`、`NVCC_THREADS=1` 完整构建成功，没有删减反向内核。FlashAttention 使用 sm80 内核，Wall-X 扩展使用 sm86，实际前向内核检查通过。
4. DMuon 可选的训练 fast-clip CUDA 扩展通过公开选项 `DMUON_BUILD_FAST_CLIP=0` 禁用；该扩展不用于本轮 flow 资源与环境检查，训练能力未验证。
5. LeRobot 按固定 Wall-X 官方说明以 `--no-deps` 安装，避免覆盖 Wall-X 的依赖。`pip check` 仍报告 10 个未安装的 LeRobot 声明依赖：cmake、deepdiff、diffusers、gymnasium、imageio、jsonlines、pynput、pyserial、rerun-sdk、torchcodec；同时 LeRobot 要求 Hub `<0.36`，实际 Hub 1.33.0 满足 Wall-X 的 Transformers 5.2.0 路径。该轮推理模块导入检查通过，但不能认定完整 LeRobot 功能、其所有依赖声明或后续训练链路通过。没有修改包 metadata 来隐藏冲突。

Wall-X、LeRobot、DMuon 的跟踪源码均无本地 diff。Wall-X 目录内的未跟踪 `.so` 是安装生成的 CUDA 扩展，已记录哈希。已保存本机编译的 FlashAttention wheel 到 `artifacts/`；安装脚本优先复用该 wheel，避免在相同平台重复长时间编译。原生文件校验见 [native-artifacts.sha256](metadata/native-artifacts.sha256)。

## 历史环境复现命令（原 3060 Ti／CUDA 13.0 环境）

这些命令说明当时如何安装与检查，不是 RTX 5070 Ti 的部署步骤；不要把本机 sm80／sm86 扩展或旧激活环境作为目标主机的已验证环境。新部署使用上方链接中的 D0 派生镜像、完整匹配的 CUDA／Torch 工具链与 sm_120 重编译流程。

```bash
cd /home/pangk/workspaces/WALL-OSS-CRP
source scripts/activate.sh

# 2026-10-26 记录环境的检查；不进行完整模型推理。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_setup.py

# 全部本地文件校验；会顺序读取约 8.33GB 的权重文件。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/download_models.py --verify-only

# 需要重新下载安装时，使用 metadata 中的固定 revision。
python scripts/download_models.py

# 安装步骤及激活方式另见 environment.md。
bash scripts/install_runtime.sh
```

以下是 2026-10-26 原拟后续执行的官方 smoke test 命令，**该轮没有执行**，保留为历史命令样例。参数帮助已保存到 [fake-inference-help.txt](logs/fake-inference-help.txt)。

```bash
source /home/pangk/workspaces/WALL-OSS-CRP/scripts/activate.sh
cd /home/pangk/workspaces/WALL-OSS-CRP/wall-x
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/fake_inference.py \
  --checkpoint-path /home/pangk/workspaces/WALL-OSS-CRP/models/wall-oss-0.5 \
  --train-config-path /home/pangk/workspaces/WALL-OSS-CRP/models/wall-oss-0.5/config.yml \
  --norm-key x2_normal --action-mode flow \
  --cam-names face_view right_wrist_view \
  --image-size 448 --action-horizon 32 --seed 0
```

## 截至 2026-10-26 尚未验收的项目

官方预训练参数在 GPU 上完整加载、真实模型动作形状和有限值、目标配置重复推理、WebSocket 往返、冷启动及稳态性能、完整模型离线重启均未执行；本轮没有 `metrics.json` 或 `sample_actions.npy`。未启动推理服务，没有遗留本任务下载、编译或 GPU 检查进程。训练、多卡 FSDP、LIBERO rollout、真机执行与抓取能力均未验证。

当时的后续建议是在资源足够的主机继续旧任务书 D–F，并按需要处理 LeRobot 依赖。这不是当前任务清单；后续路线已由 [PROJECT_STATUS.md](PROJECT_STATUS.md) 与 [推理部署说明](WALL_RLT_Inference_Deployment.md) 接替。本历史环境检查不替代完整模型或目标主机验收。

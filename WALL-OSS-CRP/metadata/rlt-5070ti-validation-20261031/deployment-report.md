# WALL-OSS RTX 5070 Ti 部署汇报

日期：2026-10-31　主机：`crp-5070ti-01`　依据：`WALL-5070Ti-操作文档-20261031/WALL_RLT_5070Ti_Official_WALL_Runbook.md`

## 结论

**官方 WALL 完整模型已在 RTX 5070 Ti 上离线推理通过。**输入为合成观测，在两个独立容器中各运行一次，两次都输出 `32×26` 动作，数值全部有限，峰值显存 allocated 7.94 GiB。

这一结论有一个前提：第 7–8 节使用的是**白名单版检查器**，不是封包内的原版。原版检查器要求权重键完全一致，两次结果都是 FAIL。经核实，两类不一致都不影响推理结果：一是 `lm_head` 与 `embed_tokens` 共用权重，二是 safetensors 中多存了一份归一化统计。用户确认后改用白名单版检查器重跑。详见第 3 节。

本次只验证了官方 32×26 模型的合成观测推理，不包括 AR5 joint8 动作、RLT 服务、训练和机械臂执行。

## 1. 环境

| 项目 | 实际情况 |
|---|---|
| 系统 | Ubuntu 24.04.4 LTS，x86_64 |
| GPU | RTX 5070 Ti，16303 MiB，compute capability 12.0 |
| 驱动 | 595.91.07（`nvidia-smi` 显示 CUDA 13.2，表示驱动兼容能力） |
| Docker | 29.1.3，当前用户无需 sudo；`nvidia` runtime 已注册 |
| 容器内工具链 | Python 3.10，Torch 2.10.0+cu128，CUDA 12.8.1，FlashAttention 2.8.3 |
| 内存 | 30 GiB，另有 47 GiB swap |

## 2. 各步骤结果

| 节 | 内容 | 结果 |
|---|---|---|
| 1 | 主机检查 | ✅ 通过；磁盘可用空间不足，处理方式见 3.1 |
| 2 | 拷贝 `WALL-OSS-CRP-transfer-20261029` 与 `WALL-RLT-inference5070-20261029-v1` | ✅ |
| 3 | 包完整性校验 | ✅ D0：`27 files` 校验通过；源码包：`INTEGRITY_PASS`，6888 个源文件；镜像归档身份一致 |
| 4 | 加载 D0、容器访问 GPU、解压模型 | ✅ D0 镜像 `sha256:8f067a0e…`；容器内可见 5070 Ti |
| 5 | 离线构建 `sm_120` 镜像 | ✅ `wall-rlt-inference:5070ti-20261031T065922Z`（`sha256:f4199e5e…`），静态 `sm_120` 覆盖检查通过 |
| 6 | GPU 小算子检查 | ✅ 第 2 次通过：BF16 matmul PASS，FlashAttention PASS。第 1 次为 BLOCKED，原因见 3.2 |
| 7 | 官方 WALL 完整推理（两轮） | ⚠️ 原版检查器两轮 FAIL；白名单检查器两轮 PASS，见 3.3 |
| 8 | 报告汇总 | ✅ 基于白名单报告 |
| 9 | 打包回传报告 | ✅ |

### 完整推理指标（白名单检查器）

| | 第 1 轮 | 第 2 轮（新容器） |
|---|---|---|
| 状态 | PASS | PASS |
| 动作形状 | 32×26 | 32×26 |
| 动作取值范围 | −0.0492 ~ 1.421 | −0.0492 ~ 1.421 |
| 模型加载时间 | 38.8 s | 38.2 s |
| 首次推理时间 | 2.43 s | 1.39 s |
| 峰值显存 allocated / reserved | 7.94 / 8.07 GiB | 7.94 / 8.07 GiB |

首次推理时间包含首次运行开销，不能作为稳定推理速度或控制频率。检查器没有保存完整动作，因此不能据此判定两轮动作完全一致。

## 3. 问题与处理

### 3.1 磁盘空间不足

根分区最初只有 67 GB 可用，低于操作文档建议的 100 GB。清理了以下内容，可用空间增至 89 GB：

- 回收站 11 GB
- 10-27 残留的 5 个 Chrome 下载临时文件，共约 10.5 GB
- pip 缓存约 1.8 GB

各 conda env 和其他项目目录都没有动。部署完成后剩余约 **54 GB**。

### 3.2 容器内 uid 1000 没有用户记录

操作文档用 `--user 1000:1000` 运行容器，但镜像中没有该 uid 的 passwd 记录，且 `HOME=/` 不可写，导致两个问题：

- `torch._dynamo` 初始化时调用 `getpass.getuser()` 失败（`KeyError: getpwuid(): uid not found: 1000`），第 6 节因此 BLOCKED。
- Triton 无法创建 `/.triton` 缓存目录（`PermissionError`）。

处理方法：所有带 `--user` 的 `docker run` 增加以下参数，镜像和检查标准都不变。

```bash
-e USER=$(id -un) -e HOME=/tmp -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
```

这组参数已作为 `WALL_CONTAINER_ENV` 记录在 `session.env` 中。第 6 节第一次的失败报告保留不动。

### 3.3 权重键不一致（第 7 节）

原版检查器 `verify_full_model.py`（sha256 `cae9f11c…a8c9d6`）只要发现任何 missing 或 unexpected 键就判为 FAIL。实际加载时出现以下不一致：

| 类型 | 键 | 核实结果 |
|---|---|---|
| missing | `lm_head.weight`（1 个） | 配置为 `tie_word_embeddings=True`，加载后 `lm_head` 与 `embed_tokens` **共用同一块显存**，checkpoint 本来就不含该键 |
| unexpected | `action_preprocessor.normalizer_{action,propri}.{min,delta}.<数据集>`（共 224 个，覆盖 56 个数据集） | 运行时的归一化统计由 adapter 从 `normalizer_*.pth` 加载，再通过 `set_normalizer()` 设置到模型上；safetensors 中这份是重复副本。**224/224 个张量逐一比对，最大差值为 0** |

D0 包在 H100 上只做过环境检查（runtime.log 的 scope 为 "setup only; no full model load"），没有加载过完整模型。因此这个问题是本次首次暴露，与 5070 Ti 硬件无关。

**白名单检查器**：经用户确认，把官方检查器复制到本轮目录，只修改键检查部分（其余加载、推理、形状和数值检查保持不变），规则如下：

- missing 只允许 `lm_head.weight`，并且每次运行时实际验证 `tie_word_embeddings=True`、确认与 `embed_tokens` 共用显存。
- unexpected 只允许匹配 `normalizer_(action|propri).(min|delta).*` 的键，要求 checkpoint 目录中没有 `norm_stats.json`，并且每个张量与 `.pth` 中的对应值完全相等。
- 不满足以上任何一条，仍然判为 FAIL。

## 4. 产物位置

本轮目录：`/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt/runs/official-wall5070-20261031T063658Z-n0xT6K/`

| 内容 | 路径（相对本轮目录） |
|---|---|
| 会话变量（恢复用） | `session.env` |
| 全部报告与日志 | `reports/` |
| 原版检查器 FAIL 报告 | `reports/base-wall-full-model{,-restart}.json` |
| 白名单检查器 PASS 报告 | `reports/allowlist-wall-full-model{,-restart}.json` |
| 小算子检查 | `reports/host-kernel-check.json`（BLOCKED），`reports/host-kernel-check-retry1.json`（PASS） |
| 构建日志 | `reports/inference-build-20261031T065922Z.log` |
| 只读诊断 | `diagnostics/` |
| 原版与白名单检查器 | `checker-allowlist/verify_full_model.official.py`，`checker-allowlist/verify_full_model_allowlist.py` |
| 过程说明 | `reports/STOPPED_AT.txt` |
| 模型（已解压） | `models/`（7.8 GB） |

回传报告包（不含权重和数据）：

`/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt/official-wall5070-20261031T063658Z-n0xT6K-reports-20261031T081306Z.tar.gz`
SHA256 `34e416f3d9406a781a8d73ef3cb2f816bf8fe238a5fd2d34c6fe4cc19bbca1d5`

## 5. 给工具包开发者的建议

1. 在官方 `verify_full_model.py` 中加入第 3.3 节的白名单与核验逻辑，下一版工具包发布后用原版检查器复验一次，以去掉本次结论的前提条件。
2. 在操作文档所有带 `--user` 的 `docker run` 命令中补上第 3.2 节的环境变量；或者在镜像中设置可写的 `HOME`，并兼容任意 uid。
3. 操作文档第 1 节预留了 100 GB 空间，本次实际占用约 51 GB：拷贝两个包约 16 GB，加载 D0、解压模型和构建派生镜像约 35 GB（构建过程中的峰值未单独测量）。可据此修订预算。

## 6. 尚未覆盖的事项与下一步

| 事项 | 状态 |
|---|---|
| D0 依赖声明（`pip check` 11 项） | 仍为 FAIL，本阶段不处理 |
| 训练后 WALL＋RLT／Actor 权重 | 缺失，需先在 H100 完成 Stage 1 训练 |
| AR5 joint8（32×8）推理、RLT 特征、HTTP 服务、真实观测回放 | 未执行，等待训练权重 |
| 稳定推理时延、控制频率 | 未测 |
| 机械臂执行、抓取成功率 | 未验收 |
| Stage 2 实验参数 | 待用户决定 |

下一步：在 H100 上用 `top_error` 目标数据完成 Stage 1 训练并导出权重，然后回到本机，按 `WALL_RLT_Inference_Deployment.md` 第 7 节起验证 WALL＋RLT 推理。本机已构建的 `sm_120` 镜像可直接复用。

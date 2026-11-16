# WALL-OSS-CRP

**当前状态入口：[PROJECT_STATUS.md](PROJECT_STATUS.md)。** 以下摘要截至 2026-10-31；实现原理见 [详细审阅说明](WALL_RLT_Implementation_Review.md)，目标机操作见 [5070 Ti 推理部署说明](WALL_RLT_Inference_Deployment.md)。

后续 Agent 接手先读 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)：整合当前结论、交付位置、下一步入口、关键约束与历史记录边界。

WALL＋RLT 已实现目标数据适配、Stage 1 训练／恢复接口，以及权重导出入口、WALL＋Encoder 加载、HTTP 服务、已有观测回放与重启核验工具。Stage 2 已完成的运行仅为单步合成 CPU 检查。最新 [CPU 汇总检查](logs/rlt-inference-20261029/cpu-check/report.json) 为 **377 项 PASS，0 失败／错误／跳过**；源码锁、训练入口预检和合成 Stage 2 检查也为 PASS。真实微型 WALL 经 RGB 输入、混合精度权重恢复和 HTTP 往返已通过 CPU 测试。

**用户已确认：5070 Ti 的 `sm_120` 镜像构建、BF16／FlashAttention 小算子及官方 WALL 合成观测推理均已通过。** 完整模型使用受限键兼容检查器，在两个独立容器中各运行一次，输出均为 `32×26` 且数值有限，峰值显存 allocated 约 **7.94 GiB**；见 [5070 Ti 验证记录](metadata/rlt-5070ti-validation-20261031.json)。下一步是在 H100 完成 Stage 1 训练，再验证训练后的 WALL＋Encoder 目标推理。Stage 1 权重与 Stage 2 RL Actor 权重仍缺失，在线 RL、机械臂任务效果及最终 WALL-RLT 封版尚未完成。

## 已交付到 My Passport 的内容

| 包 | 字节数 | 内容与边界 |
|---|---:|---|
| [WALL-OSS-CRP-transfer-20261029](</media/pangk/My Passport/WALL-OSS-CRP-transfer-20261029>) | 16,787,299,846 | D0 基础 Docker 环境、官方模型、历史验证数据；服务器待验 |
| [WALL-RLT-training-candidate-20261029-v1](</media/pangk/My Passport/WALL-RLT-training-candidate-20261029-v1>) | 5,781,431,598 | 训练源码、44 个离线 wheel、350 条目标轨迹；不含训练后 RLT 权重 |
| [WALL-RLT-inference5070-20261029-v1](</media/pangk/My Passport/WALL-RLT-inference5070-20261029-v1>) | 10,979,923 | 8 个包文件、6,888 个源码成员；离线构建和推理工具，不含已编译的 5070 Ti 镜像 |

三包合计 **45 个文件、22,579,711,367 字节（约 22.58 GB）**。此前 D0＋训练候选两包的 **37 文件、22,568,731,444 字节**仍是这两包的正确小计，不包含新推理工具包，也不包含包外旁置说明。三包内部保持原样。新包的整包校验、实际解包和 portable CLI 入口检查 PASS，证据见 [推理部署交付记录](metadata/rlt-inference-deployment-20261029.json)。

D0 的整包校验与构建机组件检查见 [基础交付记录](logs/server-environment-delivery-20261029.json) 和 [交付验收报告](logs/docker-validation-delivery-20261029/)。旧 `WALL-OSS-CRP-transfer-20261028` 是保留的 INCOMPLETE 目录，不可部署。D0 仍有 11 项依赖声明问题，`pip check` 为 FAIL；不能把局部组件通过写成完整训练依赖验收。

## 主机、数据与下一步

推理主机是 Ubuntu `crp-5070ti-01`，**RTX 5070 Ti 16GB（16303 MiB）、驱动 595.91.07**，目录 `/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt`。用户已通过 My Passport 拷贝，在目标机完成离线镜像构建、小算子及官方 WALL 基线验证；本机不通过 SSH 部署。后续在训练后的 WALL／Encoder 权重就绪后验收 joint8 推理、RLT 特征和真实观测回放。具体步骤见 [部署说明](WALL_RLT_Inference_Deployment.md) 和 [主机配置](configs/inference/rtx5070ti.json)。

训练服务器资源为用户确认的 **8 张 H100**；单卡显存和当前作业分配尚未实测，现有 [双卡 FSDP 候选](configs/rlt/stage1_2gpu.yaml) 不变。本机不连接 H100；用户自行经电脑和专用网线中转。此前 SSH 超时只是历史记录，不是继续尝试连接的指令。H100 历史网络信息见 [使用说明](H100使用说明.md)，不套用于另一台推理主机。

训练数据已采集完成。当前使用 `My Passport/top_error` 七类共 **350 条轨迹**，排除 `mouse`、`tape`，忽略历史验证集；字段语义、下一帧标签、划分、normalizer 和代表性真实 batch 已完成 CPU 核验。不安排新采集。详见 [数据格式](数据集格式说明.md)、[数据契约](configs/rlt/top_error_ar5_joint_next_v1.json) 和 [batch 报告](metadata/rlt-top-error-selected-20261029/cpu-batch.json)。缺少真实奖励／终止标记，因此 Stage 2 仍仅有单步 synthetic CPU 证据。

已实现的 Stage 1 推理接口定义输出为 **WALL 参考动作 `32×8` 和可选 `z_rl`**，导出产物包含 WALL＋Encoder，不含 RL Actor／Decoder，不执行机器人。Stage 1 不产生训练后的 RL Actor，后者需另行完成 Stage 2 训练，其关键参数由用户决定。训练后的 WALL＋Encoder 在目标 Ubuntu 完成验收后，才冻结通过验证的代码、依赖、权重、配置与报告并制作最终成品；后续服务器迁移复验仍是独立验收步骤。

## 文档索引

- [项目当前状态与证据入口](PROJECT_STATUS.md)
- [实施方案简版：分工、已完成内容与剩余验收](WALL_RLT_Human_Review.md)
- [具体实施计划：代码、训练配置和 GPU 验收步骤](WALL_RLT_Agent_Implementation_Plan.md)
- [详细实现审阅：RLT 机制、术语与代码依据](WALL_RLT_Implementation_Review.md)
- [5070 Ti 环境构建与官方 WALL 复验命令](WALL_RLT_5070Ti_Official_WALL_Runbook.md)
- [5070 Ti 离线部署与推理验收](WALL_RLT_Inference_Deployment.md)
- [环境及 D0 迁移记录](environment.md)
- [2026-10-28 静态源码审计快照](metadata/wall-rlt-design-audit-20261028.json)
- [历史数据核验／采集计划，采集路线已取消](WALL-OSS-CRP_Stage1_Agent_Plan.md)

## 2026-10-26 构建机准备记录

这一节是历史复查入口，不能替代上述最新状态。当时完成了官方仓库、独立环境、WALL-OSS-0.5 权重和 processor 下载及完整性检查；构建机为 RTX 3060 Ti 8GB，未进行完整模型加载、动作推理或 WebSocket 验收。

```bash
cd /home/pangk/workspaces/WALL-OSS-CRP
source scripts/activate.sh

# 复查组件与 processor，不加载完整模型。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_setup.py

# 仅读取本地文件，复查原模型清单。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/download_models.py --verify-only
```

对应历史证据：[stage1_report.md](stage1_report.md)、[sources.lock.json](sources.lock.json)、[environment.freeze.txt](environment.freeze.txt)、[model_files.manifest.json](model_files.manifest.json)、[setup-verification.json](setup-verification.json) 和 [configs/assets.json](configs/assets.json)。当时记录的权重存储约 7.76 GiB，不能据此承诺在构建机剩余显存中完成整模型推理；LeRobot 的依赖声明问题也一直单独记录。

# WALL-OSS-0.5＋RLT：实施方案简版

更新：2026-10-31。**当前状态以 [PROJECT_STATUS.md](PROJECT_STATUS.md) 为入口**；本文概括分工、已有实现和后续验收。详细机制及术语见 [人类审阅说明](WALL_RLT_Implementation_Review.md)，代码与命令见 [具体实施计划](WALL_RLT_Agent_Implementation_Plan.md)。

## 当前交付与分工

D0 基础环境、H100 训练候选和 **5070 Ti 推理工具包均已在 My Passport 上 READY**。最新 [CPU 报告](logs/rlt-inference-20261029/cpu-check/report.json) 包含 **377 项通过，0 失败／错误／跳过**，源码锁、训练预检和合成 Stage 2 检查也通过。推理工具的完整校验、实际解包及 portable 入口检查通过，见 [交付记录](metadata/rlt-inference-deployment-20261029.json)。

| 已交付包 | 字节数 | 状态边界 |
|---|---:|---|
| `WALL-OSS-CRP-transfer-20261029` | 16,787,299,846 | 基础镜像、官方模型和历史验证数据；H100 待验 |
| `WALL-RLT-training-candidate-20261029-v1` | 5,781,431,598 | 训练源码、44 个 wheel、350 条目标轨迹；训练记录为 NOT_RUN |
| `WALL-RLT-inference5070-20261029-v1` | 10,979,923 | 8 个文件、6,888 个源码成员；目标机离线构建工具，未附目标镜像或训练后权重 |

三个包合计 **45 文件、22,579,711,367 字节**。旧 D0＋训练候选的 **37 文件、22,568,731,444 字节**是这两包的小计；原 [H100 中转说明](</media/pangk/My Passport/WALL-H100-中转说明-20261029.txt>) 和 [旧两包清单](</media/pangk/My Passport/WALL-H100-SHA256SUMS-20261029.txt>) 保留其原范围，不作为新三包的统一清单。所有包内部保持原样，包外说明不计入以上总数。

| 工作 | 当前约定 |
|---|---|
| 数据 | 已采集；只使用已有数据，不安排试采／扩采 |
| 训练 | 服务器共 8 张 H100，单卡显存与当前作业分配尚未实测；双卡 FSDP 候选保持不变，本机不连接服务器 |
| 推理 | Ubuntu `crp-5070ti-01`，RTX 5070 Ti 16GB／16303 MiB，驱动 595.91.07，目录 `/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt` |
| 部署执行 | 用户已从 My Passport 拷贝，在目标机完成 `sm_120` 构建、小算子及官方 WALL 基线验证；本机不经 SSH 部署，见 [部署说明](WALL_RLT_Inference_Deployment.md) |
| 机器人／在线 RL | 服务只返回模型预测，不执行机器人；在线部署位置、奖励与环境交互尚未实施 |

**用户已确认 5070 Ti 环境构建及官方 WALL 完整模型推理通过。** `sm_120` 派生镜像构建、BF16／FlashAttention 小算子均已完成；完整模型使用受限键兼容检查器，在两个独立容器内分别对合成观测推理，均输出有限数值的 `32×26` 动作，峰值显存 allocated 约 **7.94 GiB**。依据见 [5070 Ti 验证记录](metadata/rlt-5070ti-validation-20261031.json)。该结果覆盖官方 WALL 基线，不等于 AR5 joint8、RLT 服务或机械臂任务效果验收。

**下一步进入 H100 Stage 1 训练。** 项目尚无训练后的 Stage 1 WALL＋Encoder 和 Stage 2 RL Actor 权重；Stage 1 不产生训练后的 RL Actor。后续目标推理、在线 RL 和最终封版仍待完成。基础包仍有 11 项已知依赖声明问题，`pip check` 为 FAIL；旧 `WALL-OSS-CRP-transfer-20261028` 目录为 INCOMPLETE，不可部署。

## 已实现的数据与模型链路

目标数据为 `My Passport/top_error`，排除 `mouse`、`tape`，忽略历史验证集。七类物体各 50 条，共 **350 条、176,523 帧**；按物体分层划分 train/dev/test 为 245/49/56 条。用户确认数据来自未来推理使用的主机和同一机械臂。字段、语义和代表性 batch 的 CPU 检查已通过，全部视频逐帧质量与逐条抓取结果标注不在已通过范围内。

首版 `ar5_joint_next_v1` 用当前图像／状态监督下一帧的七个实测关节角和二值夹爪指令；去掉每条无未来标签的最后一帧，得到 176,173 个有效观测。原始 7 维末端增量是上一帧到当前帧的实测变化，不能直接冒充未来关节命令；实测夹爪开度也不能冒充开合指令。见 [数据契约](configs/rlt/top_error_ar5_joint_next_v1.json)、[数据审计](metadata/rlt-top-error-selected-20261029/audit.json)、[CPU batch](metadata/rlt-top-error-selected-20261029/cpu-batch.json)。

`ar5_joint8_reinit_v1` 明确重定义 26 维模型接口的前 8 个槽位。新迁移训练加载底座后重置动作输入／输出投影，恢复和推理不重置。实现和微型测试已完成，完整模型迁移训练尚未执行。

Stage 1 同时计算 `L_total = L_rlt + alpha × L_vla`，首版 alpha=1。WALL 由 flow 动作损失更新；Encoder／因果 Decoder 由观测前缀重建损失更新，重建分支对 WALL 停止梯度。全局梯度裁剪可能在优化尺度上间接耦合两分支，因此不能声称加 RLT 后的 WALL 更新必然与纯 SFT 完全一致。

Stage 2 已实现冻结特征和单步 `z_rl + proprio + ref_chunk` 接口，并调用固定 RLinf RLT-AC 完成合成 CPU 的 Actor／Critic 各一次更新。该证据没有训练后的 WALL 特征，没有真实奖励 replay，也没有在线收益。现有示范未提供 reward／terminated／truncated，不能补造这些字段来宣称 RL 训练完成。

已实现的推理工具提供 Stage 1 权重导出入口、WALL＋Encoder 严格加载、HTTP 输入输出、已有观测回放和不同服务实例的重启比较。真实微型模型在 CPU 通过这些工具的集成测试，尚无训练后 artifact 或目标服务实跑报告。接口定义输出为 **32×8 的 WALL 物理空间参考动作及可选 z_rl**；当前不含 Actor、Critic、训练 Decoder 或优化器，Encoder 特征不会自动改写 WALL 动作。H=32 是预测长度，Stage 2 单步 C=1 是另一套执行／transition 约定。

## 已完成的软件工作与剩余验收

| 工作包 | 已完成及验证范围 | 尚需验收 |
|---|---|---|
| P0 扩展与依赖 | 配置、builder、外部注册、新进程导入及预检 | H100 独立依赖与实际作业资源 |
| P1 数据适配 | 真实轨迹、时间对齐、split、统计、mask、官方 processor batch | 完整模型上的目标数据训练 |
| P2 前缀与 RLT | 无目标泄漏、梯度隔离、因果重建、训练／prefill 一致、原路径回归 | 全尺寸 GPU 数值与性能 |
| P3 训练与恢复 | SFT worker、短训配置、小模型独立进程恢复 | H100 单步、100 步、实际 FSDP 分片保存／恢复 |
| P4 Stage 2 | C=1 转换、终止／截断语义、合成 AC 更新 | 训练后特征、真实交互与收益 |
| P5 推理与交付 | 导出、runtime、HTTP、回放与交付工具通过 CPU 检查；5070 Ti 构建、小算子及官方 WALL 合成观测推理由用户确认通过 | Stage 1 训练权重；WALL＋Encoder 的 AR5 目标推理、真实观测回放；最终封版 |

当前 [双卡 FSDP 候选](configs/rlt/stage1_2gpu.yaml) 保留模型规模和有效 batch=4。静态状态下界为未分片 73.15 GiB、理想双卡约 36.57 GiB／卡，均不含激活等开销；不能替代实测。双卡候选关闭 dev 循环；单卡 64 窗口 smoke 验证也不能当作多卡或全量 dev 评估。

## 接下来的执行顺序

1. 用户在已分配的 H100 上完成训练依赖和资源核验，再运行完整模型单步、100 步及分片保存／恢复，用 `top_error` 目标数据完成 Stage 1 训练，生成匹配的权重。本机不连接 H100。
2. 导出 WALL＋Encoder artifact，在 5070 Ti 验证 AR5 观测到动作／特征、时延、峰值显存及重启一致性。现有 `sm_120` 镜像在模型及预处理源码匹配时可复用；训练后模型仍按其真实输入单独验收。导出验证和 artifact 哈希不替代模型质量评测。
3. 目标端完整链路实际 PASS 后，冻结代码、依赖、权重、processor、normalizer、配置与报告，制作最终成品；后续服务器复验同一版本。当前三份 READY 包保持原样，不表示最终封版已经完成。

Stage 2 的 reward、warmup、RL 更新、transition 和 chunk 等关键设计由用户决定，再实施真实在线训练并产生 RL Actor。Stage 1 与官方 WALL 基线通过不替代这项工作。

历史检查保留其时间与范围：2026-10-29 首轮实现 [147 项 CPU 检查](logs/rlt-cpu-check-20261029/report.json)、同日训练准备阶段 [218 项 CPU 检查](logs/rlt-h100-preparation-20261029/cpu-check/report.json)，以及同日推理实施完成后的 [377 项检查](logs/rlt-inference-20261029/cpu-check/report.json)。最新数字不是前三者求和；历史报告不改写为新验收结果。

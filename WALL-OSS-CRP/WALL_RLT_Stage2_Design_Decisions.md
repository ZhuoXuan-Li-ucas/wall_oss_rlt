# WALL＋RLT Stage 2 设计决策清单

更新：2026-10-29。用户明确要求掌控 reward、warmup、RL 更新、transition、RL chunk 等关键设计。**正式实验设计由用户决定；助手负责整理备选及其影响、实现配置和验证。当前合成测试值不自动成为正式实验参数。**

本文是待决策清单，不是已接入的可执行配置。项目整体状态见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，已有机制说明见 [实现审阅文档](WALL_RLT_Implementation_Review.md)。

## 0. 2026-11-05 更新：进入 Stage 2 前的状态、步骤与决策表（优先于下文）

用户 11-05 告知：H100 Stage 1（bolt）训练权重正在拷贝，接下来进入 RLT Stage 2。本节记录当时的本机状态、权重到位后必须按顺序做的事，以及 Stage 2 的决策表。**11-05 下午补充：** 用户决定 Stage 2 尽量复用 Evo-RLT 源码并设对照组调参，完整的科研计划（阶段、工作包、真机协议、对照组决策树、调参手册、待确认决策 D1–D13）在 `WALL_RLT_Stage2_Research_Plan.md`；本节 0.3 的候选列仍待定版。表中“论文”一列来自 RLT 论文（arXiv 2604.23073v2，HTML 版，2026-04-30）及 PI 博客，“候选”一列只是整理出来的备选，**没有任何一项已经由用户定版**。

### 0.1 本机状态（5070 Ti，2026-11-05 上午）

| 事项 | 状态 |
|---|---|
| GPU | **不可用，需重启。** 06:47 无人值守升级把 NVIDIA 驱动从 595.91.07 升到 595.99.02 并安装了内核 7.0.0-42；运行中的内核模块仍是 595.91.07，`nvidia-smi` 与 `nvidia-container-cli` 都报 driver/library version mismatch，`/var/run/reboot-required` 已存在。重启由用户执行；重启后应进入 7.0.0-42 并加载 595.99 模块。权重拷贝不在本机进行，重启不影响 |
| 推理镜像 | **必须重建。** 开发工作区与现有镜像 `wall-rlt-inference:5070ti-20261031T065922Z` 的构建源码有 13 个 `wall_rlt` 文件不同，其中 `model/builder.py`（补零修复，sha `f83b2aeb…`）、`data/ar5.py`、`inference/runtime.py` 都在 `MODEL_SOURCES` 内，旧镜像会拒绝加载新训练产物。已从当前工作区打新工具包 `yuhang_workspace/wall_oss_rlt/incoming/WALL-RLT-inference5070-20261105-v2`（INTEGRITY_PASS，6893 个源文件），并按运行手册第 5 节在后台重建；构建不需要 GPU。新镜像的 tag／ID 以 `WALL_INFERENCE_IMAGE_TAG_V2`／`WALL_INFERENCE_IMAGE_ID_V2` 追加在 `runs/official-wall5070-20261031T063658Z-n0xT6K/session.env`，旧的已验证镜像 ID 不动；新镜像的 GPU 小算子与官方推理复验要等重启后再跑 |
| 权重 | 尚未到本机；最终形态未知：可能是 H100 上已导出的 artifact 目录，也可能是原始 RLinf checkpoint `global_step_N/actor/` |
| Stage 2 代码 | 仍只有 C=1 单步合成检查（`wall_rlt/stage2.py`）；没有在线主循环、真实 replay、机器人环境封装、奖励录入 |
| 执行器 | `yuhang_workspace/wall_oss_rlt/tools/wall_execute.py` 只在模拟臂上验证过；默认走到第 10 步（0.33 s）就去要下一段，切段渐变 0.1 s |

### 0.2 权重到位后、Stage 2 之前必须按顺序完成的步骤

1. **重启本机**，确认 `nvidia-smi` 显示 595.99，`docker run --gpus 'device=0' … nvidia-smi -L` 能列出 5070 Ti。
2. **权重入库。** 两种情况二选一：
   - 已导出 artifact（含 `wall.safetensors`、`encoder.safetensors`、`assets/`、`manifest.json`）：放到 `yuhang_workspace/wall_oss_rlt/artifacts/<名称>/`，逐文件核对 H100 侧的 SHA256 清单。
   - 原始 checkpoint `…/global_step_N/actor/`（须含 `wall_rlt_data.json`、`wall_rlt_assets/`、`model_state_dict/full_weights.pt`）：在本机 `.venv-rlt` 里用 CPU 导出，`python scripts/export_wall_rlt_inference.py --checkpoint …/actor --output …/artifacts/<名称>`。`full_weights.pt` 为 FP32，约 20 GB，导出用 mmap 读取，本机 30 GB 内存够用；不需要 GPU，不需要镜像。
   - 两种情况都要注意：artifact 记录了训练时 `MODEL_SOURCES` 各文件的哈希，服务启动时与镜像内源码逐一比对，不一致会拒绝加载。本工作区 `builder.py` 已打同一补丁（`f83b2aeb…`），若 H100 侧还改了别的文件，须先同步到本工作区再重建镜像，不得绕过检查。
3. **新镜像 GPU 验证**：运行手册第 6 节小算子检查；第 7 节官方模型推理可省略，直接进入下一步。
4. **训练产物的目标推理**（部署说明第 7–9 节）：`serve_wall_rlt.py --check-only` → 生成回放契约（`root` 改为 `~/atom_episodes/wall_unlocked_v2`，bolt 契约的 dev episode 为 14、41）→ 起服务 → `check_wall_rlt_inference.py` 同 seed 两次 → 重启服务再比。EE 契约下动作是 `32×10`（不是部署说明里 joint8 的 `32×8`），响应带 `absolute_targets`，`z_rl` 为 2048 维。记录时延与峰值显存：官方模型在本机是中位 227 ms、7.9 GiB，训练后应相近。
5. **真机 Stage 1 基线（RLT 的决定性指标）**：`wall_execute.py` 先 `--fake-policy hold`，再 `--fake-policy replay:<bolt episode>`，最后接 `--server`；在 bolt 任务上按固定起始状态跑一批（建议 20 条起），人工记录成功率与完成时间。目标区间大约 20–70%。明显低于 10% 先回到数据／训练，明显高于 90% 则 RLT 的收益主要在速度而非成功率。该成功率同时是 Stage 2 的对照组，必须与 Stage 2 用同一套起始状态与成功判定。
6. 用户按 0.3 定版 → 实现 0.4 的缺口 → 在线运行。

### 0.3 Stage 2 决策表：论文参考值与本项目候选

控制频率、预测长度等“本项目现状”来自契约 `wall_v2_bolt_ar5_ee_rel6d_next_v1`：30 Hz，H=32（1.07 s），state 10 维（法兰位置 3＋rot6D 6＋实测开度），action 10 维（相对位置 3＋相对 rot6D 6＋二值夹爪指令），槽位 10–19。论文：50 Hz，H=50（1 s），14 维动作。

| 项目 | RLT 论文（附位置） | 本项目现状／约束 | 候选（待用户定） |
|---|---|---|---|
| 任务与指令 | 单任务固定指令；学 token 时丢弃语言（IV-A 脚注） | bolt 专用模型，指令固定 "insert the bolt into the sleeve"；WALL 仍收指令 | 沿用固定指令，不做语言变体 |
| RL chunk 长度 C | C=10 @50 Hz＝0.2 s，C<H 为了更灵敏（III、附录 B）；Alg. 1 每 C 步重新观测、整段开环执行 | 代码只支持 C=1；推理一次约 227 ms（10 步流匹配）；执行器默认第 10 步重规划 | C=8（0.27 s）或 C=10（0.33 s）@30 Hz；C 必须盖过推理时延，否则要像执行器那样提前请求下一段。执行步数 L=C |
| 参考动作块 | Actor 收 ã_{t:t+C−1}（Alg. 1 第 7 行），取 VLA 预测的前 C 步 | `extract_rlt_obs` 只截第 1 步 `[B,1,A]` | 改为前 C 步 `[B,C,A]`，与 replay 中动作同一空间 |
| 本体感受 proprio | 位置＋速度（附录 B）或末端位姿（VI-A），维数未说 | state 10 维，均经 WALL 归一化前的物理量 | 10 维物理 state，或再加有限差分速度 10 维；须与 `extract_rlt_obs` 的 proprio 一致 |
| 动作空间与边界 | 14 维 delta 动作，高斯未提 tanh | RLinf Actor 经 tanh 输出 `[-1,1]`，`BoundedActionTransform` 需显式每维上下界，越界报错不限幅。train 集每步相对量 q01／q99：dx −0.057／0.027，dy −0.068／0.060，dz −0.063／0.103 m；rot6D 相对项 r00、r11 在 0.985–1.0，其余 ±0.13；夹爪 0／1。执行器有 5 cm／20° 偏离保护 | 以 train 集 min／max 外扩约 20% 作边界并写成版本化配置；分位数不是控制边界，边界须用户对照工作空间盒与偏离保护确认。rot6D 六个数由 Actor 直接输出时须在解码处做 Gram–Schmidt 正交化；夹爪连续输出按 0.5 取阈（执行器用 0.4／0.6 滞回） |
| 探索噪声 | 固定小标准差，数值未说（附录 B） | RLinf `fixed_std=0.002`，作用在 tanh 前的 RL 空间，不是物理单位 | 先干跑一次把 RL 空间噪声换算成 mm／度再定；候选 0.002–0.01 |
| 参考动作 dropout | 训练时 50% 置零，推理时始终提供（附录 B、IV-B） | RLinf `reference_dropout_prob` 默认 0 | 0.5 |
| Actor／Critic 结构 | 2 层 MLP，隐层 256；螺丝任务 3 层 512（附录 B）；双 Q，目标取 min（附录 B） | RLinf `RLTMLPPolicy`，默认结构见 0.4 | 先用 2×256 |
| Critic 目标与折扣 | Q̂=Σ_{t'=1}^{C} γ^{t'−1} r_{t'} + γ^C min Q'(x',a')，a'∼当前 Actor（式 3）；γ 未说 | RLinf 按 reward 张量宽度算 γ^k，提前终止的短块必须有有效步 mask（见第 4 节） | γ 按基础步定义（30 Hz）；候选 0.99（C=8 时块级 0.92）。关键阶段 10 s＝300 步，γ=0.99 下终端奖励传回起点约 0.05，可接受但要监控 Q 尺度 |
| Actor 损失 | −Q + β‖a−ã‖²（式 5），β 未给，β=0 是最大单项退化；每个 Actor 更新配 2 次 Critic 更新（V） | RLinf `q_weight`、`bc_weight`，合成检查用 1／1 | q 1、bc 1 起，critic:actor 更新 2:1 |
| UTD、子采样 | UTD 约 5（HTML 印成 55，疑为重复数字）；replay 采样 stride 2（V、附录 B） | 无项目级调度 | UTD 5；异步学习进程与采集解耦 |
| warmup | 先用 VLA 跑 N_warm 个环境步填 replay，之后很快开始学习（V、Alg. 1）；N_warm 未给 | 无 | 先 10–20 条纯参考策略 episode（Actor 零初始化／直通）再开始更新；期间谁出动作、哪些网络更新须分别写明 |
| replay 容量、batch、lr、优化器、target 更新 | 均未说；遵循 TD3（IV-B） | RLinf 默认见 0.4 | 沿用 RLinf 默认，写进版本化配置 |
| 奖励 | 操作员在 episode 结束时标成功／失败，成功 r_T=1 否则 0（III、附录 B）；无过程项、无惩罚 | 无奖励入口；实验室 Evo-RLT 用过键盘 r／s／f | 键盘：r＝成功（terminated）、f＝失败（terminated）、s＝截断（truncated，不关 bootstrap）；时限到也是 truncated |
| episode 与复位 | 从关键阶段前的半完成状态开始，初始状态略随机；关键阶段 5–20 s；人决定何时把控制从 VLA 交给 RL；可人工干预（VI-A、V） | bolt：末段 30 mm 中位 5.7 s；采集时插入后即松手；套筒固定，松手点各轴 std 约 8 mm；插入时夹爪斜 50° | 起始状态＝已夹住螺栓、悬在套筒上方 3–5 cm，位置 ±1 cm 随机；由主臂遥操摆位，或按论文让 WALL 从 0 位跑到该处再由操作员交接。时限候选 20 s。复位方案 A：episode 在“已插入”判成功后结束，不松手，脚本抬起 5 cm 直接下一条（几乎零人工）；方案 B：松手后人工把螺栓放回支架再抓。A 更快，但 Stage 2 的动作就不含松手。**11-05 用户定版：从拿螺栓开始（采集同款起点：`centre_joints`、夹爪张开、螺栓在蓝色座里），成功＝把螺栓放进套筒** |
| 可观测性风险 | 论文靠 VLA 图像特征 | 关键阶段俯视只见约 20 px 套筒边、腕部只见螺栓头、无 F/T（见 rlt-scenario 记录） | 先看 Stage 1 基线；若基线对位靠猜，RLT 也学不到，届时再考虑侧视相机（但 Encoder 没见过该视角，需重训） |
| 评估 | 关键阶段设定下每任务 50 条（VI-A）；指标成功率与每 10 分钟完成次数 | 无 | 对照 WALL 单独 vs WALL＋Actor，各 20–50 条，同起始状态协议、同判定 |
| 交互预算 | 每任务 400–1000 条，机器人数据 15 分钟–5 小时（VI-B）；以太网任务总计 2 小时训练（博客） | 关键阶段约 10 s＋复位约 10 s | 400 条约 2.5 小时纯交互，分 2–3 次现场完成 |
| 算力分配 | 异步 rollout／学习（V） | 5070 Ti 16 GB：WALL＋Encoder 服务约 7.9 GiB，Actor／Critic 很小 | 同一张卡：服务进程＋学习进程；特征冻结，replay 只存 z_rl、proprio、参考块、动作、奖励、标志、时序，图像另存供审计 |

### 0.4 实现缺口（依据固定版 RLinf `e1a82ebf` 的只读分析，2026-11-05）

**RLinf 已有，可在单机、不启动 Ray／FSDP 的情况下直接复用**（import 时需要 ray 已安装，但不需要集群；损失路径不依赖 `torch.distributed`）：

- 网络与损失：`RLTMLPPolicy`（`rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py`）：backbone 256-256-256 tanh，Actor 输出 `C×A`，双 Q 头各 256×3；Critic 输入 `z_rl + proprio + 展平动作`，**不含参考块**；`fixed_std` 默认 0.002，作用在 tanh 之前；参考 dropout 按样本把整块置零（`:107-118`），rollout 推理不做 dropout。`RLTACLossMixin`（`rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py`）：Critic 目标 `Σ_k γ^k r_k + (1 − 块内任一步 terminated)·γ^C·min Q'(x', π(x'))`，其中 `γ^k`、`γ^C` **完全由 reward 张量宽度决定**（`:87-95, 270-276`），`truncations` 不参与 target（正确的时限处理）；Actor 损失 `−q_weight·Q1(x, π) + bc_weight·MSE(a, 人工动作或参考块前 C 步)`（`:301-368`），`actor_weight_schedule` 可按 `update_step` 从 warmup 权重线性过渡到 online 权重（`:148-225`）。
- 调度与存储：`TrajectoryReplayBuffer`（均匀采样，只在最近 `sample_window_size` 条轨迹内，可存盘恢复，无优先级）；`SAC` 基类的 `update_one_epoch`（critic → 每 `critic_actor_ratio` 次 actor → 每 `target_update_freq` 次 `tau` EMA）；RLT 调度项 `warmup_min_size`（单位是 replay 行数）、`warmup_post_collect_updates`、`train_every_transitions`／`train_every_episodes`、`update_epoch`、`max_updates_per_train_step`；warmup 后 UTD = `update_epoch / train_every_transitions`。
- 真机相关：`RealworldRLTRoute`（按键 `b` 后整块执行 Actor 动作，否则执行参考块）、`KeyboardRLTPolicySwitchWrapper`（evdev）、autoreset 的 `final_observation` 处理、人工干预替换已执行动作并同时写进 `ref_chunk`（BC 目标两条路都指向人工动作）、spacemouse／gello 等遥操接入。
- RLinf 自带的真机示例配置 `examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml`：C=10、gamma 0.96、tau 0.005、lr 3e-4／3e-4、batch 256、critic:actor 4:1、每个 runner step 8 次更新、q 0.1、bc 5、`max_episode_steps` 300、proprio 19 维。**这是上游示例，不是本项目默认。**

**需本项目实现：**

- 单机 trainer：optimizer 分组（actor = backbone + actor_mean；critic = q_head；`actor_logstd` 冻结）、上述更新顺序与梯度裁剪、`update_step` 与 warmup／UTD 预算计数、checkpoint（model + target + 两个 optimizer + replay）。规模：D=10、C=8、A=10 时 Actor＋双 Q 约 2M 参数，显存可忽略。
- AR5＋OmniPicker 环境（gym 风格 `reset/step/chunk_step`，`num_envs=1`）：观测→推理服务（`z_rl`、`absolute_targets`／`actions`）→ `action_transform`／`slot_map` 转到 RL 空间；RL 动作→物理目标→执行器那条 50 Hz 插值路径；硬件复位与时限。RLinf 的 real 包只注册了 dosw1／franka／gim_arm／piper／so101／xsquare／yam，没有 ROKAE。
- **多步 chunk 的有效步 mask（C>1 时必做）**：RLinf 没有 valid mask；real 的 `chunk_step` 在 terminated 后继续执行剩余步并把 termination 折叠到最后一个 slot（`rlinf/envs/real/env.py:410-425, 457-462`），terminal 之后的 reward 会被累加进 target，`γ^C` 也不变；sim 环境有冻结逻辑，real 没有。本项目要在 env wrapper 内块内早停、把 terminal 之后的 reward／步清零，并把 bootstrap 折扣改成 γ^(有效步数)，或者保证块内终止只能发生在最后一步。第 4 节的公式即此要求。
- `extract_rlt_obs` 改为返回前 C 步参考块（当前 `policy.py:107` 只截 1 步），并满足 `ref_num_action_chunks ≥ C`。
- 奖励录入：RLinf real 的三种来源（位姿阈值、reward model、键盘打分）中，键盘打分 wrapper 与 RLT 的 `b` 切换 wrapper 占同一个槽位、互斥。本项目要自己写一个同时提供 r／f／s 打分与 `b` 切换的键盘 wrapper，或者不用 `b` 切换、改用固定起点协议。
- Actor 接管门控：real route 没有任何 warmup 门控，按 `b` 后哪怕 MLP 刚随机初始化也立即执行；sim route 才有按 `update_step` 的门控。需要实现门控，规则由用户定。
- replay 过滤：real 分支把切换前由参考策略执行的块也全部入库（`fsdp_rlt_ac_policy_worker.py:641`），sim 分支按 `record_transition` 过滤；要不要入库由用户定，过滤由本项目实现。

**可借鉴的 Evo-RLT 实现（用户 2026-11-05 指示：Stage 2 实现可借鉴 RLinf 与 Evo-RLT 的代码）。** 来源是 My Passport 的 `wy/EVO/Evo-RLT.zip`（8 月 3 日版，含权重共 52 GB；盘根目录的 `Evo-RLT.zip` 是 7 月 28 日旧版）。它基于 LeRobot、pi0.5 和 SO-101，接入 WALL 推理服务与 AR5 需要写适配层。以下按只读阅读记录：

- 块内提前终止：`adapters/lerobot/online_collector.py` 在成功或失败时立即关闭当前块，记录 `actual_steps`，后续步补零，终端奖励写在最后一个有效步；`core/losses.py` 的 critic 目标用 `γ^actual_steps·(1−done)` 做 bootstrap。这正好覆盖上文"多步 chunk 的有效步 mask"缺口。
- 执行层：`policies/action_modifier.py` 中 VLA 段执行 `chunk_exec_steps`（默认 25）步，RL 段由 actor 输出并执行 C 步；`action_clip_delta` 把 actor 输出限制在参考块 ±delta 内；`core/actor.py` 的 `residual_to_ref` 把最后一层零初始化，未训练的 actor 输出恰好等于参考块。
- 交接与按键：`r` 第一次进入关键阶段、第二次判成功，`u` 判失败，`m` 为 milestone，`i`／Space 为人工接管；关键阶段结束后立即切回 VLA，之后的 VLA 动作不进 RL buffer。`core/phase_controller.py` 有手动和学习式两种交接，学习式是 z_rl 上的二分类器，对应论文第 V 节"测试时自动切换"。
- 训练调度：`record/online_trainer.py` 有 warmup 门槛（episode 数、transition 数、成功数、失败数）、critic-only 窗口、按 episode 结算的 UTD、完整状态存盘恢复，以及从在线 buffer 删除错误 episode 的工具。
- 离线数据：`cli/build_transition_cache_v2.py` 用 `--frame-stride 2` 从示范数据构建固定离线 buffer，训练时与在线 buffer 按比例混采。论文没有这一项。在线采集只在块边界切 transition，没有做 stride。
- **与论文不同、不能直接当本项目默认**：milestone 中间奖励、按块计的 `time_decay`、离线示范 buffer 混采、RankQ 排序损失、target Q 截断、critic LayerNorm；residual actor 的参考 dropout 只清零网络输入，输出仍加完整参考块，与论文的 dropout 语义不同；README 示例值 `fixed_std` 0.05、γ 0.9995、β 0.6、UTD 2 也只是该项目的取值。

**需用户决定：** C；proprio 定义；成功判定与奖励来源（人工按键还是位姿阈值）；接管规则（`b` 键、固定起点自动、还是 warmup 之后）；切换前的参考策略块是否入库；是否启用人工干预及用哪种遥操设备（主臂）；以及 0.3 表中的全部候选项。

## 1. 已完成的范围

[stage2.py](wall_rlt/stage2.py) 实现了单步 transition 校验，以及固定 RLinf Actor-Critic loss 的合成 CPU 检查。该检查使用随机特征，Actor 和 Critic 各更新一次，target network 保持不变；没有使用训练后的 WALL Encoder，没有真实 replay，也没有在线训练。

项目尚无正式 Stage 2 启动配置或 WALL 在线训练主循环；`configs/rlt/` 中现有训练 YAML 属于 Stage 1。固定 RLinf 源码包含 replay／warmup／权重调度能力，但这些能力尚未构成本项目已接通、已验证的 Stage 2 系统。

证据：[最新合成检查报告](logs/rlt-inference-20261029/cpu-check/stage2.json)。当前 `--synthetic` 入口只用于测试。

## 2. 测试设定与待定实验设计

| 项目 | 当前实现或测试值 | 用户需要决定的正式设计 |
|---|---|---|
| reward | 没有任务 reward 函数；测试中的 `[0, 1, -0.1]` 是人工构造的三条样例 | 成功／失败如何判定；奖励由人工、传感器或模型产生；是否稀疏奖励、是否有过程项；尺度、惩罚、发放时机和版本 |
| warmup 数据阶段 | 合成检查直接创建 batch，没有真实预填充 replay | 初始数据来源与行为策略；按 transition、episode 还是实际时间计数；开始学习的门槛 |
| warmup 优化阶段 | 没有运行阶段切换或初始多步预训练 | 是否先训练 Critic、是否让 Actor 先拟合参考动作；更新次数、BC／Q 权重切换方式，以及何时让 Actor 接管动作生成 |
| 学习率 warmup | Stage 2 未设计；Stage 1 的 `lr_warmup_steps=10` 不属于此项 | Actor／Critic 各自学习率和调度，是否需要学习率预热、持续多少更新步 |
| RL 目标与更新 | 测试复用 RLT AC：`gamma=0.99`、`bc_weight=1`、`q_weight=1`；两个 Adam 的 `lr=1e-3` | 是否沿用该目标；折扣的时间单位、BC／Q 权重及调度、Actor／Critic 更新频率、batch、target 更新系数与频率 |
| 探索 | 测试继承上游 `fixed_std=0.002`，作用于 tanh 前的高斯分布；参考动作 dropout 默认 0 | 探索方式与强度、参考动作 dropout、评估是否确定性；该标准差不是机器人关节角的弧度值 |
| transition | 当前只接受一条实际步长的记录；动作 `[B,1,A]`、reward／终止标记 `[B,1]`；默认 `fps=30`，`delta_t` 约 `1/30 s` | 一个经验条目覆盖何时到何时；执行动作来源、重规划时刻、时间戳；哪些情况终止、哪些只是截断；人工接管和复位如何记录 |
| RL action chunk | 当前强制 `C=1`，特征接口只截取 WALL 参考块第一步 | Actor 每次输出几步、实际执行几步、何时重新观察与决策；提前结束如何处理 |
| replay 与训练节奏 | 只有 batch 构造与 loss 更新检查，没有项目级长期 replay 调度 | 容量、采样、数据混合比例；每新增多少交互数据更新多少次，即 UTD（更新次数／新增数据量）的计数口径 |
| 动作接口 | 测试为 8 维 `[-1,1]` 动作；物理范围须由调用方显式提供 | 沿用何种物理动作表示；每维上下界、夹爪输出处理与执行反馈；不能把数据分位数直接当控制边界 |
| 评估与结束条件 | 没有真实任务成功率或在线收益验收 | 评估回合数、随机种子、对照策略、成功定义、交互预算和停止条件 |

上述数值的实际位置在 `synthetic_ac_check()`；它们目前不是一套可直接编辑后启动正式 Stage 2 的统一配置。其他单元测试也使用不同样例数值，例如手算折扣测试使用 `gamma=0.5`，同样不代表实验选择。

## 3. warmup 必须分别定义

“warmup”至少可能指四件事：先积累交互经验、先进行初始优化、调整初期到后期的 BC／Q 权重，以及逐渐升高学习率。它们的计数单位可能是环境 transition／episode、梯度更新步或学习率调度步，不能共用一个含义不明的 `warmup_steps`。

固定 RLinf 的源码已有 `warmup_min_size`、`warmup_post_collect_updates`、`train_every_transitions`／`train_every_episodes` 等调度项，以及 Actor 的 `actor_weight_schedule`。这些是可复用实现，不是本项目已选用的数值。尤其“预热”不自动等于“只训练 Critic”：需要单独决定哪些网络更新、哪些参数冻结，以及这一阶段谁生成实际动作。

上游的 Actor 接管路由区分仿真与真机分支，不能把仿真示例的门控逻辑直接写成我们的机器人已经具备的能力；示例里的经验数量、初始更新次数也不作为本项目默认值。

正式接入时应在独立配置中区分这些含义，记录每个计数的单位与切换条件。当前不替用户选择门槛或阶段长度。

## 4. transition、chunk、reward 与折扣需要一起设计

当前有两个不同长度：

- **H=32**：WALL 参考动作的预测长度，已经绑定现有 Stage 1 数据／模型契约。
- **C=1**：现有 Stage 2 验证中 Actor 动作和 transition 的长度，是测试范围，不是已确定的在线控制方案。

正式运行还需记录每次实际执行的步数 **L** 和执行时长。预测多少步、执行多少步、一个 replay 条目汇总多少步，可以分别设计，但必须在数据和更新公式中保持一致。

例如，若一个 transition 汇总实际执行的 L 步，并将 gamma 定义为每个基础环境步的折扣，标准终止处理下的目标可以写为：

```text
chunk_return = r_t + gamma*r_(t+1) + ... + gamma^(L-1)*r_(t+L-1)
target = chunk_return + (1 - terminated)*gamma^L*Q_target(next_obs, next_action)
```

这是与固定上游 loss 对应的一种设计形式，不是替用户定下 reward 或时间尺度。当前上游实现按 reward 张量宽度计算折扣长度；本项目因此只验证 C=1。若改为多步或可变步长，必须同步处理真实长度、奖励聚合、提前终止、下一观测与 replay mask，不能只把 `num_action_chunks` 从 1 改为另一个数，也不能把 padding 当作已执行步骤。

现有单步检查区分 `terminated`（任务结束）与 `truncated`（例如时间上限），标准路径仅由 termination 关闭 bootstrap。bootstrap 指使用下一状态的价值估计补足未来回报。自动复位时使用复位前的 `final_obs`，不能把下一回合的初始状态接到上一回合。真实系统中哪些事件属于哪类，需要用户设计，并与所选上游路径一起验证。

原示范的下一帧实测关节角是 Stage 1 监督标签，不能自动当作 RL replay 中“实际执行的动作”；当前原始示范也没有足够的 reward／终止字段。真实 transition 的来源必须单独明确。

## 5. 由用户掌控的推进顺序

先决定任务成功／失败与奖励来源、动作表示、决策节奏，以及 transition／chunk／终止语义。这些决定影响需要保存哪些经验和模型输入输出，应该在正式 Stage 2 接口与交互记录落地前确定。

在上述语义明确后，再确定 warmup 的各阶段、replay 与更新比例、学习率、折扣、BC／Q 权重、探索强度、target 更新和评估预算。参数可以按实验版本调整；每次运行应保存完整配置及相应 reward／transition 版本，避免把含义不同的经验或 checkpoint 混用。

后续先向用户提供具体的备选设计、影响和待定参数表；用户选定后再实现对应配置与训练流程。未选择的实验参数保持“待定”，不继承合成测试数值作为默认结论。当前官方 WALL 基础推理验证可以独立推进。

代码依据：[单步 transition 与合成检查](wall_rlt/stage2.py)、[冻结特征接口](wall_rlt/model/policy.py)、[物理／RL 动作转换](wall_rlt/data/transforms.py)、[固定 RLT AC 与调度源码](.cache/rlinf-rlt/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py)、[固定 Actor 实现](.cache/rlinf-rlt/rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py)。上游链接对应本地固定工作树，尚不表示这些模块均已接入项目在线运行。

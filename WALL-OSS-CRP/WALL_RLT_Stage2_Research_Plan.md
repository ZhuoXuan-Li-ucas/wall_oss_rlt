# WALL-RLT Stage 2 科研计划：复用 Evo-RLT、按对照组调参、以提高 bolt 成功率为目标

日期：2026-11-05。作者：Claude（按用户 11-05 的要求起草）。本文是**计划草案**：所有带"建议"字样的数值都是提议，需用户确认后才写进版本化配置；确认结果请记到 `WALL_RLT_Stage2_Design_Decisions.md` 第 0.3 节的"候选"列并改名为"定版"。

与本文配套的文档：`WALL_RLT_Stage2_Design_Decisions.md`（决策表与缺口清单）、`WALL_RLT_Inference_Deployment.md`（服务与回放）、`WALL_RLT_5070Ti_Official_WALL_Runbook.md`（镜像与 GPU 验证）、`yuhang_workspace/wall_oss_rlt/AGENT_HANDOFF.md`（本机交接）。论文：RL Token（arXiv 2604.23073v2，桌面 `rlt.pdf`）。

---

## 0. 一页总览

**科学问题。** 在 bolt-into-sleeve 的关键阶段（已夹住螺栓、对准并插入套筒），冻结 Stage 1 训好的 WALL-OSS-0.5 + RL token，只在线训练一个小 actor-critic，能否在 400 条左右真机 episode 内把自主成功率显著提高到基础策略之上；其次是否缩短完成时间。

**主指标。** 关键阶段设定下、固定起始状态协议、同一判定标准的**自主成功率**（无人工干预的 episode 中成功的比例），评估 50 条；对照组是同协议下的基础 WALL 策略。**次指标**：成功 episode 的中位完成时间、每 10 分钟完成次数、安全停机次数。

**方法。** 复用 Evo-RLT 的 actor-critic 核心、采集器、在线 trainer（它已解决块内提前终止、warmup 门槛、critic-only 窗口、安全限幅、零初始化 actor），替换掉它的 LeRobot/pi0.5/SO-101 外壳，接到我们的 WALL 推理服务 + AR5 执行器上。主实验配置忠于论文；Evo-RLT 自己加的东西（milestone 奖励、时间衰减、离线示范 buffer、RankQ）默认关闭，作为"救援"消融备用。

**阶段与预算（真机时间为纯交互估计，不含调试）。**

| 阶段 | 内容 | 真机时间 | 门槛（过了才进下一阶段） |
|---|---|---|---|
| P0 | 重启、镜像 GPU 验证、选 Stage 1 checkpoint、执行器真机验证 | 约 2 h | 执行器在真机上跑 `hold`→`replay`→`server` 无安全停机；服务时延、显存记录在案 |
| P1 | 真机 Stage 1 基线（对照组） | 50 条 ≈ 30 min | 成功率落在 15–80%；低于 10% 回到 Stage 1；高于 90% 改以速度为主指标 |
| P2 | 代码：WP1–WP7，全部在模拟臂与离线数据上验收 | 0 | 零初始化 actor 在模拟臂上逐帧复现纯 VLA 轨迹；离线 buffer 上 critic 能区分成功/失败 |
| P3 | 主实验（Arm 1）：warmup 20 条 + 在线 200 条 + 中途评估 | ≈ 2.5 h | 滚动 20 条自主成功率高于基线，或至少不降且没有安全事故 |
| P4 | 主实验续训到 400 条 + 终评 50 条 | ≈ 2.5 h | 终评成功率与基线的差有统计意义（见 §4.5） |
| P5 | 消融/救援（按 §4 的顺序，每个 200–400 条） | 每个 ≈ 2–4 h | 视 P4 结果决定做哪几个 |

按每天 3–4 h 真机时间算，P0–P4 约 3 个工作日，P5 每个消融 1 天。

---

## 1. 前置条件（P0、P1，硬性）

1. **重启并验证 GPU**：`nvidia-smi` 显示 595.99；运行手册第 6 节小算子检查在 v2 镜像上通过；另外在 `dexgraspvla` 环境里确认 `torch.cuda.is_available()` 且能在 5070 Ti（sm_120）上跑一个小 MLP 的前向反向。该环境已有 torch 2.11.0+cu128，RL 训练进程将运行在这里（见 §2.1）。
2. **选 Stage 1 checkpoint**（4 个候选：10k/20k/30k/40k，H100 dev 指标互有胜负）。两步走：
   - 离线：对 4 个 artifact 各起一次服务，在 bolt 的 dev episode 14 和 41 上跑 `check_wall_rlt_inference.py`，并用 `wall_offline_replay.py` 类似的方式比对预测位姿与记录位姿的误差（位置 mm、旋转度、夹爪错误率）。淘汰明显差的。
   - 真机：剩下的 2 个各跑 10 条关键阶段 episode（协议同 §3），看成功率与行为是否合理。选成功率高者；持平选 dev 位姿误差小者。
   - 记录：选定的 artifact id、服务时延中位数与 p95、峰值显存。时延直接决定 §3.4 的 C 与延迟策略。
3. **执行器真机验证**：`wall_execute.py` 目前只在模拟臂验证过。顺序 `--fake-policy hold` → `--fake-policy replay:<bolt episode>`（臂先摆到该条首帧）→ `--server`。确认 5 cm/20° 偏离保护、切段跳变告警、夹爪滞回都按预期。任何一次保护触发都要查原因，不调大阈值。
4. **真机 Stage 1 基线（对照组，P1）**：按 §3 的起始状态、复位、判定、时限协议，跑 50 条纯 WALL。这 50 条同时给出：基线成功率、完成时间分布、失败模式分类（没对准 / 对准了插不进 / 夹爪提前松 / 超时 / 安全停机）。失败模式决定 §4 里哪个救援消融优先。
5. **可观测性检查**：在基线中抽 10 条失败的，看两路相机画面里能否由人判断出"偏了多少、往哪偏"。如果人看画面也判断不了，RL 的上限就是基线附近，这时优先考虑的不是调参而是视角（侧视相机需重训 Stage 1，超出本计划）。

---

## 2. 代码方案：复用 Evo-RLT

### 2.1 总体架构

单进程，运行在 `dexgraspvla` 环境（Python 3.11，torch 2.11+cu128，AR5 驱动与 OmniPicker 都在这里，推理客户端只依赖标准库与 PIL）。进程内三部分：

- **执行回路**（50 Hz，沿用 `wall_execute.py` 的 `Runner`：观测、插值、逆解、保护），改造成可插入"动作改写器"的形式。
- **RL 动作改写器**（对应 Evo-RLT 的 `RLTActionModifier`）：VLA 阶段原样放行 WALL 的块；RL 阶段取 WALL 块的前 C 步为参考，调 actor 得到 C 步动作，换算成法兰目标交回执行回路。
- **在线 trainer**（对应 Evo-RLT 的 `OnlineRLTrainer` + `RLTOnlineCollector`）：按帧收集、按块成 transition、episode 结束后做梯度更新、存盘。更新在 episode 间隙进行（Evo-RLT 的做法），不与执行并发；按 UTD 5、每条 episode 约 40 个 transition 计，每条约 200 次小 MLP 更新，GPU 上不到 1 秒。

WALL 推理服务仍在 v2 镜像容器里（约 7.9 GiB），actor/critic 约 2M 参数，同一张 16 GB 卡放得下。服务是单线程 HTTP，一次只处理一个请求；RL 阶段每块一次请求，与现在一样。

### 2.2 复用清单

> **11-10 实际落地位置改为独立顶层包 `rlt_online/`**，不放进 `wall_rlt/`：`wall_rlt/stage2.py` 已存在，同名子包会遮蔽它；H100 训练包只收 `wall_rlt/`、`configs/rlt/`、`tests/rlt/`，在线代码只在 5070 Ti 上跑，不应进训练包。vendor 在 `rlt_online/_vendor/`，配置在 `configs/rlt_online/`，测试在 `tests/rlt_online/`。进度见第 9 节。

把 Evo-RLT 的以下文件原样复制到 `wall_rlt/_vendor/evo_rlt_core/`（保留 Apache-2.0 LICENSE，记录来源 zip、日期与每个文件的 sha256），不改内容：

| 文件 | 用途 | 备注 |
|---|---|---|
| `core/actor.py` | `ChunkActor`：MLP，固定 σ，参考 dropout，`residual_to_ref` 零初始化，`action_mask` | `residual_to_ref=True` 时 dropout 只作用在网络输入，输出仍加完整参考块（与论文语义不同，见 §4.3 的消融） |
| `core/critic.py` | `TwinCritic` | |
| `core/losses.py` | `critic_loss`（γ^actual_steps bootstrap、target Q 截断、可选 RankQ）、`actor_loss`（−Q + β·‖μ−ã‖²，按样本求和再取均值）、`q_action_sensitivity` | BC 项是对 C×A 维求和，β 的量纲随 C 变化，换 C 要重调 β |
| `core/replay_buffer.py` | `ReplayBuffer`：均匀/分层采样、按 episode 回滚、成功/失败计数 | |
| `core/interfaces.py` | `ChunkTransition` 数据结构 | |
| `core/utils.py` | `flatten_chunk`/`unflatten_chunk`/`soft_update`/`compute_discount_vector`/`build_mlp` | |
| `adapters/lerobot/online_collector.py` | `RLTOnlineCollector`：按帧累积、每 C 帧成块、成功/失败时提前关块并记 `actual_steps`、终端奖励写在最后有效步、人工帧占多数时把参考块换成执行块 | 只依赖 `annotations.SOURCE_HUMAN` 常量，复制常量即可 |
| `core/phase_controller.py` | `PhaseController` 手动/学习式交接 | 学习式分类器留到 §6 的后续项 |

以下文件**改写**到 `wall_rlt/stage2/`（保留逻辑，去掉 LeRobot 依赖）：

| 来源 | 目标 | 要改的点 |
|---|---|---|
| `adapters/lerobot/record/online_trainer.py` | `stage2/trainer.py` | 它本身不依赖 LeRobot，只要求一个 policy 对象提供 `.actor/.critic/.target_critic/._critic_step/.forward(batch)→(loss, info)/.train/.eval/.parameters/.save_pretrained`，并且 `forward` 从 trainer 热改的同一个 `policy_cfg` 对象读 `tau` 与 `actor_update_interval`。**首选：trainer 近乎原样保留，另写一个约 100 行的 `WallChunkACPolicy` 替身**（`ChunkActor` + `TwinCritic` + deepcopy 的 target + `_critic_step` buffer，`forward` 照抄 `modeling_rlt_ac.py:191-275` 的 critic/actor 交替逻辑）。保留：warmup 四门槛、critic-only 窗口、按新增 transition 数 × UTD 的更新预算与上限、离线/在线混采、分层采样开关、`latest_online_state.pt` 原子存盘与恢复、`step_NNNNNN` 快照、自主成功率滚动统计、wandb 可选。要改：存盘频率（它每条 episode 把整个 replay pickle 两次，400 条后约 250 MB，改成每 5 条一次或异步）；时限到而无标签时按 truncated 处理而不是像 Evo-RLT 那样抛错中止 |
| `adapters/lerobot/policies/action_modifier.py` | `stage2/modifier.py` | 去掉 `PrefixOutputCapture`/`RLTokenModule`（z_rl 由服务直接给）；`compute_chunk(vla_chunk, proprio, z_rl)`；保留 `action_clip_delta`、`clamp(-1,1)`、`get_last_chunk_tensors`、队列与 metadata。**要加**：训练 rollout 用 `actor.sample`（Evo-RLT 的 rollout 只用均值，σ 形同虚设，没有探索噪声），评估用均值 |
| `adapters/lerobot/record/prune_state_cli.py` | `stage2/prune.py` | 删除错误 episode |
| `cli/build_transition_cache_v2.py` | `stage2/offline_cache.py` | 数据源改为 bolt 的 npz+mp4，特征改为调推理服务 `return_features=True` 得到 z_rl 与 32 步参考；frame stride 2；关键阶段标签文件格式沿用 `critical_segments.json` |
| `diagnostics/critical_segment_labeler_cv.py` | `stage2/label_critical.py` | 读我们的 mp4 与 npz 而非 LeRobot parquet；按键不变 |

**不复用**：`record/loop.py`、`runner.py`、`backend.py`、`hil.py`（LeRobot 的 Robot/Teleoperator/录制）、`pi05_adapter.py`、`modeling_rlt_ac.py`（LeRobot 策略封装）、RTC 相关。`wall_rlt/stage2.py` 里基于 RLinf 的 C=1 合成检查保留作交叉验证（§2.4 WP1）。

### 2.3 必须新写的部分（与 Evo-RLT 的实际差异）

1. **VLA 接口。** Evo-RLT 在进程内跑 pi0.5 并截取 prefix token 再过 RL token encoder；我们由服务返回 `z_rl`（2048 维，已是 encoder 输出）、`actions`（32×10 物理相对量）、`absolute_targets`（32×7）。state_vec = concat(z_rl, proprio_norm)，维度 2058。
2. **动作空间与归一化（`stage2/action_space.py`）。** Evo-RLT 在 LeRobot 归一化后的关节空间里工作；我们是 30 Hz 的末端相对量（Δpos 3 + 相对 rot6D 6 + 夹爪 0/1）。actor 在 [−1,1] 里输出，所以要一个版本化的边界表：每维 `[lo, hi]`，物理 ↔ [−1,1] 线性映射。解码时 rot6D 六个数经 `rot6d_to_matrix` 正交化；夹爪连续值按 0.5 取阈成 0/1（执行器再做 0.4/0.6 滞回）；再用 `ee_absolute_window(relative, state_at_obs)` 得到 7 维法兰目标交给 `Chunk`。**验收**：actor 为恒等时，解码出的 absolute targets 与服务返回的 `absolute_targets` 逐元素一致（1e-6）。边界值见 §7。
3. **时间轴与延迟（`stage2/timing.py`）。** 论文：观测 → 出块 → 执行第 0..C−1 步。我们的执行器：请求时拍观测，推理约 227 ms（约 7 个 30 Hz 步），返回时跳过已过去的步、从第 d 步起执行并 0.1 s 渐变。RL 阶段二选一（§7 决策 D3）：
   - **固定延迟补偿（建议）**：块 k 在观测时刻 t_k 请求，执行的是该块的第 D..D+C−1 步，D 固定（建议 8 步 = 267 ms，大于时延 p95；推理更快就等到 D），下一次观测在 t_k + C（块内第 C 步时）。参考块取 WALL 预测的第 D..D+C−1 步。延迟成为环境的固定属性，MDP 平稳；代价是 actor 看到的状态比它控制的时段早 D 步。可在 proprio 里附加"已承诺的最后一个目标位姿"（10 维相对量）让 critic 知道在飞的指令。
   - **停等**：块执行完再观测、再推理，臂每块停约 0.23 s。与论文一致，但动作节奏与 Stage 1 数据不同，基线本身可能变差。
   无论哪种，VLA 阶段沿用执行器现有节奏（第 10 步重规划）。
4. **AR5 环境封装（`stage2/env_ar5.py`）。** 在 `Runner` 上加：episode 生命周期（起始位姿、复位脚本、时限）、按帧回调（把执行的动作、当前 state_vec、参考块、来源、是否关键阶段喂给 collector）、键盘事件。复用 `atom_robot.ar5.collect.RawKeyboard`（termios 裸 fd），不用 evdev。
5. **按键与奖励（`stage2/keys.py`）。** 沿用 Evo-RLT 在线按键：`r` 第一次进入关键阶段、第二次判成功；`u` 判失败；`s` 截断（不关 bootstrap）；`Space` 人工接管（第一期不接主臂，见 D8）；`Esc` 紧急停。成功/失败立即 `flush_episode`，之后的动作不入库。
6. **交接门控。** 固定起点协议下，episode 一开始就是关键阶段，起始动作由谁出按 trainer 状态决定：warmup 未满足 → 纯参考块；critic-only 窗口 → 仍纯参考块（actor 不更新也不执行）；之后 → actor（`residual_to_ref` 零初始化保证第一次执行等于参考块）。这一条 Evo-RLT 的真机路由里是由操作员按 `r` 决定的，我们把它和 trainer 状态绑定，避免 RLinf 那种"按键即执行随机 actor"的风险。
7. **评估模式。** `--eval` 下 actor 用均值（无噪声）、始终提供参考块、不入库、不更新；`--eval --vla-only` 下完全不调 actor。两种模式共用同一套起始状态与判定流程。
8. **日志。** 每条 episode：`chunks.jsonl`（含 d、D、时延）、每帧 50 Hz 的 ticks、每个 transition 的 `state_vec` 哈希与原始图像（两路 mp4，30 fps，供复算与审计）、按键时间戳、结局、操作员备注。

### 2.3b Evo-RLT 实时回路核对结果（11-05 只读分析）与适配层的硬性要求

对 `record/loop.py`、`backend.py`、`runner.py`、`online_cli.py`、`modeling_rlt_ac.py` 的逐行核对发现以下几点，适配层必须按此处理：

1. **单位不一致（Evo-RLT 的 bug）。** 它的 collector 存的 `exec_chunk` 是反归一化后的机器人单位，而 `ref_chunk`、`state_vec` 和离线 cache 都是归一化量，critic 实际在 Q(s, 度) 和归一化 actor 输出之间训练。我们的 collector 输入**必须**是归一化后的 C 步指令块（actor 输出经 clamp/clip 后的值，或 VLA 阶段的参考块），不是执行器 50 Hz 插值后的量。
2. **块边界错位。** 它的 collector 从 `start_episode` 起每 C 帧切一块，而 actor 的块从按 `r` 或接管释放那一刻开始，两者不对齐，`exec_chunk` 会横跨两个 actor 块。我们改为**按块喂**：每算出一个块就调用一次 collector 的块起点（或直接写一个块级 `add`），`state_vec`/`ref_chunk` 用该块自己的；固定起点协议下 episode 第 0 帧就是第一个块，天然对齐；提前终止靠 `flush_episode` 的 `actual_steps`。
3. **没有探索噪声、没有参考 dropout。** 在线 CLI 实际跑的是 `fixed_std=0`、`ref_dropout=0`，rollout 取均值。主配置按论文开噪声与 dropout（§4.3），属于对 Evo-RLT 在线设定的有意偏离。
4. **零初始化 actor 在 RL 阶段不等于 VLA。** 它用分位数 q01/q99 归一化，VLA 参考超出分位数的部分在 RL 阶段被 clamp 到 [−1,1]，第一块就与 VLA 不同。我们的边界表用 train 集 min/max 外扩（D2），保证参考块永不触边，零初始化 actor 逐元素等于 VLA。
5. **推理同步阻塞。** 在线模式 RTC 关闭，推理在队列空的那一帧同步执行，该帧超时且不追帧，相当于每块末尾停一下。这印证 D3 要明确选延迟策略，不能照搬。
6. **人工接管期间状态陈旧。** 接管时不推理，collector 的 peek 拿到的是接管前的 `state_vec`，连续人工块 `state == next_state`。第一期不接管，第二期接主臂时要在接管期间继续算 z_rl。
7. **每条 episode 只收一次关键阶段尝试**；时限到或无标签直接抛错中止会话。我们的时限按 truncated 处理（§3.3），不中止。
8. **Evo-RLT 在线实际默认值**（供参考，非本项目默认）：actor/critic 512×3 残差、critic LayerNorm、`lr_actor` 3e-5、`lr_critic` 1e-4、β 0.3、clip_delta 0.1、RankQ α=1/1、分层采样开、单臂 mask、`actor_rl_arm=left`（我们单臂要 `both`）。
9. 训练标签只经 replay buffer 到达 trainer（`done=1` 的 transition 计成功/失败），`is_critical` 字段训练时不用；`ChunkTransition.source` 的编码与 `interfaces.py` 的常量不一致，但训练不用它。

### 2.4 工作包与验收（P2，全部不碰真机）

| WP | 内容 | 验收 |
|---|---|---|
| WP1 vendor + 单测 | 复制 Evo-RLT 核心，跑它的 `test_losses`、`test_replay`、`test_actor_critic`、`test_online_collector`、`test_online_trainer_offline_buffer`（都不需要 LeRobot）；再用 `wall_rlt/stage2.py` 的 RLinf 合成检查做交叉验证：同一批 C=1 数据，两套 critic 目标数值一致 | 全绿；交叉验证误差 < 1e-5 |
| WP2 动作空间 | 边界表、归一化往返、rot6D 正交化、夹爪阈值、`ee_absolute_window` 对接 | 在 50 条 bolt 数据的全部帧上往返误差 < 1e-6；恒等 actor 的 absolute targets 与服务一致 |
| WP3 环境封装 | 模拟臂 + `--fake-policy replay` 上跑 RL 阶段，actor 零初始化、σ=0 | 逐帧目标与纯 VLA 模式一致；collector 产生的 transition 数 = 帧数/C（向上取整），块边界与 actor 块对齐，`exec_chunk` 是归一化量且等于 `ref_chunk`，`actual_steps` 正确 |
| WP4 离线 cache | 标注 50 条 bolt 的关键阶段起止（§3.1 的定义），服务开 `return_features` 逐帧算 z_rl 与参考块（关键阶段约 170 帧/条，stride 2，约 4k 次请求，约 20 min），写成 `ChunkTransition` 列表 | 用它单独训 critic 2k 步：成功段终端前的 Q 高于失败段；`q_action_sensitivity` 非零 |
| WP5 trainer | 改写 trainer、存盘恢复、prune | 用 WP4 的 cache 当"在线数据"跑 20 个假 episode，恢复后 loss 曲线连续 |
| WP6 评估 | `--eval` / `--eval --vla-only`，结果表 | 模拟臂上可跑 |
| WP7 端到端干跑 | 模拟臂，合成奖励（例如法兰到达预设套筒位姿 2 mm 内判成功）跑 100 条 | 自主成功率上升（证明信号链通），无崩溃，GPU 显存稳定 |

WP7 的合成奖励只用于验链路，不代表真机能学会。

---

## 3. 真机实验协议（基线与所有实验组共用）

### 3.1 关键阶段定义

> **11-05 用户已改定起点与判定（见 §7 D5、D7）**：从拿螺栓开始，成功＝把螺栓放进套筒。下面 §3.1–3.4 按"已夹住螺栓、悬在套筒上方"起点写的内容不再适用于基线，Stage 2 部分待按新起点重写。

起点：螺栓已被夹住，法兰在套筒正上方 3–5 cm（高度在协议里固定，建议 4 cm），水平位置在 ±1 cm 内随机（用预先生成的随机表，按 episode 序号取，基线和实验组用同一张表），夹爪姿态按采集时的斜向下姿势。终点：螺栓插入套筒到底（操作员判定），或失败（掉落、卡住 20 s、安全停机）。

### 3.2 起始与复位（建议方案 A）

- 起始位姿由脚本生成：从一个标定过的"套筒上方基准位姿"加随机偏移，用执行器的 `goto` 路径走过去（不经过 WALL）。
- 成功后不松手：脚本抬起 5 cm 回到基准高度，直接开始下一条。约 5–8 s。
- 失败且螺栓仍在手：同上。螺栓掉落：操作员放回支架，用主臂或脚本重新抓取（约 30–60 s），这类 episode 在日志里标 `reset=manual`。
- 每 25 条检查一次套筒与底座位置没有被撞偏（基准位姿复核）。

### 3.3 判定与按键

| 事件 | 按键 | 记录 | 对 critic 的影响 |
|---|---|---|---|
| 成功 | `r`（第二次） | terminated, reward 1 在最后有效步 | 块提前关闭，bootstrap 置零 |
| 失败 | `u` | terminated, reward 0 | 同上 |
| 操作员中断 / 时限 20 s | `s` / 自动 | truncated | bootstrap 保留 |
| 安全停机 | 自动 | truncated + `safety_stop=1` | 同上；另计次 |

成功的操作定义要写成一句话贴在操作台上，基线与所有实验组不变。建议同时记录判定瞬间的法兰 z 与夹爪开度，事后可核查判定一致性。

### 3.4 时间参数

30 Hz 模型步；C 与延迟策略见 §7；时限 20 s = 600 步 = 60–75 个块；每条 episode 关键阶段约 6–12 s。

### 3.5 评估

- **基线**：P1 的 50 条；主实验开始当天再跑 20 条作漂移对照（同一天设备状态）。
- **训练中评估**：每 50 条训练 episode 后插入 20 条 `--eval`（均值 actor），不入库。这是调参的主要依据。
- **终评**：训练结束后 50 条 `--eval`，与同一天 20 条 `--eval --vla-only` 交替进行（10 条一组交替），消除时间漂移。
- **统计**：成功率差用两比例的 Wilson 区间或 Fisher 精确检验；50 对 50 时能稳定检出约 25 个百分点以上的差异，20 对 20 只能看趋势。完成时间用成功 episode 的中位数与 Mann-Whitney。

---

## 4. 对照组与调参路线

### 4.1 原则

- 真机每条 episode 约 20 s 全周期，400 条约 2.5 h，一天最多一个实验组。**一次只动一个因素**，每个因素的取舍以"训练中评估的成功率"和 §5 的诊断指标为依据。
- 先做"忠于论文"的主配置，再做 Evo-RLT 的附加项；附加项只在主配置学不动时作为救援启用。
- 不复制论文已经证明"去掉会大幅变差"的消融（β=0、无 RL token、C=1），那些在我们这儿没有信息量。

### 4.2 Arm 0：对照组（纯 WALL）

P1 的 50 条 + 每个实验日 20 条。

### 4.3 Arm 1：主配置（忠于论文，建议值待确认）

| 项 | 建议 | 依据 |
|---|---|---|
| C | 10（延迟补偿方案下；停等方案下 8） | 论文 10；补偿方案要给 D=8 留余量 |
| 参考块 | WALL 预测第 D..D+C−1 步（补偿）或第 0..C−1 步（停等） | 必须是实际会执行的那段 |
| proprio | 10 维物理 state 按契约统计归一化 | 论文"末端位姿" |
| actor/critic | 2 层 256，relu；critic 带 LayerNorm（Evo-RLT 默认，RLPD 做法） | 论文 2×256；LayerNorm 防 Q 爆 |
| actor 形式 | `residual_to_ref=True`，零初始化 | 真机安全：初始等于 VLA |
| 参考 dropout | 0.5 | 论文 |
| σ | 先按 §7 D6 标定到每步约 1 mm / 0.5°，归一化后约 0.02–0.05 | 论文"小"；Evo-RLT 0.05 |
| `action_clip_delta` | 归一化空间 0.1（约对应物理边界的 10%） | Evo-RLT README 示例；安全 |
| γ | 0.99（按 30 Hz 步） | 决策表候选 |
| β | 1.0 | Evo-RLT `ac_paper_screw.yaml`；论文未给 |
| critic:actor | 2:1 | 论文 |
| UTD | 5，每条 episode 上限 1000 次 | 论文 |
| batch | 256；lr actor 3e-5、critic 1e-4；τ 0.005；grad clip 1.0 | Evo-RLT 在线实际值：actor 直接驱动机器人，学习率压低；核心默认与 TD3 常用的是 3e-4/3e-4，可作 A4 之后的备选 |
| target Q 截断 | 100（Evo-RLT 默认，实际上限 1/(1−γ^C)≈10，不会触发） | 保险 |
| 奖励 | 终端 1/0；milestone 0；time_decay 1.0 | 论文 |
| 离线 buffer | 关 | 论文 |
| RankQ | 关 | 论文 |
| 采样 | 分层采样开（Evo-RLT 默认）；这是对论文"均匀采样"的偏离，但在稀疏奖励下能让成功样本不被淹没，若要严格忠于论文则关 | 见 §4.5 |
| warmup | 20 条纯 VLA、≥3 成功、≥3 失败、≥500 transition；之后 critic-only 10 条 | Evo-RLT 门槛思路；论文只说 N_warm |
| 人工干预 | 不启用 | 第一期简化 |
| 预算 | 400 条 + 评估 | 论文 400–1000 |

### 4.4 决策树（P3/P4 后）

```
终评 Arm1 vs Arm0
├─ 显著更好（≥ +20 pp 且区间不含 0）
│    → 做理解型消融（可选，按价值排序）：
│       A2 离线示范 buffer 混采 0.5（期望：学得更快）
│       A3 C = 8 vs 10
│       A4 β = 0.3 / 3（探索幅度）
│       A5 不用分层采样（论文原味）
├─ 持平（区间含 0，无安全事故）
│    → 先看 §5 诊断：
│       Q 对动作不敏感 → 救援 R1（RankQ）或 R2（离线 buffer）
│       actor 几乎不偏离参考 → 降 β 到 0.3、提 σ、确认 dropout 生效
│       时延/跳变告警多 → 改停等方案（D3 另一分支）
│       失败模式是"看不见" → 停，讨论视角
│    → 再续训 200 条
└─ 更差或有安全事故
     → 停训，查 §5 的异常项；检查边界表、σ、clip_delta；
       回到 WP7 干跑复现；不在真机上盲调
```

### 4.5 需要留意的"忠于论文"偏离项

分层采样、critic LayerNorm、target Q 截断、`residual_to_ref` 是 Evo-RLT 的工程选择，论文没有。主配置保留它们是为了稳定与安全；如果最终要写成"复现 RLT"的结论，至少 A5 要做一次。

---

## 5. 监控指标与调参手册

每条 episode 结束后记录（Evo-RLT 的 wandb 键名可直接沿用）：

| 指标 | 健康 | 异常与对策 |
|---|---|---|
| `autonomous_success_rate_rolling_20` | 从基线水平逐步上升 | 100 条后仍不升：看下面各项 |
| `loss_critic` | 早期下降后平稳 | 持续增大或 NaN：lr_critic 减半、确认 LayerNorm、查 reward 尺度 |
| Q 值均值 / 最大 | 在 [0, 1/(1−γ^C)] 内，成功 episode 末块接近 1 | 超界：target 截断、τ 减小；全在 0 附近：reward 没写进去或成功太少 |
| `q_action_sensitivity` | 明显非零且随训练上升 | 接近 0：critic 退化成 V(s)，启用 RankQ 或增加成功/失败样本 |
| `loss_actor` 的 BC 分量 | 随训练缓慢上升（actor 在偏离参考） | 恒为 0：dropout 未生效或 β 太大；暴涨：β 太小或 Q 错误 |
| 每块 `mean_abs_delta`（actor 输出与参考的偏差，物理单位 mm/度） | 几 mm 以内、逐渐有方向性 | 顶到 clip_delta：Q 可能在利用误差，先降 lr_actor |
| 切段跳变告警、安全停机次数 | 0 | > 0：先查时延与边界，不继续训练 |
| 推理时延 p95 | < D 步对应时间 | 超过：D 要调大或改停等 |
| 成功 episode 中位时长 | 下降 | 上升：actor 学到"试探"，检查 γ 与时限 |
| 干预率 | 第一期恒 0 | |

调参顺序（一次一个）：β → σ/clip_delta → 采样/离线 buffer → C。lr、τ、batch 不动。

---

## 6. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 关键阶段可观测性差（俯视只见约 20 px 套筒边、腕部只见螺栓头、无力传感） | RL 上限受限 | P1 的失败分析先判断；若人也看不出偏差，优先讨论视角而非调参 |
| 执行器未在真机验证 | 安全 | P0 第 3 条逐级验证；clip_delta、偏离保护都不放宽 |
| 推理时延与块长不匹配 | MDP 不平稳、跳变 | 固定延迟 D；记录每块实际 d；超限告警 |
| 复位劳动量 | 每天有效 episode 数 | 方案 A 不松手；掉落才人工 |
| 成功判定不一致 | 指标失真 | 一句话定义 + 记录法兰 z；同一操作员；交替评估 |
| Stage 1 checkpoint 选错 | 基线偏低 | P0 第 2 条的两步筛选 |
| GPU 共用（服务 + 训练） | 显存/时延抖动 | 更新只在 episode 间隙做；监控 p95 时延 |
| 时间漂移（光照、磨损） | 对照失真 | 每个实验日 20 条 vla-only；终评交替 |
| Evo-RLT 与论文偏离 | 结论归因 | §4.5 的 A5 |

---

## 7. 需要用户确认的决策（附建议）

| 编号 | 决策 | 建议 | 备注 |
|---|---|---|---|
| D1 | C | 10（补偿）/ 8（停等） | **11-10 用户定：C=12（30 Hz，每 0.4 s 决策一次），固定执行起始步 D=11**，即 actor 输出的块是观测时刻 WALL 窗口的第 11–22 步；第 k+1 次观测在第 k 次之后 C 步；中间 D 步是上一块的尾巴，作为 RTC 前缀发给 WALL；回答早到就等到第 D 步，晚到则从实际到达的步开始并标记该 transition。依据：11-10 真机 404 次时延中位 0.289 s、最大 0.311 s，带 z_rl 再加约 20 ms；D=11 留约 37 ms、C=12 留约 70 ms 余量；纯 WALL 现有节奏是每 11 步一次、零停顿。BC 项在 C×10=120 维上求和，β 要按 C 重估 |
| D2 | 动作边界表 | train 集每维 min/max 外扩 20%，再与工作空间盒、5 cm/20° 保护取交；rot6D 各维限在 [−1.2, 1.2]；夹爪 [0,1] | 需你对照实际工作空间签字 |
| D3 | 延迟策略 | 固定延迟补偿 D=8 | 停等作为分支。**11-05 用户：不用停等**。基础策略执行器（`wall_execute.py`）改成 LeRobot 异步推理的做法：剩 16 步就要下一段、只丢推理期间实际走过的步（第一段不丢）、重叠段渐变；传图默认 jpeg（真机延迟 0.45 → 约 0.27 s）。同日用户要求上 RTC：服务端 `wall_rlt/inference/runtime.py` 经 wall-x `_execute_flow_action` 钩子接入（不改 MODEL_SOURCES），请求可带旧段剩余目标；默认 inpaint（每个去噪步把前缀拉向加噪旧段，推理 0.24 s 不变），LeRobot 梯度引导为 `guidance` 选项（0.47 s：每步 32 token 前向被 wall-x MoE 的 Python 逐专家循环拖成 16 ms，反向同样慢）。离线前缀偏差 不开 11.2 / guidance 6.4 / inpaint 0.1 mm。镜像 `wall-rlt-inference:5070ti-20261105T082442Z`（v3） |
| D4 | proprio | 10 维归一化 state；是否附加"已承诺目标"10 维 | 建议先不加 |
| D5 | 起始状态与复位 | ~~§3.1/3.2 的方案 A，高度 4 cm，±1 cm~~ | **11-05 用户定：从拿螺栓开始**，即采集同款起点：臂在 collect 的 `centre_joints`（50 条 bolt 首帧与它的关节差均为 0），夹爪张开，螺栓插在左侧蓝色座里，套筒固定。§3.1–3.4 按关键阶段起点写的部分随之作废。**11-10 用户定交接方式：WALL 从拿螺栓开始执行，搬到套筒附近再交给 actor**，即论文第 V 节"基础模型做容易的部分、关键阶段交给 RL"与附录 B 全任务阶段的做法；RL 只对交接之后的片段入库、训练。交接的触发方式（论文：训练时由操作员决定，测试时用学到的交接预测）与具体位置尚未定。11-10 真机数据里 WALL 在 10–14 s 进入示范松手点 2 cm 以内，然后悬停 |
| D6 | σ | 干跑一次：把 σ=0.02/0.05 换算成 mm/度后定 | **11-10 用户定（位置与姿态）**：Evo-RLT `ChunkActor` 自带噪声设为 0；探索改由动作空间层按物理单位添加：每块抽一次偏移，位置三轴各一个、姿态一个小旋转，从块的第 1 步按 1/C 线性增长到第 C 步满额；大小为每块位置每轴标准差 2 mm、姿态每轴 0.5°；评估不加噪声。这是对论文逐元素独立高斯的有意偏离：我们每步的动作是相对观测位姿的目标，独立噪声只制造相邻步抖动，几乎没有净位移；损失里不出现噪声分布，off-policy 下换形状不改变算法。依据：示范在套筒 3 cm 内每步移动 p90 1.4 mm，WALL 预测相邻步中位 0.93 mm，14 次决策累计每轴约 7 mm，覆盖 11-10 悬停时离示范松手点 3–7 mm 的差距。夹爪的探索见 D14 |
| D7 | 奖励与判定 | 终端 1/0；判定一句话由你写 | **11-05 用户定：成功＝把螺栓放进套筒**（按采集时的做法理解为松手后螺栓留在套筒里）。基线时限暂用 60 s（50 条 bolt 示范 12–35 s） |
| D8 | 人工干预 | 第一期不接主臂 | 第二期可接 `LeaderBus` |
| D9 | warmup 与门控 | 20 条 / ≥3 成功 / ≥3 失败 / ≥500 transition；critic-only 10 条 | **11-10 用户定：warmup 期间开启松手探索**（人工按键 + 每块 0.1 随机松手），这是对论文算法 1 "warmup 只执行 VLA 参考"的有意偏离，否则 WALL 悬停不松手，warmup 拿不到成败对比；位置与姿态在 warmup 不加噪声（按提议记录）。**用户随后定门槛**：交接后的 RL 片段每条只有约 5–14 个 transition，原来 500 的门槛与 256 的 batch 要 18–100 条才达到，改为 warmup 20 条、至少 3 成 3 败、至少 150 个 transition，之后 10 条只更新 critic，batch 128；actor 约从第 30 条开始学习。隔 2 步子采样（论文附录 B）暂不做 |
| D10 | 评估条数 | 训练中每 50 条评 20；终评 50 + 20 vla-only | |
| D11 | 分层采样、LayerNorm、residual actor | 主配置保留 | 与论文的偏离，见 §4.5 |
| D12 | Stage 1 checkpoint | 按 P0 第 2 条筛选 | |
| D13 | 预算 | 400 条，分 2 天 | **11-10 用户定：首轮 200 条训练 episode**，在第 50、100、150、200 条各评估一次（评估不加噪声、不按键、不入库）。理由：约 69% 的 episode 带一次直接的松手实验，每条信息量高于论文的只有末尾成败。是否续训、每次评估多少条（D10 原提议 20 条）待定 |
| D15 | 交接触发 | 自动按位置 | **11-10 用户定**：每次向 WALL 请求（拍照）时检查，法兰离示范松手点中位（0.1973, −0.3087, 0.3791 m，50 条，标准差 7–9 mm）不到 3 cm，且实测开度在 0.12–0.30（夹着螺栓时 0.17–0.22，张开约 0.85）时交接，之后不切回；不设手动交接键。RL 阶段最长 **20 s**（提议 15 s，用户改为 20 s；合 50 块），未松手记为截断；WALL 阶段 30 s 内未交接记为 WALL 阶段失败，不入库、不计入 200 条预算、计入端到端成功率。两组评估用同一规则 |
| D16 | 按键 | 沿用采集习惯 | **11-10 用户定**：回车开始下一条；`o` 人工松手（人工干预）；`s` 成功；`f` 失败；`r` 丢弃本条；Ctrl-C 紧急停止；空格不触发任何动作（采集里空格是夹紧）。松手发生时 RL 在该步停止，臂等夹爪张开后上抬 5 cm，按 `s`/`f` 后才入库并更新 |
| D14 | 夹爪（何时松手）由谁决定 | 第一轮交给 WALL | **11-10 用户定：交给 actor**，理由是 11-10 真机上已经对准却迟迟不松手。随之必须改三处，否则 actor 仍松不了手：①夹爪维不受 `actor_action_clip_delta` 限幅（0.1 归一化只允许夹爪偏离 WALL 0.05）；②BC 项对夹爪单独加权（归一化后从闭合到张开每步差 2，12 步 BC 代价约 48β，而 Q 最多约 1）；③夹爪的探索不能靠高斯噪声，需要离散的"试着松手"。**11-10 用户定松手来源：人工按键为主、随机尝试为辅**：操作员判断已对准而 actor 未松手时按键，该块按人工干预入库（夹爪指令改为张开，参考动作换成人的动作，同论文第 V 节）；另以每块概率 p 随机试松手，给 critic 提供"这里松手会失败"的反例。**用户定 p=0.08**（先提议 0.1，用户要求稍小；约 69% 的 episode 含一次随机松手，中位在交接后约 3.6 s）。夹爪不受限幅（`actor_clip_exempt_dims: [9]`）。**11-10 用户定夹爪 BC 权重 0.002**（其余 9 维为 1）：整块违背 WALL 去松手的代价约 12×4×β×0.002≈0.1，约为成功价值的十分之一 |

---

## 8. 后续（本计划之外）

- 学习式交接（Evo-RLT `HandoverClassifier` 在 z_rl 上的二分类），用于全任务自动切换。
- 全任务两阶段训练（论文附录 B）：WALL 从 0 位抓取搬运，到位后切换。
- 侧视相机 + Stage 1 重训（若可观测性是瓶颈）。
- 主臂人工干预（`LeaderBus` 接入执行回路）。

---

## 9. 实施进度

### 9.1 2026-11-10：WP1、WP2 完成，WP5 的核心完成

用户 11-10 告知：顶部相机已复位到与 `wall_unlocked_v2` 画面一致；纯 WALL 基线成功率（P1）**推迟到后面**再测。因此先做不需要机械臂、也不依赖待定参数的代码。

| 内容 | 位置 | 验证 |
|---|---|---|
| Evo-RLT 原样引入：`evo_rlt/__init__`、整个 `core/`、在线 collector、在线 trainer、prune 工具、import 用的 adapters 骨架，共 35 个文件（含 5 个 yaml）；另有 16 个不依赖 LeRobot 的上游测试模块及其辅助文件 | `rlt_online/_vendor/`（`NOTICE.md`、`MANIFEST.json`） | 每个文件的 CRC32 与大小都和 zip 中央目录一致；`rlt_online.evo.verify_vendor()` 复核 sha256。上游测试 144 个中 143 通过，唯一失败的那个需要未引入的 LeRobot `ChunkACPolicy`，等价性质在本项目测试里补上 |
| 动作空间：按窗口步 k（0–31）和维度分别取训练集 min/max 再外扩 20%，旋转 6D 限在 ±1.2，夹爪固定 [0,1]；[−1,1] 编解码；换算成 7 维法兰绝对目标；本体归一化（训练集 state min/max） | `rlt_online/action_space.py`；边界提案 `configs/rlt_online/action_bounds.bolt.20261110.proposal.json`（状态：D2 未确认） | 窗口行数 817,824，与契约归一化计数一致；训练集往返无截断、误差 1e-12；还原出的绝对位姿与录制位姿一致；dev/test 越界比例 < 0.1%。**对运行中的 bolt 20k 服务**：三帧真实观测，服务的 32×10 相对动作经编解码与换算后，和服务自己的 `absolute_targets` 一致（位置误差 0，旋转 < 2e-6°）；z_rl 为 2048 维 |
| 策略替身 `WallChunkACPolicy` + 配置 `ChunkACConfig`／`OnlineConfig`：上游 trainer 原样可用；所有影响实验的字段都必填；rollout 支持带噪采样（上游在线只用均值）与按维度掩码 | `rlt_online/policy.py` | warmup 门槛、critic-only 窗口内 actor 不变、窗口后 actor 更新、tau 与更新间隔热改后复原、`latest_online_state` 断点续训、存取 checkpoint；零初始化 actor 逐元素等于参考块；actor 损失不给 critic 加梯度 |
| 块内提前终止 | `tests/rlt_online/test_vendor_and_target.py` | 成功发生在块内第 3 步时 collector 关出 `actual_steps=3`、奖励在第 3 步、执行块后部补零；TD 目标手算核对：满块 γ^C·Q'，终止短块无 bootstrap，截断短块 γ^3·Q' |
| Arm 1 配置提案 | `configs/rlt_online/arm1.proposal.yaml`（11-10 改名为 `arm1.bolt.yaml`） | C（D1）、σ（D6）、save_dir 留空，加载器拒绝运行；填上后可加载 |
| 机器人环境冒烟 | dexgraspvla（Python 3.11、torch 2.11+cu128、5070 Ti） | 导入与 vendor 校验正常；C=10 时 actor+critic 共 1.88 M 参数；一次 `act()` 0.08 ms；每条 episode 后 200 次更新（batch 256）0.65 s；显存峰值 66 MiB；存续训状态 0.06 s |

测试：`tests/rlt_online` 23 个，与原有 `tests/rlt` 一起 440 个全部通过（需用 `scripts/check_wall_rlt.py` 同款 PYTHONPATH，并去掉 ROS 的 PYTHONPATH）。

### 9.2 实现中得到的、影响决策的事实

1. **σ 与 clip_delta 的物理含义由边界表决定，且随窗口步 k 变大。** 当前提案表覆盖整条任务（含大幅搬运），归一化 1.0 对应的位置幅度：

   | 窗口步 k | 1.0 单位 ≈（x, y, z） | σ=0.02 | σ=0.05 | clip 0.1 |
   |---|---|---|---|---|
   | 0 | 8, 13, 13 mm | 0.2–0.3 mm | 0.4–0.7 mm | 0.8–1.3 mm |
   | 9 | 70, 107, 112 mm | 1.4–2.2 mm | 3.5–5.6 mm | 7–11 mm |
   | 15 | 96, 142, 158 mm | 1.9–3.2 mm | 4.8–7.9 mm | 10–16 mm |

   RTC 异步执行器下，块从观测后约第 8 步开始执行，所以实际落在 k≈8–17 那几行。上游 actor 的噪声是逐元素独立的，相邻 33 ms 两步的目标会互相抖动几毫米，可能触发关节限速。D6 要同时决定 σ 的大小，以及是否改成块内时间相关的噪声。另一个选项是只为关键阶段另算一张更窄的边界表（D2）。
2. **第一次梯度更新要等 replay 至少有一个 batch。** batch 256、每条约 40 个 transition 时，至少要 7 条 episode；这和 warmup 门槛（D9）一起决定什么时候开始学习。
3. WALL 的夹爪预测会略超出 [0,1]（−0.01 到 1.01），旋转 6D 对角元会略大于 1（约 1e-4）。前者截到 [0,1] 不影响执行；后者是边界改为 ±1.2 的原因。

### 9.2b 2026-11-10 晚：已定决策的代码落地

| 决策 | 实现 | 验证 |
|---|---|---|
| D6 位置与姿态探索：每块线性增长偏移，2 mm／0.5° | `rlt_online/exploration.py` 的 `ChunkExplorer`：在解码后的物理相对量上加，位置在基座系平移，姿态以基座系小旋转左乘；eval 不加 | 偏移逐步等于 (i+1)/C 倍；20000 次抽样标准差在 3% 内；加噪后在 D1 的第 11–22 行边界内无截断 |
| D14 随机松手 p=0.08、人工按键为主 | 同上：触发时整块夹爪指令为张开；人工按键由执行回路按人工干预处理（待 WP3） | 触发频率 0.08±0.006；整块张开 |
| D9 warmup 只开松手探索 | `mode='warmup'` 时不加位姿噪声、照常随机松手 | 测试覆盖 |
| D14 夹爪不受限幅 | `ChunkACConfig.actor_clip_exempt_dims`，配置为 `[9]` | 位置被截在 0.1，夹爪不截 |
| D14 夹爪单独 BC 权重 | `actor_bc_dim_weights`（10 个，按 C 重复）与 `weighted_actor_loss`；全 1 时与上游 `actor_loss` 逐位相同 | 权重 0 时夹爪偏离不产生 BC 代价。**取值未定**，配置里为 null，加载器拒绝运行 |
| D9 门槛 | `arm1.bolt.yaml`：20 条、3 成 3 败、150 个 transition、critic-only 10 条、batch 128 | 配置加载测试 |

测试：`tests/rlt_online` 33 个，加原有 `tests/rlt` 共 450 个全部通过；dexgraspvla 环境 GPU 导入与零初始化检查通过。

随后 D15／D16 落地：

| 内容 | 实现 | 验证 |
|---|---|---|
| episode 状态机：WALL → RL → 已松手 → 结束；自动交接、20 s／30 s 时限、按键含义、哪些入库与计入预算 | `rlt_online/episode.py` 的 `EpisodeProtocol`，参数从 yaml 的 `handover`／`limits`／`keys` 读 | 远处或空夹不交接、只交接一次；30 s 未交接不入库不计预算；交接后 20 s 截断；`o` 只在 RL 阶段生效一次；`s` 必须在松手后；`f` 在 RL 或松手后；`r` 随时丢弃；空格无效 |
| 按块喂数据、截断保留 bootstrap | `rlt_online/collector.py` 的 `ChunkCollector`（继承上游采集器，vendor 不改）与 `attach_chunk_collector(trainer)` | 块边界与 actor 块一致；松手所在的短块 `actual_steps` 正确、奖励在最后一步；截断时全部 `done=0`、最后一个 transition 接到时限时刻的观测、不算成功也不算失败；人工块的 BC 目标换成人的动作 |

测试共 460 个全部通过。

### 9.2c 2026-11-10 晚：Stage 2 暂停，先扩数据重训 WALL

用户决定暂停 Stage 2：先重新采集约 100 条 bolt，重训 Stage 1，看纯 WALL 能做到什么程度。因为相机出过偏移又复位，**旧的 50 条 bolt 数据倾向弃用**（不删除，只是不再使用）。新数据单独放在 `~/atom_episodes/wall_unlocked_v3`，采集参数与 v2 相同（dry-run 已比对通过）。

依赖旧 bolt 数据、新数据到位后必须重做的东西：

| 项 | 位置 | 处理 |
|---|---|---|
| 契约、归一化、划分 | `configs/rlt/`、`metadata/` | 按新根目录重新生成 |
| Stage 1 模型 | `artifacts/wall_v2_bolt_step*`、运行中的 bolt v3 服务 | 用新数据重训后替换 |
| 动作边界提案（D2） | `configs/rlt_online/action_bounds.bolt.20261110.proposal.json` | 用新训练集重算 |
| 交接参考点（D15） | `arm1.bolt.yaml` 的 `handover.reference_point_m` | 用新数据的松手点中位重算；夹持开度范围复核 |
| 归一化路径 | `arm1.bolt.yaml` 的 `normalization` | 指向新契约 |
| 用到旧 episode 的测试 | `tests/rlt_online/test_action_space.py`、`test_exploration.py` | 改指新契约 |

其余已定决策（D1、D6、D9、D13–D16）与代码不依赖具体数据，保持不变。

**同日用户改为"双管齐下"**：采集新数据的同时，用现有 bolt 20k 模型和旧数据派生的参数（边界表、交接点）继续 Stage 2；真机在采集间隙穿插跑 RL。新模型训好后再按上表整体替换。

### 9.2d 2026-11-10 晚：WP3 真机在线回路完成（模拟臂 + 真实推理服务验证）

`yuhang_workspace/wall_oss_rlt/tools/wall_rlt_online.py`：导入并复用 `wall_execute.py` 的 Chunk、Fade、weighted 后端、偏离保护、失速检测、实时画面，不改它；同一进程连续跑多条 episode，训练器常驻内存，每条存一次可续训状态。纯逻辑部分在 `rlt_online/rollout.py`（`ChunkLedger`）。

实现中处理的时序问题：

| 问题 | 处理 |
|---|---|
| WALL 阶段的切段渐变要到旧段走完，RL 阶段会把 actor 的行和旧窗口里 WALL 的预测混在一起 | RL 阶段新块在第 D 步之前切换完；RTC 前缀正好是 D 步、全部硬约束 |
| 观测时刻是相机帧时间戳，相邻两次不一定正好隔 C 帧（模拟 12.6 帧；11-10 真机在第 12 步触发拍到第 11 帧） | 块尾之后停在 actor 最后一步，空档不跑 WALL；触发时刻按实际帧间隔的滑动平均自动挪一步（真机初值晚 1 步）；入库按实际执行行数，间隔不足 C 时存为短的非终止 transition，bootstrap 用 γ^实际步数 |
| 交接后 actor 第一块还没执行，WALL 自己就松手了 | 松手来源记为 `wall`；没有入库任何 RL 行的 episode 不计入 200 条预算 |
| 截断后走完当前块时随机松手块可能张开夹爪 | 截断后夹爪指令冻结 |

模拟臂验证（真实 bolt 20k 服务，假相机画面）：人工松手（最后一块 1 步、人工干预、奖励 1）、按 `f` 判失败、截断（全部不标终止、最后一块满 12 步）、评估模式（`o` 被忽略、不建训练器）、对照模式；RL 块推理时延约 0.25 s、RTC 前缀 11 步、无迟到块。随机松手的端到端路径在模拟画面下触发不到（WALL 总先张开夹爪），由单元测试覆盖。`tests/rlt_online` 59 个加原有共 466 个通过。

真机开跑前还差：①~~夹爪 BC 权重~~ **11-10 用户定 0.002**，配置改名为正式运行配置 `configs/rlt_online/arm1.bolt.yaml`（头部列出仍沿用 §4.3 提议、未逐项确认的值），模拟臂用它干跑通过；②~~RL 阶段死区~~ **11-10 用户定 1 mm / 0.5°**（只在交接后生效，WALL 阶段仍 3 mm / 1.7°；已设为工具默认）；③机械臂与夹爪恢复供电（11-10 16:10 起断开）；④评估每次多少条（D10，运行时 `--episodes` 给）。

### 9.2e 2026-11-10 晚：超参数复核（用 16 次真机 bolt 执行记录重放）

| 项 | 结论 |
|---|---|
| 交接规则（3 cm、开度 0.12–0.30） | 断电前 13 次里 10 次触发，在开始后 9.3–13.3 s、离参考点 21–29 mm（WALL 还在靠近时）；没触发的 3 次分别是偏离保护停机、跑太短、丢了螺栓，都该拒绝。46 次 pick 运行一次没误触发 |
| **阻塞：断电后 WALL 抓不到螺栓** | 恢复供电后 9 次（17:49–18:26）合上后的开度全是 0.01，断电前 7 次都是 0.17–0.20。**11-11 用户确认读数没问题，是这 9 次都没抓到螺栓**，交接开度范围 0.12–0.30 保持不变。与 11-10 晚的分析一致：回放 ep11 能抓到（摆放没问题），合爪位置也和示范一致 ±6 mm，嫌疑是光照（抓得到的都在白天 14:04–14:20，失败的都在天黑后）和调相机这两个混杂因素。WALL 抓不到就不会交接，RL 开不了 |
| 交接后 WALL 何时停稳 | 1.6–9.8 s（中位 4.8 s），停稳时离参考点 5.5–19 mm。RL 阶段 20 s 足够 |
| **建议改：随机松手时机** | 每块 0.08 时，按各次停稳时间算，约 60% 的 episode 会在 WALL 还没停稳就随机松手，多半掉螺栓、要人工复位，数据也集中在靠近阶段。建议只在臂基本停稳后（如 0.5 s 内速度 < 5 mm/s）才允许随机试松手；概率不变。**未定** |
| 其余 | 探索 2 mm / 0.5° 与 RL 死区 1 mm 配合正常；5 mm 位置修正的 BC 代价约 0.13（Q 最多约 1），夹爪约 0.1，强度合适；warmup 150 个 transition 约 10–20 条可达；γ 0.99 下交接时终端奖励折到约 0.1–0.2，偏小但可用，监控 Q 尺度 |

同时给在线工具加了：启动设置摘要、底部实时状态行、需要操作员时响铃、按键大小写都认、每条后累计结果与 warmup 进度、到评估点自动停并打印评估命令、退出打印续训命令、没交接时说明原因、到了附近但开度不对当场提醒；相对路径按启动目录解析。

### 9.2f 2026-11-11～13：bolt2 重训，30k 真机基线 10 次

**新模型。** bolt2 共 100 条，剔除 24、36、57 后剩 97 条，按 93/2/2 划分，契约 `923edc00`。H100 训练 40000 步，每 10000 步存一次。

dev 只有 2 条，对比如下：

| 检查点 | 位置误差 | 旋转误差 | 夹爪误差 | RLT 重建误差 |
|---|---|---|---|---|
| 10k | 10.8 mm | 0.89° | 5.6% | 0.72 |
| 20k | 11.2 mm | 0.91° | 6.7% | 0.66 |
| 30k | 9.7 mm | 0.86° | 6.6% | 0.43 |
| 40k | 9.8 mm | 0.87° | 6.3% | 0.46 |

两个检查点已导出到 5070 Ti：

| 检查点 | 本地目录 | artifact |
|---|---|---|
| 20k | `yuhang_workspace/wall_oss_rlt/artifacts/wall_v3_bolt2_step20000` | `e0b31aae` |
| 30k | `yuhang_workspace/wall_oss_rlt/artifacts/wall_v3_bolt2_step30000` | `82d3f732` |

两个都通过了校验、`--check-only` 和模拟臂预热。H100 上 4 个检查点都还在，每个 76 GB。结果文件在 My Passport `WALL-H100-bolt2-训练结果-20261111/`。

**执行器 `wall_execute.py` 的变化。** 下面几项都是默认开着的：

| 改动 | 时间 | 说明 |
|---|---|---|
| 计划等臂 | 11-12 | `--lag-slow 15 3`、`--lag-stop 40 8`。修复的问题：抓起后臂跟不上，偏离保护停机 |
| 观测同步 | 11-13 | 用户加并审核：臂位姿和顶部图都对齐到腕部图的拍摄时刻 |
| 录像 | 11-13 15:20 | 新文件 `tools/wall_recorder.py`，每路相机存一个 mp4，另存 `video_frames.npz`，时钟和 `ticks.npz` 相同；`--no-video` 关闭 |

改动前的版本都有备份：
- `tools/wall_execute.py.bak_before_obsalign_20261113_102433`
- `tools/wall_execute.py.bak_before_video_20261113_151936`

**基线（P1 的缩小版）。**
- 模型 30k，`--ik sdk`（控制柜解逆解），每条最长 `--max-seconds 60`，死区默认 1 mm / 0.5°。
- 记录在 `~/wall_exec_logs/bolt2_30k_sdk_baseline_20261113/`。`summary.md` 由 `tools/wall_rollout_summary.py` 生成。
- 结果：**用户记 10 次里 9 次成功**，09 失败。
- 用时和示教对比（示教是训练用的 97 条，时间从开始录制算）：

| | 基线 | 示教中位 |
|---|---|---|
| 抓取 | 6.0–8.0 s | 4.4 s |
| 松手（9 条成功的中位） | 29.7 s | 17.5 s |
| 插入用时（9 条成功的中位） | 22.1 s | 12.6 s |

- 有 3 条插得很久，松手在 51.4、56.6、59.4 s；其中两条整条跑满了 60 s 上限。
- 10 条里 6 条松手后夹爪又合了一下，再张开。
- 01–03 没有录像（那时执行器还不录），04–10 有录像，两路相机都没丢帧。

**判读这 9/10 时要注意：**
- **实际一共跑了 13 次，用户舍弃了 3 次后补跑：**
  - 最初的第 4 次：30.2 s 松手，没有标结果，用户说"只保留前三次"后删除。
  - 最初的 09：标了成功，但 60 s 内日志里没有松手。
  - 最初的 10：10.7 s 就停了，没有标结果。
- 10 次远少于协议定的 50 次。
- 起点和判定按 D5、D7 来，但没有用随机起点表。

**Stage 2 接下来要做的。**
1. §9.2c 那几项要按新契约 `923edc00` 重做：边界表、交接参考点（新的松手点中位是 (0.1951, −0.3182, 0.3818)）、归一化路径、测试。没做完之前，`wall_rlt_online.py` 会拒绝新模型启动。
2. Stage 2 用 20k 还是 30k，用户还没定；基线用的是 30k。
3. **控制条件和基线不一致**：`wall_rlt_online.py` 里逆解写死为 weighted（第 589 行），也没有观测同步、计划等臂和录像。它的 `RLExecutor` 用 getattr 取默认值，这几项全是关的。要和这次基线比，先得让用户定 RL 用哪套控制，再把工具对齐。

### 9.3 还没做、且被决策卡住的部分

| 工作包 | 卡在 |
|---|---|
| WP3 环境封装：把动作改写器接进 `wall_execute.py` 的异步 + RTC 回路 | 结构性决策已全部定下（交接、触发、D1、按键、时限）；剩下的是实现：RTC 前缀与 actor 块拼接、键盘监听、松手后上抬与回 0 位、按块请求带 z_rl。运行前还需定夹爪 BC 权重（配置为 null 会拒绝加载） |
| WP4 离线 cache（示范数据逐帧算 z_rl 和参考块） | 交接点的定义决定从哪一帧开始切；是否启用离线 buffer 本身是救援项 |
| WP6 评估模式、WP7 模拟臂端到端干跑 | WP3 |
| 按键与奖励录入 | 交接触发方式、D7 的操作化（"放进套筒"按哪个键、何时按） |

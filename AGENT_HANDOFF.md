# WALL＋RLT：后续 Agent 交接摘要

更新：2026-11-13（**接手先读第 0 节第一行"11-13 交接"**；下面的"当前结论"段落是 10-31 的旧状态）。工作区：`/home/pangk/workspaces/WALL-OSS-CRP`。本文整合当前项目文档、用户现场确认和本轮对话，供接手时快速恢复上下文；不是新的训练或部署验收报告。

**当前结论：5070 Ti 环境构建与官方 WALL 完整推理已通过；解锁关节前的数据已作废并于 11-02 删除，训练动作格式改为官方末端相对位姿（见第 0 节 11-02 两行）。下一步是在 5070 Ti 主机重新采集 → 按官方格式审计与生成新契约 → H100 Stage 1 训练 → 导出 WALL＋Encoder → 5070 Ti 真实观测推理。尚无训练后的 Stage 1 权重，最终 WALL＋RLT 未封版。**

## 0. 2026-10-31 晚 5070 Ti 现场更新（优先于下文旧描述）

| 事项 | 内容 |
|---|---|
| **11-13 交接：现在的状态（以此为准）** | ① **推理服务**：在跑的是 `wall-rlt-5070ti-bolt2-30k-20261112T080056Z-v3`（bolt2 30k，artifact `82d3f732`，契约 `923edc00`）；20k 服务停着、保留。只剩推理镜像 v3。② **执行器** `tools/wall_execute.py`：默认开计划等臂（11-12）、观测同步（11-13 用户加并审核）、录像（11-13 加，`tools/wall_recorder.py`，`--no-video` 关）。录像这处改动用户接受并已用于真机，没有逐行复核。③ **基线**：30k、`--ik sdk`、60 s 上限，用户记 10 次 9 次成功。实际跑了 13 次，舍弃 3 次后补跑。成功各条的松手时间中位 29.7 s，是示教（17.5 s）的约 1.7 倍。记录在 `~/wall_exec_logs/bolt2_30k_sdk_baseline_20261113/`（`summary.md`），细节见研究计划 §9.2f。④ **下一步**：Stage 2 按新契约重做边界表、交接点、归一化和测试（§9.2c）。用户要定：Stage 2 用哪个检查点；RL 用哪套控制。`wall_rlt_online.py` 写死 weighted，没有观测同步、等臂和录像，和基线不一致。另外：随机松手限制还没定；硬盘还剩约 75 GB；其他项目的旧镜像（holobrain、dexgraspvla、openwam 旧版等）等用户定要不要删 |
| **11-13 基线 10 次 + 录像** | 用户：重测基线，30k 跑 10 次，用 sdk 控制器那版脚本，加观测同步（用户审核过）。实验要求记录成功用时和视频。执行器加录像：格式同采集，mp4v、30 fps、640×480，RGB 转 BGR，独立线程，用 `frames_since` 逐帧取。模拟臂和真相机都测过不丢帧，颜色和训练数据一致。新工具 `tools/wall_rollout_summary.py` 出用时汇总，示教对照放在 `~/wall_exec_logs/bolt2_demo_timing.json`。结果见上一行和研究计划 §9.2f。命令见 `常用命令.md` 第 3 节 |
| **11-12 bolt2 训练完成，20k 上线** | H100 结果在 My Passport `WALL-H100-bolt2-训练结果-20261111/`（20 个文件校验一致，约 14 h，无报错／NaN；4 个检查点留在 H100）。dev（29、84 号）上 30k≈40k 最好、20k 最差；用户仍定用盘上已导出的 20k。拷到 `artifacts/wall_v3_bolt2_step20000`（校验一致，`--check-only` PASS，artifact `e0b31aae`），服务容器 `wall-rlt-5070ti-bolt2-20k-20261112T070233Z-v3`（镜像 v3、参数同旧 bolt 服务）在跑，模拟臂预热通过（首段 1.6 s，之后约 0.25 s）。真机还没跑。`wall_rlt_online.py` 在边界表按新数据重做前会拒绝这个模型（§9.2c）。命令见 `常用命令.md` 第 2、3 节。**同日 16:01 用户又导出 30k 做 rollout**：`artifacts/wall_v3_bolt2_step30000`（校验一致，check-only PASS，artifact `82d3f732`，来源哈希 = dev 评估的 30k），服务 `wall-rlt-5070ti-bolt2-30k-20261112T080056Z-v3` 在跑（20k 服务已停、保留）。用户要求删旧镜像：已删推理镜像 v1 `5070ti-20261031T065922Z`（连同 10-31 构建残留 4 层）、v2 `5070ti-20261105T020149Z` 及其 11-05 旧容器 `wall-rlt-5070ti-20261105T063509Z`；`runs/official-wall5070-…/session.env` 里 v1/v2 的 tag 已失效 |
| **11-11 bolt2 采完、打包送 H100** | 新 bolt 100 条（`~/atom_episodes/wall_unlocked_v3/bolt2`）检查后剔 24、36、57（移到 `wall_unlocked_v3_rejected/bolt2/`），97 条、划分 93/2/2，契约 `wall_v3_bolt2_…` 指纹 `923edc00…`。My Passport 包 `WALL-RLT-training-candidate-20261111-bolt2-v1` + 文档 `WALL-H100-bolt2-训练-20261111.md`：40000 步、每 10000 存，其余同上次 bolt；训练后画 loss 曲线（`tools/plot_wall_rlt_training.py`）和 dev 对比，用户选定后再导出。新模型到了以后，Stage 2 里从旧数据算的东西要按研究计划 §9.2c 重做（新松手点中位 (0.1951, −0.3182, 0.3818)）。上午 RL 真机 9 条（入库 2 条成功）在 `~/wall_rlt_runs/bolt20k_20261111` |
| **11-10 夜 双管齐下：Stage 2 真机在线工具完成** | 用户：采新数据的同时，用现有 bolt 20k 模型继续 RLT，真机在采集间隙穿插。新工具 `tools/wall_rlt_online.py`（复用 `wall_execute.py`，不改它；训练 / 评估 / WALL 对照三种模式、续训、模拟臂干跑参数），用法见 `常用命令.md` 第 6 节；纯逻辑在开发工作区 `rlt_online/rollout.py`。模拟臂接真实服务验证了人工松手、判失败、截断、评估、对照五条路径；466 个测试通过。真机开跑前要用户定夹爪 BC 权重、RL 阶段死区，并恢复臂和夹爪供电（11-10 16:10 起断开）。细节见研究计划 §9.2d |
| **11-10 晚 Stage 2 暂停，重采 bolt** | 用户：先再采约 100 条 bolt、重训 Stage 1，看纯 WALL 的上限；因相机偏移又复位，旧 50 条 bolt 倾向弃用（不删）。新数据放 `~/atom_episodes/wall_unlocked_v3`（`python3 tools/wall_collect.py --out ~/atom_episodes/wall_unlocked_v3 -n 100`，dry-run 通过，参数与 v2 相同）。Stage 2 已定决策与 `rlt_online/` 代码保留；依赖旧数据的边界表、交接参考点、契约与模型要在新数据后重做，清单见研究计划 §9.2c |
| **11-10 Stage 2 代码开工** | 用户：顶部相机已复位；纯 WALL 基线成功率推迟。开发工作区新建独立包 `rlt_online/`（不进 `wall_rlt/`，避免遮蔽 `wall_rlt/stage2.py`、不进 H100 训练包）：Evo-RLT 核心与在线 trainer 原样 vendor 在 `rlt_online/_vendor/`（CRC32 对过 zip），动作空间编解码与按窗口步的边界提案（D2 未确认），trainer 用的策略替身。`tests/rlt_online` 23 个 + 原 `tests/rlt` 共 440 通过；dexgraspvla 环境 GPU 冒烟通过；对运行中的 bolt 20k 服务核对编解码与 `absolute_targets` 一致。环境封装、离线 cache、评估、按键都卡在交接方式与 C、σ 等决策上。详见研究计划第 9 节 |
| **11-05 下午 Stage 2 科研计划** | 用户定下 Stage 2 的做法：尽量复用 Evo-RLT 源码（My Passport `wy/EVO/Evo-RLT.zip`，8 月 3 日版；只解压 src/docs/tests）、按本项目改适配层、设对照组调参、目标是提高 bolt 成功率。计划在开发工作区 `WALL_RLT_Stage2_Research_Plan.md`：P0 前置（重启、GPU 验证、从 4 个 bolt checkpoint 里选一个、执行器真机验证）→ P1 纯 WALL 基线 50 条（对照组）→ P2 代码工作包 WP1–WP7（全部在模拟臂与离线数据验收）→ P3/P4 主实验 400 条 → P5 消融/救援。架构：单进程跑在 `dexgraspvla` 环境（已有 torch 2.11+cu128），在 `wall_execute.py` 的 Runner 上加动作改写器与在线 trainer；Evo-RLT 核心原样 vendor 到 `wall_rlt/_vendor/evo_rlt_core/`，trainer／modifier／离线 cache 改写到 `wall_rlt/stage2/`。主配置忠于论文，Evo-RLT 自加的 milestone／time_decay／离线 buffer／RankQ 默认关。计划 §7 的 D1–D13 是**建议值，尚未经用户确认**。当天权重已到（`artifacts/wall_v2_bolt_step{10000..40000}`，v2 镜像上 `--check-only` 四个都 PASS），本机仍未重启 |
| **11-05 进入 Stage 2 准备** | 用户告知 H100 bolt Stage 1 权重正在拷贝，下一步 RLT Stage 2。本机当天状态：**GPU 不可用、需重启**（06:47 无人值守升级把驱动升到 595.99.02 并装了内核 7.0.0-42，运行中的模块仍是 595.91.07，`nvidia-smi`／`nvidia-container-cli` 报 version mismatch；`/var/run/reboot-required` 存在；重启由用户做）。**推理镜像已重建**：从当前开发工作区打新工具包 `incoming/WALL-RLT-inference5070-20261105-v2`（INTEGRITY_PASS），构建出 `wall-rlt-inference:5070ti-20261105T020149Z`（ID `sha256:8205eabd…`，静态 sm_120 检查 PASS，flash-attn／wall-x 层复用缓存，只换了 project 层），tag／ID 以 `WALL_*_V2` 追加在 `runs/official-wall5070-20261031T063658Z-n0xT6K/session.env`，旧镜像 ID 不动；新镜像的 GPU 小算子与推理验证要等重启后做。原因：开发工作区有 13 个 `wall_rlt` 文件与旧镜像不同，含 `MODEL_SOURCES` 内的 `builder.py`（补零修复 `f83b2aeb…`）、`data/ar5.py`、`runtime.py`。权重到位后的步骤、Stage 2 决策表（论文参考值＋候选）、RLinf 在线栈的缺口分析都写在开发工作区 `WALL_RLT_Stage2_Design_Decisions.md` 第 0 节，**没有任何 Stage 2 参数已定版** |
| **数据作废** | 用户决定：解锁关节前采的数据**全部作废**，包括 `top_error` 全部任务（j3、j5 锁死）、350 条训练候选、契约 `top_error_ar5_joint_next_v1` 及其统计。不删除，只是不再使用。下文第 2、5 节凡涉及这批数据的结论都随之失效 |
| **重新采集** | 新数据用笛卡尔遥操、`weighted` 逆解、7 轴全动。5070 Ti 上的入口是 `wall_oss_rlt/tools/wall_collect.py`，它调用 SO101 的 `collect.py`：固定 OmniPicker / 笛卡尔 / weighted / 位置 0.8 / 转动 0.2 / 姿态权重 0.5 / 预览（11-02 真机调定；sdk-MoveL 试过，停顿 27%、落后 0.6 s，弃用），起始姿势用 `config.py` 的 `cartesian_centre_joints`（夹爪斜向下 50°，11-02 加；11-03 提交为 DexGraspVLA- `9aec332`，分支 `ar5-cartesian-start-posture`，未合并 main、未推送），其余用 `config.py`（指尖 130 mm、50 Hz / 30 Hz）。11-02 两条试采 `orange_block` 审计 PASS、j3/j5 都动、EE 契约生成正常。生效参数和采集代码版本写进数据根目录的 `wall_collection.json`；同一根目录参数不一致、或往旧数据目录写，都会被拒绝。11-03 用户与 MoveL 对比后定用 weighted，在 `~/atom_episodes/wall_unlocked` 采了 2 条（orange_block）。同日用户要夹爪能转大角度：原映射把主臂整只手的转动换到基座系统一乘 0.2，拧主臂手腕大半成了从臂夹爪"歪"（起始姿势拧 30° 转 13°、歪 27°）。SO101 加 `--spin-scale`（主臂 wrist_roll 直接给夹爪自身轴；提交 DexGraspVLA- `ec624a6`，同一分支），启动器固定 `--spin-scale 0.6`（试过 1.0、0.7 后用户定 0.6）。存盘后回 0 位被跳变门拒死、臂抱死（11-02、11-03 各一次）已修：`goto_joints` 从驱动设定点起步，提交 `ce1a0a8`。指尖被工作空间盒 x 下界 0.20 挡住（操作区在臂右前方），用户定盒子下界改为 x 0.00、y -0.60（z 0.08 不动），提交 `7cedebf`；11:00 起正式采集进 `wall_unlocked_v2`，默认根目录改为 `~/atom_episodes/wall_unlocked_v2`；旧根目录那 2 条参数不同，不再续。11-03 采完 50 条 `orange_block`（全在 `7cedebf` 之后录），`wall_dataset_check` PASS（`data-audit/check-v2-20261103-133201`，要加 `--exclude-group 1`：`1/` 是误建的空任务目录）；逐条看：49 条结尾都夹着方块抬起，ep44 方块滑掉、空手结束，ep30 滑掉后重抓（70 s），ep41 夹爪迟 4.7 s 才动、臂停 4.5 s；用户已删 44、把 30/41 移到 `~/atom_episodes/wall_unlocked_v2_rejected/`，补采 50–52。**11-03 16:10 采集全部完成**：orange_block、black_bushing、bolt 各 50 条，共 66286 帧，全在 `7cedebf` 上；全量检查 PASS（`data-audit/check-v2-all-20261103-161413`）。用途（用户定）：**bolt 专用于 RLT**（"insert the bolt into the sleeve"；螺栓、套筒位置固定，松手点各轴标准差约 8 mm；插入时夹爪斜 50°，俯视相机能看到一部分套筒边，腕部相机只看到螺栓头；最后 3 cm 中位 5.7 s；ep30 只有 2.3 s、没夹过，待剔），**orange_block、black_bushing 用于研究 WALL-OSS 的泛化**。用户同意后删了 bolt ep30 和空目录 `1/`，定两个模型：bolt 单独一个，另两个合一个。随后在 `~/workspaces/WALL-OSS-CRP`：
- 重写 `数据集格式说明.md`（sha256 `29772dd8…`）。
- 两次审计（互相排除）。
- `prepare_ar5` 生成两份 EE 契约：`wall_v2_bolt_…`（34/7/8）、`wall_v2_pick_…`（70/14/16）。
- `wall_rlt/data/check.py` 补了 EE 分支，原来写死 joint8；两份 CPU batch PASS，全套 400 项通过。
- 打两个训练候选到 My Passport：`WALL-RLT-training-candidate-20261103-{bolt,pick}-v1`，清单 `WALL-H100-SHA256SUMS-20261103.txt`；试解包＋解包后检查 PASS。

详见该工作区 `PROJECT_STATUS.md` 第 0 节。

**同日晚 v2，以此为准：** 用户补采 bolt 50 号，bolt 划分改 46/2/2（audit.py 加 `--dev-per-task/--test-per-task`）。说明 sha256 `c3e4af54…`，契约 bolt `f47efc75…`、pick `94a55ed2…`，401 项通过。新包 `WALL-RLT-training-candidate-20261103-{bolt,pick}-v2`，清单 `WALL-H100-SHA256SUMS-20261103-v2.txt`。v1 包已过时，未删，待用户定。

**H100 操作说明：** My Passport `WALL-H100-训练操作-20261103.md`。
- 要点：D0 镜像里**没有 git**，改为在主机上解包、核对源码、拷入 GPU 扩展；容器按原路径挂载；`--ipc=host`；两卡配置；导出用新包自带的脚本。
- 第 3–4 节已在 5070 Ti 用 D0 镜像演练通过，GPU 步骤未跑 |
| **采完检查** | `wall_oss_rlt/tools/wall_dataset_check.py`，在 sm_120 推理镜像里原样调用 `wall_rlt.data.audit.audit_npz`，再检查参数记录、sdk 试验数据、j3/j5 是否真的动了、软限位余量、逆解跳变、实测帧率。作废的 black_sleeve 用它检查：审计 PASS，但被判"锁关节数据"不通过 |
| **开发机要改 `prepare_ar5.py`** | 原版写死了 `locked_joint_indices_zero_based=[2,4]`，还强制要求排除 mouse、tape，新数据过不了。补丁在 5070 Ti 的 `wall_oss_rlt/patches/wall_rlt-prepare_ar5-unlocked-20261031.patch`，改为按 train 实测锁定关节、排除组参数化、采集参数写进 provenance。只改这一个文件，不涉及推理侧 `MODEL_SOURCES`，5070 Ti 现有镜像不用重建。验证范围见补丁头；`tests/rlt` 要在开发机 `.venv-rlt` 里复跑 |
| **数据说明** | `数据集格式说明.md`（契约里记录的 sha256 为 `0b7b76…`）描述的是作废数据。等新数据采完、核对实测后再重写，用户要求在此之前不写 |
| **官方 WALL 真实观测回放** | 在作废的 black_sleeve 上零样本回放 626 次，只用来验证推理链路：之后的推理中位 **227 ms**、p90 230 ms，峰值 7.92 GiB，同一 seed 逐位可复现。零样本精度不如"原地不动"，不能用于控制。结果见 5070 Ti 的 `runs/official-wall5070-20261031T063658Z-n0xT6K/replay/RESULTS.md`；精度数字不作为后续对照基准 |
| **11-02 旧数据已删除** | 用户决定删除 5070 Ti 的 `~/atom_episodes/top_error`（black_sleeve）和 My Passport 的 `top_error`（10 类，53,997 个文件），删前确认无 10-31 之后的文件。旧训练候选包 `WALL-RLT-training-candidate-20261029-v1`、`~/atom_episodes` 下其他非 WALL 采集、回放结果保留 |
| **11-02 开发机工作区迁到 5070 Ti，EE 模式已实现** | 恢复在 `~/workspaces/WALL-OSS-CRP`（含软链接，故不放 yuhang_workspace）。新增 `ar5_ee_rel6d_next_v1`／`wall_oss05_right_arm_ee_v1`，夹爪标签按用户选择保留二值指令，槽位 10–19，不重置投影；394 项 CPU 测试通过。状态与后续事项以该工作区 `PROJECT_STATUS.md` 第 0 节为准 |
| **11-02 打包泛化、执行器** | 开发机工作区的训练打包脚本改为必须给 `--contract`（400 项 CPU 测试通过）。执行器 `tools/wall_execute.py`：默认采集同款 weighted 逆解＋实时关节流，`--ik sdk` 走 MoveL 队列；有偏离保护（5 cm／20°）；姿态权重默认取 `wall_collect.py` 固定的采集值 0.5（11-03，`--ori-cost` 可改，记进 run.json），不用 config 的 0.15；只在模拟臂上验证。真机先跑 `--fake-policy hold`，再把臂摆到某条新数据首帧后跑 `--fake-policy replay:<该条>` |
| **11-02 训练改用官方末端相对位姿** | 用户决定。依据 wall-x `ce2a1ae`（即上游最新，2026-09-11）：LeRobot v3；`observation.state`＝当前帧末端实测位姿＋夹爪，`action`＝**下一帧**末端实测位姿（`x2robot2lerobot.py:969-970`，跟随臂实测值，不是指令）；loader 把姿态转 6D、把 32 步动作换成相对当前位姿，`action_padding` 补到 26 维；训练前跑 `scripts/compute_norm_stats.py`。参考 `workspace/README.md` 第 3 部分、`workspace/example/maniparena_example.yml`、`workspace/TRAIN_ARRANGE_3_FLOWERS.md`。现行 joint8（`ar5_joint8_reinit_v1`，重置两处动作投影）将被替换，上面的 `prepare_ar5` 补丁只适用于 joint8，仅作参考。**采集不变**：npz 已有每帧末端实测 `ee_pose`、夹爪开度、两路相机。因官方标签也是下一帧实测值，不再计划在采集时记录关节指令，DexGraspVLA- 暂不改动（用户要求）。开发机工作区准备迁到 5070 Ti，在那里改训练侧；5070 Ti 根盘 11-02 只剩 55 GB |

## 1. 接手后先做什么

1. 先读本文及 [PROJECT_STATUS.md](PROJECT_STATUS.md)。接受用户已确认的 5070 Ti 结果，不以回传原始日志或重跑官方模型作为继续工作的前提。
2. 聚焦 H100 的训练环境、实际分配 GPU、完整模型前后向、100 步短训和 checkpoint 恢复。现场操作由用户执行，本工作区不主动连接两台目标机。
3. 有实际 Stage 1 checkpoint 后，从 [推理部署说明第 7 节起](WALL_RLT_Inference_Deployment.md#7-训练后才能执行导出与验证-stage-1-推理-artifact)推进导出、真实观测回放及服务重启比较。
4. 审核优先找影响实际训练和推理的功能问题。已有通过项不因旧文档措辞退回待验；没有新修改、失败或疑点时，不重复构建、下载和大归档校验。

信息冲突时：以用户最新明确指令／现场确认及相应新记录为准；项目状态入口为 `PROJECT_STATUS.md`。本摘要是截至上述日期的接手快照。技术长文中的旧状态不覆盖新结论，已封包文档和原始日志保留其历史含义。

## 2. 已完成与未完成

| 项目 | 当前结论与边界 |
|---|---|
| 数据接入 | ~~已有目标数据、划分、统计和 CPU batch 验证；无需新采集~~ → **10-31 作废，需重新采集**（第 0 节）；数据管线代码和 CPU batch 验证仍然有效 |
| Stage 1 软件 | joint8 适配、WALL＋RLT 联合训练、RLinf worker、checkpoint／恢复已实现，CPU 检查通过；完整 H100 训练未验收 |
| 推理软件 | 导出、严格权重恢复、HTTP 服务、录制观测回放和重启比较已实现，微型模型 CPU 检查通过 |
| 5070 Ti 环境 | 专用 `sm_120` 镜像已构建，BF16 matmul、FlashAttention 通过 |
| 5070 Ti 官方 WALL | 用户确认两个独立容器的合成观测完整推理通过；输出 `32×26`，数值有限；峰值 allocated/reserved 约 7.94/8.07 GiB |
| H100 Stage 1 | 待环境验收、真实前后向、短训／恢复和后续训练；尚无本项目训练后的 WALL＋Encoder 权重 |
| 5070 Ti WALL＋Encoder | 待 Stage 1 权重；AR5 joint8、真实观测、Encoder 特征及目标服务未验收 |
| Stage 2 | 只有合成 CPU 单步 Actor／Critic 更新；无正式在线主循环、真实 replay、训练后的 RL Actor 或收益结果 |
| 机器人／最终成品 | 未执行机械臂闭环，未完成任务效果验收；最终版本尚未封版 |

5070 Ti 的依据为 [现场验证记录](metadata/rlt-5070ti-validation-20261031.json)及[用户部署汇报](metadata/rlt-5070ti-validation-20261031/deployment-report.md)。首次推理 2.43 s／1.39 s 不是稳定控制频率；两次输出形状和值域一致也不能代替完整动作逐项比较。

## 3. 主机、设备和传输约定

| 对象 | 信息与职责 |
|---|---|
| 当前开发机 | 本工作区所在机器，历史构建 GPU 为 RTX 3060 Ti 8GB；负责代码、CPU 检查和交付准备，不承担完整训练 |
| H100 | `aiuser@10.241.90.194`；用户确认共有 8 卡，负责训练及后续复验；单卡显存和本次 GPU 分配仍需现场确认，不能默认占用全部卡 |
| 5070 Ti | Ubuntu 24.04.4，主机 `crp-5070ti-01`，RTX 5070 Ti 16303 MiB，驱动 595.91.07；目录 `/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt`；负责推理 |
| 已通过的推理镜像 | `wall-rlt-inference:5070ti-20261031T065922Z`，位于目标 5070 Ti 主机；源码／模型契约匹配时可复用 |
| 移动硬盘 | 开发机挂载路径 `/media/pangk/My Passport`；换机后按实际挂载点／盘符使用 |
| 机械臂 | 用户指定 ROKAE AR5＋夹爪；当前模型服务不下发机器人动作。**2026-11-14 起台上换成天机 Marvin 双臂，AR5 已换下**，执行端还没接入，见 `~/yuhang_workspace/数据_硬件/天机Marvin/README.md` |

H100 中转路线：**My Passport → 用户电脑 → 专用网线 → SSH/SCP → H100**。网关只能用 `10.241.245.6`，1～5 已占用；掩码 `/26`，上传端 IP 由现场分配。Windows 仅负责文件传输，不做环境部署、解包或 Windows 专项校验；完整性检查在 Ubuntu 执行。已有压缩归档无需再次压缩。密码不写入交接文档或交付包。

本机不主动连接 H100 或 5070 Ti；历史 SSH 超时不是继续重试的指令。遇到新的网络问题直接说明具体失败，不掩盖成其他原因。早期“全部推理放服务器”的路线已由上述主机分工更新。

## 4. 三份已交付包

都位于 `/media/pangk/My Passport/`，是各自冻结时的内容：

| 目录 | 规模 | 用途 |
|---|---:|---|
| `WALL-OSS-CRP-transfer-20261029` | 29 文件，16,787,299,846 字节 | D0 Docker 基础镜像、官方 WALL／processor、32 条历史验证数据及校验工具；已有 `BUNDLE_READY` |
| `WALL-RLT-training-candidate-20261029-v1` | 8 文件，5,781,431,598 字节 | 固定训练源码、44 个离线 wheel、350 条目标轨迹；不是已训练模型 |
| `WALL-RLT-inference5070-20261029-v1` | 8 文件，10,979,923 字节 | 6,888 个源文件及离线构建、导出、服务、回放工具；包内没有已编译的 5070 Ti 镜像或训练后权重 |

合计 **45 文件、22,579,711,367 字节（约 22.58 GB）**。D0 的 SHA256 清单有 27 项，另加清单自身与完成标记才是 29 文件。旧 `WALL-OSS-CRP-transfer-20261028` 为 `INCOMPLETE`，不能部署。旧两包 37 文件的清单不覆盖第三份工具包。

依据：[D0 交付记录](logs/server-environment-delivery-20261029.json)、[训练候选记录](metadata/rlt-h100-preparation-20261029.json)、[推理工具包记录](metadata/rlt-inference-deployment-20261029.json)。这些旧记录中的目标机 `NOT_RUN` 描述打包时状态，5070 Ti 后来的通过结论见 2026-10-31 记录。

工作区修改不会自动进入已封包或目标机镜像。不要原地改写已冻结包及其哈希；需要交付代码更新时，明确新版本与相应验证。2026-10-29 训练候选不含后加的导出入口；H100 导出还需携带完整推理工具包，见部署说明第 7 节。

## 5. 训练与推理必须保持的数据／模型约定

- **（10-31 作废，见第 0 节；以下前两条只记录旧数据，新数据的数量、划分、锁定关节以新契约为准）**
- 数据：`My Passport/top_error`，排除 `mouse`、`tape`；保留 7 类，每类 50 条，共 **350 条、176,523 帧**。按 episode 分层划分 train/dev/test＝**245/49/56**，不把相邻帧随机拆到不同集合。
- 目标：当前双图、指令、state 监督下一帧 **7 个实测关节字段＋二值夹爪指令**；每条末帧没有未来标签，丢弃后为 176,173 个有效观测。这是数据契约中的字段语义，不证明已恢复当时实际下发的关节命令或完成机械臂执行映射。
- 原始末端增量描述上一帧到当前帧的实测变化，不能直接冒充未来关节命令；实测夹爪开度与开合指令也不能互换。相机为 `cam_high`＝俯视、`cam_front`＝腕部。
- 机器可读契约：[top_error_ar5_joint_next_v1.json](configs/rlt/top_error_ar5_joint_next_v1.json)。`target_mode=ar5_joint_next_v1`，`model_adapter=ar5_joint8_reinit_v1`，模型内部 26 槽，前 8 槽有效，预测长度 **H=32**。训练统计只由 train 计算并绑定摘要；统计分位数不等于机器人控制边界。
- 新迁移严格加载底座后重置两处动作投影，再安装目标 normalizer；**恢复和推理不重置**。实现：[builder](wall_rlt/model/builder.py)、[AR5 adapter](wall_rlt/model/ar5_adapter.py)。
- Stage 1：`L_total = L_rlt + alpha × L_vla`，当前 alpha＝1。WALL 由 flow loss 更新；Encoder／训练 Decoder 重建观测前缀，对 WALL 停止梯度。全局梯度裁剪可能间接耦合两分支，不能预设与纯 WALL SFT 更新完全相同。
- 推理产物只含 **WALL＋Encoder**；服务返回 `32×8` 的 WALL 物理空间参考动作和 Encoder 特征（当前 `z_rl` 为 2048 维）。Encoder 特征不会自动改写 WALL 动作；服务没有训练 Decoder、RL Actor、Critic 或优化器，也不执行机器人。

数据证据：[审计](metadata/rlt-top-error-selected-20261029/audit.json)、[CPU batch](metadata/rlt-top-error-selected-20261029/cpu-batch.json)。D0 的 32 条历史 smoke 数据不等于这 350 条目标数据；未进行全部视频逐帧质量或逐条抓取成功率验收。

## 6. 下一阶段的执行入口与通过条件

| 顺序 | 入口／操作 | 需要获得的结果 |
|---|---|---|
| 0. 重新采集与数据准备（10-31 新增，11-02 改） | 5070 Ti：`tools/wall_collect.py` 采集，`tools/wall_dataset_check.py` 检查；迁移过来的开发机工作区：新增官方末端相对位姿的目标模式和数据准备（替代 joint8 的 `prepare_ar5`），生成新契约，重写数据说明，重新打训练候选包 | 检查 PASS；新契约按官方 26 维末端布局，provenance 带采集参数；新训练候选包及其校验 |
| 1. H100 准备 | [实施计划第 9.2 节](WALL_RLT_Agent_Implementation_Plan.md#92-h100-训练候选的迁移与预检)、`scripts/unpack_wall_rlt_training.py`、`scripts/prepare_wall_rlt_training_env.py` | 解包到最终运行路径；新建离线训练 overlay；解决依赖失败，保留环境报告 |
| 2. 资源与单步 | `scripts/check_wall_rlt_h100.py`、`configs/rlt/stage1_2gpu.yaml` | 核实已分配 GPU、显存和原生扩展；完成真实模型前向、反向与优化更新 |
| 3. 训练／恢复 | `scripts/train_wall_rlt.py`、`wall_rlt/workers/sft.py` | 100 步短训、有限 loss、实际分片保存／恢复；随后按实验配置完成 Stage 1，100 步不自动等于训练充分 |
| 4. 导出 | `scripts/export_wall_rlt_inference.py --checkpoint /实际checkpoint/actor --output /新的artifact目录` | checkpoint sidecar、资产及合并后的 `model_state_dict/full_weights.pt` 齐全；产出 `wall.safetensors`、`encoder.safetensors`、`assets/`、`manifest.json` |
| 5. 目标推理 | [推理部署说明第 7～9 节](WALL_RLT_Inference_Deployment.md)、`scripts/serve_wall_rlt.py`、`scripts/check_wall_rlt_inference.py` | 同一 artifact 的真实双图／state／任务文本输入，joint8 动作与特征，显存／时延及不同服务进程的固定输入重启比较 |
| 6. 封版 | 冻结代码、依赖、权重、processor、normalizer、配置和报告 | 完整目标推理通过后制作最终包，再到 H100 复验同一版本；各 GPU 的报告单独保留 |

执行约束：

- H100 当前为 **双卡 FSDP full shard 候选**：micro batch＝1、global batch＝4，FP32 主参数，WALL BF16 mixed precision，RLT FP32；双卡关闭 dev 循环，`num_workers=0` 保持恢复语义。总卡数不代表已分配卡数，也不证明显存足够。
- 训练启动显式提供 `--world-size 2` 和两次 `--gpu`（实际分配的物理编号或 UUID）；仅设置 `CUDA_VISIBLE_DEVICES` 不足以绑定 RLinf placement。单卡诊断的 `--run-single-step` 不能当双卡诊断使用。
- 恢复同时设置 `actor.model.initialization=resume` 与 `runner.resume_dir=/实际checkpoint/global_step_N`，保留原训练配置、数据契约和源码匹配关系；不要用新迁移初始化覆盖已训练投影。
- 导出器读取已经合并的完整权重，不替代 FSDP 合并。目标推理核对模型／预处理源码摘要和归一化统计，不要跳过检查强行加载不匹配产物。
- 现有 `sm_120` 镜像可在源码匹配时复用；H100 使用 `sm_90`。5070 Ti 上不运行要求 `sm_90` 的 `verify_h100_bundle.sh`。

## 7. 已知环境问题与最新修复

**D0 的 `pip check` 仍为 FAIL，共 11 项。** 缺失声明依赖为 `cmake`、`deepdiff`、`diffusers`、`gymnasium`、`imageio`、`jsonlines`、`pynput`、`pyserial`、`rerun-sdk`、`torchcodec`；另有 LeRobot 0.4.4 的 Hub 约束与实际 1.33.0 冲突。Transformers 5.2 同时要求较新 Hub，不能靠盲目降级解决。训练 overlay 检查即使 `runtime_checks_status=PASS`，仍可因声明失败返回 `BLOCKED`；不忽略退出码宣称训练环境已验收。

目标基础环境为 Python 3.10／Torch 2.10.0+cu128／CUDA 12.8；本机原 `env` 是另一套 CUDA13 环境，CPU 实现测试使用已有 `.venv-rlt`，不要混用或覆盖。依赖约束见 [constraints-training.txt](configs/rlt/constraints-training.txt)。`nvidia-smi` 显示 CUDA 13.2 是目标驱动兼容能力，不是镜像 toolkit 版本。

5070 Ti 现场已处理两类实际问题：

1. 任意 UID 容器缺少用户名／可写 HOME，导致 Torch 用户查询或 Triton 缓存失败。上一轮已在工作区 5 个文件的 **14 处 Docker 命令**补入下列参数；使用现行部署说明。该修改未改写已封包或目标镜像，无需为了参数补充重编 CUDA 扩展。
2. 官方权重省略 tied `lm_head.weight`，并额外保存 224 个重复 normalizer 张量。现场检查器核对实际绑权及重复张量完全一致后通过，其他缺失／额外键仍报错。原始通用检查器与包内快照未自动更新；不要把这个已确认的兼容情况当作官方推理未通过，也不要泛化成任意忽略权重键。

在带 `--user "$(id -u):$(id -g)"` 的容器命令中使用：

```bash
-e USER="$(id -un)" -e HOME=/tmp \
-e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
```

14 处修改位于 `WALL_RLT_Inference_Deployment.md`、`WALL_RLT_5070Ti_Official_WALL_Runbook.md`、`docker/OFFLINE-DEPLOY.md`、`docker/verify_h100_bundle.sh`、`scripts/docker_build_validate.sh`。[检查记录](logs/rlt-stage1-review-20261031/docker-command-check.json)为 shell 语法与参数检查 PASS，不是新的容器／GPU 实跑。

## 8. 测试证据、源码和 Stage 2 边界

| 证据 | 正确解释 |
|---|---|
| [2026-10-29 全套 CPU 报告](logs/rlt-inference-20261029/cpu-check/report.json) | 377 passed，0 failures/errors/skips；微型模型、接口、源码与预检等，不替代完整 H100 训练 |
| 2026-10-31 上一轮专项复跑 | 对话中子 agent 回报训练相关 52 passed、导出／推理相关 46 passed，共 98；是已有用例复跑，不是新增 98 项；没有另行保存独立 pytest 报告，不伪造报告路径 |
| 更早 147／218 项 | 不同阶段的历史快照，不与 377 或 98 累加，不回填改写旧报告 |
| FSDP 恢复的 CPU 检查 | worker 测试模拟上游分片保存／恢复；实际双 H100 分片恢复仍须现场验收 |

上一轮 98 项涉及 `tests/rlt/` 下的 `test_builder.py`、`test_sft_worker.py`、`test_checkpoint.py`、`test_explicit_placement.py`、`test_inference_artifact.py`、`test_inference_runtime.py`、`test_inference_replay.py`，使用 `.venv-rlt`、禁用 CUDA、离线模式。全套入口为 `scripts/check_wall_rlt.py`；按实际变更选择必要检查，不因接手而自动重跑所有测试。

固定源码：Wall-X `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`；RLinf `e1a82ebf0c1af69da6ab9b1519f5f3692fe33566`。锁文件为 [metadata/wall-rlt-sources.lock.json](metadata/wall-rlt-sources.lock.json)，WALL 工作副本为 `.cache/wall-x-rlt`，补丁为 `patches/wall-x-rlt-prefix.patch`，RLinf 为 `.cache/rlinf-rlt`。审核实际运行路径时不要只看未打补丁的 `wall-x`。锁文件很大，读取所需字段即可。

**Stage 2 正式设计仍未定。** 依据 [Stage 2 决策清单](WALL_RLT_Stage2_Design_Decisions.md)，reward、warmup、transition、RL chunk、replay、更新频率和评估预算由用户决定，agent 负责整理方案、落实配置与验证；现有测试值不自动成为正式参数。H＝32 是 WALL 预测长度，当前 C＝1 只是 Stage 2 单步测试范围。示范数据缺 reward／terminated／truncated，下一帧实测关节标签也不能自动当成 RL replay 的实际执行动作。Stage 1 不会产生训练后的 RL Actor。

## 9. 按需阅读索引与历史误区

- 当前状态／简版：[PROJECT_STATUS.md](PROJECT_STATUS.md)、[WALL_RLT_Human_Review.md](WALL_RLT_Human_Review.md)。
- 训练细节／命令：[WALL_RLT_Agent_Implementation_Plan.md](WALL_RLT_Agent_Implementation_Plan.md)。其中部分 2026-10-29 的 5070 Ti `NOT_RUN` 已过时；技术约束仍应结合代码核对。
- 原理与数据语义：[WALL_RLT_Implementation_Review.md](WALL_RLT_Implementation_Review.md)、[数据集格式说明.md](数据集格式说明.md)。其中旧 GPU 状态同样以新状态入口为准。
- 后续目标推理操作：[WALL_RLT_Inference_Deployment.md](WALL_RLT_Inference_Deployment.md)。官方底座历史操作：[WALL_RLT_5070Ti_Official_WALL_Runbook.md](WALL_RLT_5070Ti_Official_WALL_Runbook.md)。
- H100 连接／基础部署：[H100使用说明.md](H100使用说明.md)、[docker/OFFLINE-DEPLOY.md](docker/OFFLINE-DEPLOY.md)，选择对应主机和阶段。
- `H100.md` 是 LeRobot pi05 的迁移案例，运输方式可参考，其包名、版本、5 步训练通过记录不能用于证明 WALL 训练已通过。
- `stage1_report.md` 的“第一阶段”是 2026-10-26 环境准备，不是 RLT Stage 1 训练。旧采集计划不构成新采集任务；10-31 的重新采集是用户的新决定，按第 0 节执行。
- D0 `runtime.log` 中 RTX 3060 Ti 的组件检查属于开发机构建证据，不能改记 H100 验收。未取得现场新结果时，保留 H100 训练和训练后目标推理的待验状态。

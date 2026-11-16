# WALL＋RLT 项目当前状态

更新：2026-11-13（**最新状态看第 0 节第一行**；下面的"当前阶段"段落是 10-31 的旧状态）。本文是项目当前状态入口；技术设计与操作命令仍在各专项文档中。状态依据项目记录与用户确认。用户已确认在目标机完成复验，本次 5070 Ti 官方 WALL 部署记为通过，不以回传原始日志或检查器为继续工作的前提。

**当前阶段：RTX 5070 Ti 专用镜像已构建，GPU 小算子及官方 WALL 合成观测完整推理已通过，并经用户现场复验确认。下一步在 H100 开展 Stage 1 训练，导出 WALL＋Encoder，再回到 5070 Ti 验证目标数据推理。训练后的权重和最终 WALL＋RLT 封版尚待完成。** 最新软件检查仍为 377 项 CPU 测试通过。

## 0. 2026-11-02／11-03 更新（优先于下文旧描述）

| 事项 | 内容 |
|---|---|
| **11-13 bolt2 30k 真机基线 9/10（当前状态）** | **训练结果。** H100 bolt2 训练 40k 步完成，结果在 My Passport `WALL-H100-bolt2-训练结果-20261111/`，20 个文件校验一致。dev 只有 2 条：30k 与 40k 最好，20k 最差。**导出。** 20k、30k 已导出到 5070 Ti：`wall_oss_rlt/artifacts/wall_v3_bolt2_step{20000,30000}`，artifact 分别是 `e0b31aae`、`82d3f732`，都通过 check-only。现在在跑的服务是 30k。**基线。** 30k 加控制柜逆解（`--ik sdk`），开着观测同步和计划等臂，60 s 上限。用户记 10 次 9 次成功。实际跑了 13 次，用户舍弃 3 次后补跑。成功各条的松手中位 29.7 s，示教是 17.5 s。04–10 有两路相机录像（11-13 执行器新加 `tools/wall_recorder.py`）。记录在 `~/wall_exec_logs/bolt2_30k_sdk_baseline_20261113/`。**接下来。** Stage 2 要按新契约重做边界表、交接点、归一化和测试。`wall_rlt_online.py` 写死 weighted 逆解，没有观测同步、等臂和录像，和基线不一致，要用户先定 RL 用哪套控制。细节见 `WALL_RLT_Stage2_Research_Plan.md` §9.2f；交接见 `yuhang_workspace/wall_oss_rlt/AGENT_HANDOFF.md` 第 0 节第一行 |
| **11-11 bolt2 新数据训练包** | 用户重采插螺栓 100 条到 `~/atom_episodes/wall_unlocked_v3/bolt2`（白天、相机未动；旧 bolt 50 条倾向作废）。审计 PASS；逐条检查后用户定剔 24、36（没松手）、57（松手后空合），移到 `wall_unlocked_v3_rejected/bolt2/`（未删）。划分 93/2/2（dev 29、84，test 59、71），审计 `metadata/rlt-wall-v3-bolt2-20261111/`，契约 `configs/rlt/wall_v3_bolt2_ar5_ee_rel6d_next_v1.json` 指纹 `923edc00…`（训练样本 53,067）；`数据集格式说明.md` 加了 bolt2 几行；CPU batch PASS；`check_wall_rlt.py` 417 项通过（`logs/rlt-wall-v3-bolt2-20261111/cpu-check/`）。My Passport：包 `WALL-RLT-training-candidate-20261111-bolt2-v1`（1.9 GB 数据）、清单 `WALL-H100-SHA256SUMS-20261111-bolt2-v1.txt`（9 行，含画图脚本）、操作文档 `WALL-H100-bolt2-训练-20261111.md`。用户定训练 40000 步、每 10000 存，其余超参同上次 bolt（每步 4 样本、lr 2.5e-5 恒定）。在 5070 Ti 用基础镜像演练第 2~4 步 PASS（`~/wall_bolt2_rehearsal`）。训练后画 loss 曲线（`wall_oss_rlt/tools/plot_wall_rlt_training.py`）、dev 对比，由用户选检查点再导出 |
| **11-05 Stage 2 准备** | H100 bolt Stage 1 权重正在拷贝（用户告知），尚未到 5070 Ti。5070 Ti 当天 GPU 因驱动自动升级（595.91→595.99，内核 7.0.0-42）待重启。推理镜像已按本工作区当前源码重建为 `wall-rlt-inference:5070ti-20261105T020149Z`（静态 sm_120 PASS，GPU 未验），工具包 `WALL-RLT-inference5070-20261105-v2`。进入 Stage 2 前的步骤、决策表（RLT 论文参考值＋候选）与 RLinf 在线栈缺口见 [Stage 2 设计决策清单第 0 节](WALL_RLT_Stage2_Design_Decisions.md)；Stage 2 参数仍全部待用户定版 |
| **11-03 晚 v2（以此为准）** | 用户补采 bolt 50 号（bolt 回到 50 条），bolt 划分改为 **46/2/2**（`wall_rlt/data/audit.py` 新增 `--dev-per-task/--test-per-task`，默认行为不变，加了测试）。数据说明更新为 sha256 `c3e4af54…`；契约重生成：bolt 指纹 `f47efc75…`（审计 `metadata/rlt-wall-v2-bolt50-20261103/`）、pick 指纹 `94a55ed2…`（数据与划分不变，只因说明哈希变化）；CPU batch 均 PASS；`check_wall_rlt.py` **401 项通过**（[logs/rlt-wall-v2-20261103b/cpu-check/](logs/rlt-wall-v2-20261103b/cpu-check/)）。My Passport 新包 `WALL-RLT-training-candidate-20261103-{bolt,pick}-v2`（984 MB／1.3 GB），清单 `WALL-H100-SHA256SUMS-20261103-v2.txt`，试解包＋解包后检查 PASS。下面几行的 v1 包、49 条、34/7/8 均已被取代 |
| **11-03 新数据采完** | 5070 Ti `~/atom_episodes/wall_unlocked_v2`：`bolt` 49、`orange_block` 50、`black_bushing` 50，共 149 条／66,215 帧，7 轴全动，全部在 DexGraspVLA- `7cedebf` 下录制；剔除与原因见 [数据集格式说明.md](数据集格式说明.md) 第 1 节。用户定：**bolt 单独训一个模型接 RLT；orange_block＋black_bushing 合训一个，研究 WALL-OSS 泛化** |
| **11-03 数据说明重写** | [数据集格式说明.md](数据集格式说明.md) 按新数据逐条实测重写（sha256 `29772dd8…`，两份契约的 provenance 都绑定这一版）。旧版描述的是作废数据 |
| **11-03 两份 EE 契约** | `configs/rlt/wall_v2_bolt_ar5_ee_rel6d_next_v1.json`（指纹 `41cc302b…`，49 条，34/7/8）、`configs/rlt/wall_v2_pick_ar5_ee_rel6d_next_v1.json`（指纹 `1cf6b145…`，100 条，70/14/16）；审计与 CPU batch 在 `metadata/rlt-wall-v2-{bolt,pick}-20261103/`，均 PASS；无锁定关节 |
| **11-03 check.py 支持 EE** | `wall_rlt/data/check.py` 原来写死 joint8 的对齐比较和 8 维 mask，EE 契约必失败；改为按 `target_mode` 用 scipy 从原始 `ee_pose` 独立算期望行、mask 维数取契约（不在 `MODEL_SOURCES` 内，源码锁 `a6589fb3…` 未变）。`scripts/check_wall_rlt.py` 全套 **400 项通过**：[logs/rlt-wall-v2-20261103/cpu-check/](logs/rlt-wall-v2-20261103/cpu-check/) |
| **11-03 两份训练候选** | My Passport：`WALL-RLT-training-candidate-20261103-bolt-v1`（970 MB）、`WALL-RLT-training-candidate-20261103-pick-v1`（1.3 GB），各含数据、44 个离线 wheel、RLinf 源码、wall-x bundle；格式说明与契约一致；本机试解包＋解包后 CPU batch 均 PASS；校验清单 `WALL-H100-SHA256SUMS-20261103.txt`。状态 `TRAINING_CANDIDATE_NOT_GPU_VALIDATED`。H100 上解包后按解包报告的 `--set` 覆盖项训练，`configs/rlt/stage1*.yaml` 不改 |
| 工作区位置 | 由迁移包 `WALL-OSS-CRP-workspace-20261102` 恢复到 5070 Ti 主机 `~/workspaces/WALL-OSS-CRP`，状态 `RESTORED_CPU_CHECKED`；恢复时 `wall_rlt/`、`tests/rlt/`、`.cache/wall-x-rlt` 与 5070 Ti 已验证镜像的构建源码逐文件一致 |
| 数据 | 解锁关节前采集的 `top_error` 全部作废（10-31 用户决定），5070 Ti 本机和 My Passport 上的原始数据已于 11-02 删除；本工作区 `.cache/rlt-training-portability-20261029-v1/data/top_error` 是迁移包自带的副本，同样作废、仅供恢复检查。新数据用 7 轴全动的笛卡尔遥操重新采集 |
| 动作格式 | 用户决定改用官方 Wall-OSS-0.5 末端相对位姿：新增 `target_mode=ar5_ee_rel6d_next_v1`、`model_adapter=wall_oss05_right_arm_ee_v1`。state＝当前帧法兰位置＋6D 姿态＋实测开度，action＝未来 32 帧相对当前帧的位置／6D 姿态＋**二值夹爪指令**（用户选择），放在预训练右臂槽位 10–19；加载官方权重**不重置**动作投影，只换训练集归一化。相对量算法与 wall-x loader 一致（有测试直接对照）。joint8 保留不变。`prepare_ar5.py` 已含 10-31 解锁补丁，默认生成 EE 契约 |
| 软件检查 | **394 项 CPU 测试通过，0 失败／错误／跳过**（新增 17 项 EE 测试）；源码锁、训练入口预检、合成 Stage 2 均 PASS。报告与源码差异：[logs/rlt-ee-rel6d-20261102/](logs/rlt-ee-rel6d-20261102/)（报告内日志路径指向会话临时目录） |
| 训练候选打包已泛化 | `scripts/package_wall_rlt_training.py` 必须给 `--contract`，契约、manifest、归一化、数据根、划分、排除组、`target_mode`／`model_adapter` 都从契约读；审计目录的 JSON 随契约打包；数据放 `data/<数据根目录名>`（可用 `--dataset-name` 改）；`provenance` 里有 `collection_record_sha256` 时连 `wall_collection.json` 一起打包并核对；数据格式说明按 provenance 收录并记录是否与契约一致。旧 joint8 候选可用 `--contract configs/rlt/top_error_ar5_joint_next_v1.json --require-excluded mouse --require-excluded tape` 复现（已只读核对：1057 个数据文件、`data/top_error`）。记录格式升到 2，旧候选用其自带的解包脚本。解包器按记录的路径放数据、只改写契约 `root`，并在报告里给出训练用的 `--set` 覆盖项。CPU 测试 **400 项通过**（打包测试 28 项）。差异：[logs/rlt-ee-rel6d-20261102/package_generalize.diff](logs/rlt-ee-rel6d-20261102/package_generalize.diff) |
| 训练前还要做（11-03 已完成契约与打包，见上） | 新数据采完后用 `prepare_ar5` 生成 EE 契约（契约与审计目录放在工作区内，如 `configs/rlt/`、`metadata/`），再 `package_wall_rlt_training.py --contract … --include-data --wheelhouse … --output …` 重打训练候选包；`configs/rlt/stage1*.yaml` 不改，训练时用 `train_wall_rlt.py --set data.train_data_paths=<契约> --set actor.model.rlt.data_contract_path=<契约>`（单卡配置另加 `data.val_data_paths`），解包报告里有现成的值；推理侧 `MODEL_SOURCES` 已变（新增 `data/ar5.py`），5070 Ti 推理镜像需用本源码重建 |
| 机器人执行器（11-02） | 在部署目录 `~/yuhang_workspace/wall_oss_rlt/tools/wall_execute.py`（dexgraspvla 环境），只读复用 DexGraspVLA-/SO101 的 atom_robot：默认 `--ik weighted`＝采集同款本地逆解＋SDK 实时关节流（`step_joints` 的全部保护），`--ik sdk`＝控制柜 MoveL 队列（AR5 的 SDK 没有逆解，实时笛卡尔不动）。按拍照时刻对时间轴、跳过过时前缀、50 Hz 插值、夹爪 0.4／0.6 滞回、切段渐变；目标离实测位姿超过 5 cm／20° 原地保持并停。只在模拟臂＋假策略（回放录好的一条）上验证，真机、真相机、真 MoveL 未测；没有训练好的权重 |

## 1. 统一状态口径

| 表述 | 本项目中的含义 |
|---|---|
| 已实现 | 已有可审查代码或脚本；须另行说明测试范围，不等于目标模型已经运行 |
| CPU 检查通过 | 指定报告中的单元、微型模型或合成接口测试通过；不代表完整预训练模型、GPU 或任务效果通过 |
| 候选包就绪 | 已打包并完成相应完整性／迁移检查，可拷贝；不等于目标机部署或最终成品验收完成 |
| 目标机未验收 | 已收录记录为 `NOT_RUN`，尚无该目标机的实际通过报告 |
| 现场验证通过 | 用户确认目标主机已执行并复验相应检查，按明确范围记为通过 |
| 权重缺失 | 尚无本项目训练完成的对应权重，不能用随机权重或官方底座替代 |
| 最终未封版 | 未完成训练后权重及目标 Ubuntu 完整推理验收，不能称 WALL＋RLT 最终成品已交付 |

`PASS`、`READY`、`FAIL` 注明对象和范围。2026-10-29 推理准备记录的 `OFFLINE_SOURCE_CANDIDATE_READY` 与包清单的 `SOURCE_PACKAGE_NOT_GPU_VALIDATED` 保留当时状态；5070 Ti 的最新结果见 [2026-10-31 现场验证记录](metadata/rlt-5070ti-validation-20261031.json)。D0 的 `BUNDLE_READY` 仍只表示基础包完整。

## 2. 工作项与证据

| 工作项 | 当前状态 | 依据与边界 |
|---|---|---|
| 已采集数据接入 | 已适配，CPU 数据检查通过 | `top_error` 中 7 类各 50 条，共 350 条；排除 `mouse`、`tape`，train/dev/test 为 245/49/56。依据 [数据审计](metadata/rlt-top-error-selected-20261029/audit.json) 与 [CPU batch](metadata/rlt-top-error-selected-20261029/cpu-batch.json)；不表示所有轨迹均成功或机器人闭环已验收 |
| Stage 1 软件 | 已实现，CPU 检查通过 | 包括 joint8 动作适配、WALL＋RLT 联合损失、RLinf 注册／worker、训练入口及 checkpoint；真实微型模型验证通过，完整底座训练未验收 |
| Stage 2 软件 | 单步合成接口检查通过 | 固定 RLinf Actor／Critic 的合成 CPU 更新可运行；未用训练后的 Encoder 或真实奖励开展在线学习 |
| 推理软件 | 已实现，CPU 检查通过 | 导出、WALL＋Encoder 加载、HTTP 服务、已有观测回放和服务重启对比入口；CPU 微型模型及协议检查通过，不表示已有完整 WALL＋RLT 推理结果 |
| 最新软件检查 | **377 项通过；0 失败／错误／跳过** | [CPU 汇总报告](logs/rlt-inference-20261029/cpu-check/report.json)；源码锁、训练入口预检及合成 Stage 2 更新也通过 |
| 官方 WALL 权重 | 已有 | 随 D0 基础包交付；原始动作布局为 `32×26`，不等于本项目 AR5 joint8 模型 |
| Stage 1 WALL＋Encoder 权重 | **待训练与导出** | 下一步在 H100 完成 Stage 1；导出不包含训练 Decoder 或 Stage 2 Actor |
| Stage 2 RL Actor 权重 | **待 Stage 2 训练** | 关键实验设计由用户决定，Stage 1 不产生训练后的 RL Actor |
| 5070 Ti 专用镜像 | **已构建，通过** | `wall-rlt-inference:5070ti-20261031T065922Z`；FlashAttention 与 WALL 扩展包含 `sm_120`，用户确认现场复验 |
| 5070 Ti GPU 小算子 | **通过** | BF16 matmul 与 FlashAttention 实跑通过；运行命令补充用户名、可写 HOME 和缓存路径 |
| 5070 Ti 官方 WALL 完整推理 | **通过** | 两个独立容器，合成观测输出均为 `32×26` 且数值有限；allocated/reserved 约 7.94/8.07 GiB。使用核对绑权与归一化张量的受限键兼容检查器，用户确认复验 |
| 5070 Ti 完整 WALL＋RLT 推理 | 待 Stage 1 权重 | 后续验证 AR5 joint8、Encoder 特征、HTTP 和已有真实观测回放 |
| H100 训练与复验 | 记录为 `NOT_RUN` | 完整模型前后向、BF16／FlashAttention／FSDP、100 步短训及真实分片恢复均未验收；上传未记录完成回执 |
| D0 依赖声明 | 仍有 11 项已知问题，`pip check` 为 `FAIL` | 10 项 LeRobot 缺包声明与 1 项 Hub 约束冲突；组件检查通过不能覆盖这些问题 |
| 在线 RL、任务效果、机器人执行 | 未验收 | 当前入口不执行机械臂动作；未获得抓取成功率提升或真实在线学习结果 |
| WALL＋RLT 最终产品 | **未封版** | 候选软件包与最终成品分开管理，目标 Ubuntu 完整推理 PASS 后再冻结完整版本 |

软件和打包历史见 [2026-10-29 推理交付记录](metadata/rlt-inference-deployment-20261029.json)；目标主机结果见 [2026-10-31 现场验证记录](metadata/rlt-5070ti-validation-20261031.json)。官方模型本次首次推理为 2.43 s／1.39 s，用于记录运行情况，不作为稳定控制频率。

## 3. 三份已打包候选的范围

以下目录均位于 `/media/pangk/My Passport/`。数量按目录内交付文件统计，不含旁置说明文件。

| 包 | 文件数／字节数 | 内容与状态 |
|---|---|---|
| `WALL-OSS-CRP-transfer-20261029` | 29／16,787,299,846（约 16.79 GB） | D0 镜像、官方 WALL 权重与 32 条历史验证数据；`BUNDLE_READY`，目标服务器待验 |
| `WALL-RLT-training-candidate-20261029-v1` | 8／5,781,431,598（约 5.78 GB） | 固定训练源码、44 个离线 wheel、350 条目标轨迹；迁移 batch 检查通过，完整训练未验收 |
| `WALL-RLT-inference5070-20261029-v1` | 8／10,979,923（约 10.98 MB） | 6,888 个源文件及构建／服务／回放工具、文档和 CPU 报告；整包校验与实际解包检查通过，不含基础镜像或训练后权重 |

三包合计 **45 个文件、22,579,711,367 字节**。此前的“37 个文件、22,568,731,444 字节”只指 D0＋训练候选两包，仍是这两包的正确统计。D0 的 SHA256 清单覆盖 27 个文件；29 是加上清单自身和完成标记后的目录文件数，两者统计口径不同。

先测试官方 WALL 需要 D0＋推理工具包；之后测试训练后的 WALL＋RLT，还需另行导出训练权重，并使用目标机已有数据或训练候选中的目标数据。旧 `WALL-OSS-CRP-transfer-20261028` 为 `INCOMPLETE` 历史目录，不可部署。

## 4. 主机、数据与交付分工

| 对象 | 已确认信息 | 当前安排 |
|---|---|---|
| Ubuntu 推理主机 | `crp-5070ti-01`；Ubuntu 24.04.4；RTX 5070 Ti，16303 MiB／16GB；驱动 595.91.07；目录 `/home/crp-5070ti-01/yuhang_workspace/wall_oss_rlt` | 用户已完成本地环境构建与官方 WALL 验证；后续用于 WALL＋Encoder 推理，本机不通过 SSH 部署 |
| H100 服务器 | 用户确认资源完备，共 8 张 H100；单卡显存与当前作业分配未记录实测 | 承担训练与后续同版复验；当前双卡 FSDP 候选保持不变，不把总卡数当作已分配卡数 |
| H100 传输 | My Passport → 用户电脑 → 专用网线 → H100；已记录服务器 `10.241.90.194`、网关 `10.241.245.6` | 本机无需连接 H100；Windows 仅中转，完整性检查在 Ubuntu 执行 |
| 数据 | 用户确认采集于后续推理主机和同一机械臂；选定 `top_error`，排除 `mouse`／`tape` | 使用已有数据，不重新采集；D0 历史验证集只用于基础环境检查 |

目标主机的 `nvidia-smi CUDA 13.2` 是驱动显示的兼容能力，不作为已安装 toolkit 的版本记录。候选构建沿用 D0 的 CUDA 12.8／Torch 2.10.0＋cu128，并为 `sm_120` 重编扩展；本轮无驱动变更。目标信息来源为用户提供的输出，配置见 [rtx5070ti.json](configs/inference/rtx5070ti.json)。

## 5. 当前执行顺序

1. **已完成：** 5070 Ti 环境构建、GPU 小算子、官方 WALL 合成观测完整推理及新容器复跑。依据用户现场复验确认，进入下一阶段。
2. **下一步：** 在 H100 使用已有 `top_error` 目标数据完成 Stage 1 训练及恢复，使用已实现的导出入口生成 WALL＋Encoder 推理产物。本机不连接 H100。
3. 在 5070 Ti 主机验证同一训练产物的真实观测往返、参考动作与特征、显存／时延及固定输入重启比较。
4. 通过后冻结代码、环境、权重、配置和验收报告，制作最终 WALL＋RLT 成品包，再按既定计划到 H100 复验同一版本。

Stage 2 在线奖励、真实 replay 和机器人执行不包含在上述 CPU 检查或模型侧服务中，仍需独立实施与验收。

**Stage 2 关键实验设计由用户决定。** reward、各类 warmup、RL 更新、transition、RL chunk 等正式设置尚未定版；现有合成测试值不自动沿用。当前实现限制与待定项见 [Stage 2 设计决策清单](WALL_RLT_Stage2_Design_Decisions.md)，正式交互／训练前先明确数据与时间语义，再落实版本化配置。

## 6. 文档与历史记录的使用规则

- [README](README.md) 与 [实施简版](WALL_RLT_Human_Review.md)：项目入口和概览；当前状态以本文为准。
- [具体实施方案](WALL_RLT_Agent_Implementation_Plan.md)、[详细实现审阅](WALL_RLT_Implementation_Review.md)：代码设计、接口和术语，区分已实现能力与待执行验收。
- [5070 Ti 部署说明](WALL_RLT_Inference_Deployment.md)：当前推理主机的操作入口。[H100 使用说明](H100使用说明.md) 与 [D0 部署说明](docker/OFFLINE-DEPLOY.md) 只用于对应的 H100／基础包流程。
- `stage1_report.md` 的“第一阶段”是 2026-10-26 环境准备阶段，不是 RLT Stage 1 训练验收。旧采集、迁移和主机文档作为标注日期的历史资料，不据其启动已被更正的采集或远程推理路线。
- **147、218、377** 是不同运行阶段的 CPU 检查数量：147 对应 [早期实现快照](metadata/rlt-implementation-20261029.json)，218 对应 [训练准备报告](logs/rlt-h100-preparation-20261029/cpu-check/report.json)，377 对应 [最新推理实施检查](logs/rlt-inference-20261029/cpu-check/report.json)。不得把新数量回填为旧包当时的验证结果。
- 已封包目录、归档内文档、原始日志与验收 JSON 保留原貌。它们记录各自版本的事实；工作区当前文档在状态有新证据时更新，并保留证据链接。修改文档不生成新的模型验收结果，也不自动使旧交付包变成最终产品。

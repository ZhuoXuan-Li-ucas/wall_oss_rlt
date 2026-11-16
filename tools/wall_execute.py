"""WALL+RLT 上真机的执行器：向推理服务要 32 步法兰目标，按时间轴交给 AR5 + OmniPicker 执行。

模型（契约 ar5_ee_rel6d_next_v1）看第 t 帧（腕部相机拍摄时刻 t_obs）的两路图像 + 法兰位姿 + 实测开度，
给出第 t+1 ~ t+32 帧的法兰绝对目标（推理服务已把相对量换算好，`absolute_targets`：位置 m、
旋转向量 rad、夹爪 0/1 指令的连续预测）。第 k 步对应 t_obs + (k+1)/30 s。

**不复制驱动**（工作区约定 AR5 驱动只有 atom_robot 一份，DexGraspVLA- 不改），臂这一侧全部调现成的：

* `--ik weighted`（默认）= 采集同款：`CartesianMapper` 的 weighted 逆解（末端死区、每帧限幅、
  工作空间盒、指尖 tcp_z 都是 collect 构造它时的值；姿态权重默认取 wall_collect.py 固定的采集值，
  config 里的默认 0.15 不用）→ `AR5Env.step_joints`
  （软限位余量、限速、地板检查、驱动的跳变门）→ SDK 实时关节流。采集时的关节构型就是这条路解出来的。
* `--ik sdk` = collect 的 `--ik sdk`：`MoveLStreamer` 排队式 MoveL，控制柜自己解逆解、做插补。
  AR5 的 SDK 没有逆解 / 正解（xMate 模型库不支持 AR5），实时笛卡尔模式收命令不动臂，
  所以「全交给 SDK」只有这一条。collect 里标的是试验模式：10-31 真机跟随慢 240~500 ms、每段 ~90 ms。

回路（50 Hz，和采集控制频率一样）每拍取「现在」在当前计划上的插值（位置线性、姿态球面插值）。
异步，照 LeRobot 的异步推理（async_inference + rtc/ActionQueue）：
* 最新一段只剩 `--queue-threshold` 步时，后台线程拍下一帧去要下一段，推理期间臂接着走这段剩下的；
* 新段的时间轴从观测时刻算，推理期间臂实际走过的那几步跳过；臂没在走的（第一段、旧段先走完了）不跳，
  从收到时刻的第 0 步走起（同 ActionQueue 按实际消耗的步数丢）；
* 新旧两段重叠的那几步里从旧渐变到新（Fade），切换那一刻不跳；两段差太多会打出来；
* RTC（默认开，--no-rtc 关）：要下一段时把当前计划从观测时刻往后的目标（到它走完为止）和推理延迟
  一起发给服务，服务在去噪时让新段的前几步贴住它们、之后线性放开（权重同 LeRobot RTCProcessor；
  WALL-OSS-CRP wall_rlt/inference/runtime.py，挂在 wall-x 的 _execute_flow_action 钩子上）。
  默认 inpaint（每步把前缀拉向加噪的旧段，不额外耗时）；--rtc-method guidance 是 LeRobot 的梯度引导，
  每步多一次反向，5070 Ti 上推理 0.24 → 0.47 s（每步 32 个 token 的前向被 wall-x MoE 的 Python
  逐专家循环和上千次小内核启动拖住，反向同样），11-05 离线前缀偏差 inpaint 0.1 mm、guidance 6.4 mm。
11-05 真机前三条的教训：PNG 编码让延迟到 0.45 s（13 步）；第一段一到就跳 13 步（臂从静止冲 3~4 cm），
第 10 步就要下一段 < 延迟，每段一到就立刻再拍（拍在冲的半路上），前后段差 5~8 cm，来回晃到触发偏离保护。
夹爪：预测值 <0.3 合、>0.7 开（中间保持），越过后持续 0.2 s 才切换（防抖），起始张开，指令发法和采集一样
（`hand.set_opening(0/1)`），夹持力 / 速度默认取任务 dataset.json 里采集时用的。

推理服务协议见 WALL-OSS-CRP 的 wall_rlt/inference/protocol.py：只用它的标准库客户端
（client.py + protocol.py，只依赖 PIL），在 dexgraspvla 环境里直接 import，不需要 torch。
启动时先查 /health，契约不是 ar5_ee_rel6d_next_v1、state 不是 10 维就拒绝；之后每段的
data_contract_hash 都要和 /health 一致。

没有训练好的权重之前，用下面几种方式不碰硬件地验链路：
  --mock-arm                 模拟臂、模拟夹爪、模拟相机（AR5Config.mock）；--ik sdk 时配一个假控制柜
  --fake-policy hold         不要服务，每段都是「原地不动」
  --fake-policy replay:<npz> 不要服务，按时间轴回放一条录好的未来法兰位姿和夹爪指令 = 完美模型

用法（在哪个目录跑都行；dexgraspvla 环境；ROS 污染的 PYTHONPATH 会自动清掉）：
  PY=~/miniforge3/envs/dexgraspvla/bin/python
  $PY tools/wall_execute.py --mock-arm --fake-policy replay:<episode.npz> --autostart --max-seconds 10
  $PY tools/wall_execute.py --mock-arm --ik sdk --fake-policy replay:<episode.npz> --autostart --max-seconds 10
  $PY tools/wall_execute.py --server http://127.0.0.1:8080 --task-dir ~/atom_episodes/wall_unlocked/<任务名>
真机：拿 collect 同一把锁（/tmp/atom_ar5_collect.lock），回车后回 0 位（collect 的 centre_joints）并张开夹爪，
再回车开跑；Ctrl-C 或 --max-seconds 到点就停追加、等臂停稳、关实时流、放锁。
每次运行写一个新目录（--log-dir，已存在就拒绝）：run.json、chunks.jsonl（每段的观测时刻、延迟、跳过几步、
32 个目标、发出的 state）、ticks.npz（每拍的目标、关节指令、实测位姿、夹爪、逆解残差）。
连上设备后默认开一个实时画面窗口（tools/wall_viewer.py，独立进程）：俯视 | 腕部 + 运行时间、段号、步、
推理延迟、夹爪预测 → 指令；关掉窗口不影响执行，--no-view 不开。
"""
import argparse
import datetime
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wall_collect import FIXED_ARGS as COLLECT_ARGS

EE_MODE = "ar5_ee_rel6d_next_v1"
HORIZON = 32
CAMS = ("cam_high", "cam_front")          # 俯视、腕部；和契约 camera_mapping、采集的 key 一样
SYNC_CAM = "cam_front"                    # 采集按腕部相机对齐，t_obs 取它的拍摄时刻
# 夹爪滞回 + 确认：11-10 bolt try_02 在 0.4/0.6 下 1 s 内开合 6 次（预测在 0.5 附近来回），用户要防抖。
# 预测越过 0.3/0.7 并持续 0.2 s 才切换。不用「切换后保持」：try_01 抓着螺栓时有 20~60 ms 的假「开」，
# 保持 0.5 s 会真把螺栓松掉；按两条记录回放，确认 0.2 s 把抖动和假开都滤掉，只留模型持续 0.3 s 以上的意图
CLOSE_BELOW, OPEN_ABOVE, GRIP_CONFIRM_S = 0.3, 0.7, 0.2
JUMP_WARN_M, JUMP_WARN_RAD = 0.02, math.radians(10)
# 失速：实测位姿 3 s 内动不到 0.5 mm、指令却变了 5 mm 以上 = 控制柜把臂停了（碰撞停、急停都会断电，
# 实时通道这边看不出来，11-05 pick orange_block_02 第 18 s 停住、执行器一直发到 60 s）。
# 全部真机记录离线回放过：1 s / 2 s 窗口会把夹紧时 0.1~1.2 s 的停顿误判，3 s 只抓到那一次真停
STALL_S, STALL_MOVE_M, STALL_CMD_M = 3.0, 0.0005, 0.005
# 正式采集的姿态权重（11-02 定 0.5）。执行时夹爪朝向要跟得和采集时一样紧，不用 config 的默认 0.15
COLLECT_ORI_COST = float(COLLECT_ARGS[COLLECT_ARGS.index("--ori-cost") + 1])
ORIG_CWD = Path.cwd()                     # bootstrap 会 chdir 到 SO101，命令行给的相对路径按这里解析


def user_path(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else ORIG_CWD / p


# ───────────────────────────────────────────────────────── 路径和环境

def workspace_root():
    env = os.environ.get("YUHANG_WS")
    if env:
        return Path(env).expanduser().resolve()
    for d in Path(__file__).resolve().parents:
        if (d / "算法").is_dir() and (d / "数据_硬件").is_dir():
            return d
    raise SystemExit("找不到工作区根目录（同时有 算法/ 和 数据_硬件/ 的目录），设环境变量 YUHANG_WS")


def bootstrap():
    """清掉 PYTHONPATH（本机 ~/.bashrc 加载的 ROS 会污染 conda 环境）后重启自己，再接上 SO101 和 wall_rlt。"""
    if os.environ.get("PYTHONPATH"):
        env = dict(os.environ)
        env.pop("PYTHONPATH")
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    so101 = workspace_root() / "算法" / "DexGraspVLA-" / "SO101"
    if not (so101 / "atom_robot" / "ar5" / "ar5_env.py").is_file():
        raise SystemExit(f"✗ 找不到 {so101}/atom_robot")
    wall = Path(os.environ.get("WALL_RLT_ROOT", "~/workspaces/WALL-OSS-CRP")).expanduser().resolve()
    if not (wall / "wall_rlt" / "inference" / "client.py").is_file():
        raise SystemExit(f"✗ 找不到 {wall}/wall_rlt（开发机工作区；用 WALL_RLT_ROOT 指定）")
    sys.path[:0] = [str(so101), str(wall)]
    os.chdir(so101)        # 和 collect 一样在 SO101 下跑：SDK 的 _rokae_log_、相对路径的标定文件都在这儿
    return so101, wall


# ───────────────────────────────────────────────────────── 位姿小工具

def pose6_to_T(p):
    from atom_robot.ar5.transforms import axis_angle_to_matrix
    T = np.eye(4)
    T[:3, :3] = axis_angle_to_matrix(np.asarray(p[3:6], dtype=np.float64))
    T[:3, 3] = p[:3]
    return T


def T_to_pose6(T):
    from atom_robot.ar5.transforms import matrix_to_axis_angle
    return np.concatenate([T[:3, 3], matrix_to_axis_angle(T[:3, :3])])


def slerp_T(A, B, a):
    """A→B 走 a（0~1）：位置线性，姿态沿测地线。"""
    from atom_robot.ar5.transforms import axis_angle_to_matrix, matrix_to_axis_angle
    T = np.eye(4)
    T[:3, :3] = axis_angle_to_matrix(a * matrix_to_axis_angle(B[:3, :3] @ A[:3, :3].T)) @ A[:3, :3]
    T[:3, 3] = (1 - a) * A[:3, 3] + a * B[:3, 3]
    return T


# ── 观测对齐（11-13 用户定）：推理时的观测按采集时的对法对齐，和训练数据一致 ──
# 采集（atom_robot/ar5/sync_align.py）以腕部相机每帧的拍摄时刻为准：臂位姿按时间插值、夹爪开度插值、
# 顶部图取（收到时刻 − SYNC_LATENCY）最近的一张。推理原来取「两路最新的图」再「现读臂」：
# 11-12 实测臂读数比腕部图晚 45 ms（中位，最多 144 ms），顶部图比腕部晚 18 ms（中位），
# 54% 的观测按采集规则本该配另一张顶部图。
ALIGN_EDGE_S = 0.05          # 同 sync_align.EDGE_HOLD_S：要对的时刻离最近的读数在这以内才取端点，再远就不对齐


def cam_latency():
    """采集对齐时每路相机扣掉的延迟（秒）：直接用 collect.SYNC_LATENCY（顶部 335 只有「收到」时刻，10-21 量的
    比拍摄晚 7 ms），和训练数据用同一个数。要在 bootstrap 之后调（那时 SO101 才在 sys.path 里）。"""
    from atom_robot.ar5.collect import SYNC_LATENCY
    return dict(SYNC_LATENCY)


def interp_pose6(hist, t):
    """hist = [(时刻, 位姿6), ...]（时刻递增）在 t 的位姿：位置线性、姿态球面插值（同 sync_align）。
    返回 (位姿6, 方式)；方式 = interp / edge（端点 ALIGN_EDGE_S 以内）/ None（够不着，调用方退回现读）。"""
    if not hist:
        return None, None
    ts = np.array([h[0] for h in hist])
    if t <= ts[0]:
        return (np.asarray(hist[0][1], np.float64), "edge") if ts[0] - t <= ALIGN_EDGE_S else (None, None)
    if t >= ts[-1]:
        return (np.asarray(hist[-1][1], np.float64), "edge") if t - ts[-1] <= ALIGN_EDGE_S else (None, None)
    i = int(np.searchsorted(ts, t, side="right") - 1)
    span = ts[i + 1] - ts[i]
    a = 0.0 if span <= 0 else float((t - ts[i]) / span)
    return T_to_pose6(slerp_T(pose6_to_T(hist[i][1]), pose6_to_T(hist[i + 1][1]), a)), "interp"


def interp_scalar(samples, t):
    """samples = [(时刻, 值), ...] 在 t 的线性插值；够不着（超出两端 ALIGN_EDGE_S）返回 None。"""
    if not samples:
        return None
    ts = np.array([s[0] for s in samples])
    vs = np.array([s[1] for s in samples], dtype=np.float64)
    if t < ts[0] - ALIGN_EDGE_S or t > ts[-1] + ALIGN_EDGE_S:
        return None
    return float(np.interp(t, ts, vs))


def T_gap(A, B):
    from atom_robot.ar5.transforms import matrix_to_axis_angle
    return (float(np.linalg.norm(A[:3, 3] - B[:3, 3])),
            float(np.linalg.norm(matrix_to_axis_angle(A[:3, :3] @ B[:3, :3].T))))


class Chunk:
    """一段 32 步目标。at(t) 给出 t 时刻该在哪：k 步在 t0+(k+1)/fps，t0 到第 0 步之间从观测位姿走过去，
    走过第 32 步（end）就停在最后一个目标上。t0 = 观测时刻 t_obs 往后推「推理期间臂停着的时长」：
    臂一直沿旧计划在走就是 t_obs（走过的几步跳过）；第一段推理期间臂是停着的，t0 = 收到时刻，从第 0 步走起
    （同 LeRobot ActionQueue 按实际消耗的动作数算延迟，队列空着时一步都不丢）。"""

    def __init__(self, cid, t0, obs_T, targets, fps):
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (HORIZON, 7) or not np.isfinite(targets).all():
            raise ValueError(f"目标要是有限的 {HORIZON}x7，给的是 {targets.shape}")
        self.id, self.t0, self.fps = cid, t0, fps
        self.end = t0 + HORIZON / fps
        self.obs_T = obs_T
        self.Ts = [pose6_to_T(row) for row in targets]
        self.grip = targets[:, 6]

    def step(self, t):
        return (t - self.t0) * self.fps - 1.0

    def at(self, t):
        s = self.step(t)
        if s >= HORIZON - 1:
            return self.Ts[-1], float(self.grip[-1]), s
        if s < 0:
            a = max(0.0, s + 1.0)
            return slerp_T(self.obs_T, self.Ts[0], a), float(self.grip[0]), s
        k = int(math.floor(s))
        a = s - k
        return slerp_T(self.Ts[k], self.Ts[k + 1], a), float((1 - a) * self.grip[k] + a * self.grip[k + 1]), s


class Fade:
    """切段：旧计划和新段重叠的那几步上，从旧渐变到新（位姿球面插值、夹爪预测线性），权重在
    [t_start, t_end] 上从 0 线性升到 1，t_end = 旧计划走完的时刻（至少 --blend 秒）。
    LeRobot 异步推理对重叠步是固定加权（weighted_average：0.3 旧 + 0.7 新），切换那一刻会跳 0.7 倍的差；
    这里随时间渐变：接下来马上要走的仍是臂正在走的旧计划，越往后越听新段（RTC 前缀权重线性衰减同一个意思，
    只是 RTC 是在去噪时让新段去贴旧段，这里是在输出上混合，不用改服务）。"""

    def __init__(self, old, new, t_start, t_end):
        self.old, self.new, self.t_start, self.t_end = old, new, t_start, t_end
        self.id, self.end = new.id, new.end

    def step(self, t):
        return self.new.step(t)

    def at(self, t):
        T1, g1, s = self.new.at(t)
        if t >= self.t_end:
            return T1, g1, s
        a = max(0.0, (t - self.t_start) / (self.t_end - self.t_start))
        T0, g0, _ = self.old.at(t)
        return slerp_T(T0, T1, a), (1 - a) * g0 + a * g1, s


class PlanClock:
    """计划时钟 τ：Chunk / Fade 的时间轴都按它走，不按真实时间。当前目标离臂实测位姿在 slow 以内时和真实
    时间同速；超过 slow 线性放慢，到 stop 只按 FLOOR 的速度走，等臂追上再恢复。
    11-12 bolt2_30k trial_02：抓起后模型要 ~300 mm/s、20°/s 抬升转向，臂（关节限速 + 跟踪器加速度）跟不上，
    计划照真实时间往前跑，9.3 s 时目标领先实测 77 mm / 18°，触发 --max-offset-deg 停了；领先的那截还被当
    RTC 前缀发给服务，超出训练的相对动作范围（归一化截到 ±1），新段贴不住旧段（切段跳 46 mm / 12.4°）。
    计划等臂之后，跟不上的那段只是变慢，差距不累积。FLOOR 不为 0：目标真够不着（碰到东西、出工作空间盒）时
    计划仍慢慢往前、照常要新段（新段相对实测位姿），不会卡死在原地。"""

    FLOOR = 0.1

    def __init__(self, t, slow=None, stop=None):
        self.slow, self.stop = slow, stop        # (米, 弧度)；None = 不等臂，和真实时间同速（改之前的行为）
        self.real, self.tau = deque([t - 60.0, t], maxlen=1000), deque([t - 60.0, t], maxlen=1000)
        self.rate = 1.0

    def advance(self, now):
        self.tau.append(self.tau[-1] + self.rate * (now - self.real[-1]))
        self.real.append(now)
        return self.tau[-1]

    def set_rate(self, gap):
        """gap = (当前目标离实测位姿 米, 弧度)；None = 没有计划在走。返回下一拍的速度。"""
        if gap is None or self.slow is None:
            self.rate = 1.0
        else:
            r = min((self.stop[i] - gap[i]) / (self.stop[i] - self.slow[i]) for i in (0, 1))
            self.rate = float(min(1.0, max(self.FLOOR, r)))
        return self.rate

    def at(self, t):
        """真实时刻 t → τ；最后一拍之后按当前速度外推。"""
        if t >= self.real[-1]:
            return self.tau[-1] + self.rate * (t - self.real[-1])
        return float(np.interp(t, np.asarray(self.real), np.asarray(self.tau)))


# ───────────────────────────────────────────────────────── 策略来源

class RemotePolicy:
    """WALL+RLT 推理服务。只用它的标准库客户端，协议校验在客户端也做一遍。"""

    def __init__(self, url, timeout, encoding, rtc):
        from wall_rlt.inference.client import InferenceClient
        self.client, self.encoding = InferenceClient(url, timeout=timeout), encoding
        health = self.client.health()
        md = health["metadata"]
        problems = []
        if md.get("target_mode") != EE_MODE:
            problems.append(f"target_mode={md.get('target_mode')!r}（要 {EE_MODE}）")
        if md.get("state_dim") != 10 or md.get("absolute_target_dim") != 7 or md.get("action_horizon") != HORIZON:
            problems.append(f"state_dim={md.get('state_dim')} absolute_target_dim={md.get('absolute_target_dim')} "
                            f"action_horizon={md.get('action_horizon')}（要 10 / 7 / {HORIZON}）")
        if problems:
            raise SystemExit("✗ 推理服务不是末端相对位姿模型，不执行：" + "；".join(problems))
        if rtc and not md.get("rtc"):
            raise SystemExit("✗ 推理服务不支持 RTC（镜像是 11-05 之前的？）：换新镜像，或加 --no-rtc")
        self.rtc = rtc
        self.contract_hash = md["data_contract_hash"]
        self.describe = {"kind": "remote", "url": url, "server_instance_id": health["server_instance_id"],
                         "rtc": rtc, "metadata": md}

    def start(self, t):
        pass

    def infer(self, obs, raw, rtc=None):
        r = self.client.infer(obs, image_encoding=self.encoding, rtc=rtc if self.rtc else None)
        if r["data_contract_hash"] != self.contract_hash:
            raise RuntimeError("这一段的 data_contract_hash 和 /health 不一样：服务中途换了模型")
        if "absolute_targets" not in r:
            raise RuntimeError("服务没给 absolute_targets")
        return np.asarray(r["absolute_targets"], dtype=np.float64), r.get("timing_seconds")


class HoldPolicy:
    """每段都是「待在观测时的位姿、夹爪保持」。验回路和安全链用。"""

    describe = {"kind": "fake_hold"}

    def start(self, t):
        pass

    def infer(self, obs, raw, rtc=None):
        row = np.concatenate([raw["ee"], [1.0 if raw["gripper_cmd"] >= 0.5 else 0.0]])
        return np.tile(row, (HORIZON, 1)), 0.0


class ReplayPolicy:
    """完美模型：按时间轴回放一条录好的 episode。t_obs 换成帧号 i，返回 ee_pose[i+1..i+32] + 夹爪指令。"""

    def __init__(self, path, fps, latency):
        with np.load(path, allow_pickle=False) as z:
            self.ee = np.asarray(z["ee_pose"], dtype=np.float64)
            self.cmd = np.asarray(z["action"][:, 6], dtype=np.float64)
            self.q0 = np.asarray(z["state"][0, :7], dtype=np.float64)
            self.task = str(z["task"]) if "task" in z.files else ""
            if abs(float(z["fps"]) - fps) > 1e-6:
                raise SystemExit(f"✗ {path} 是 {float(z['fps'])} fps，执行器按 {fps}")
        self.fps, self.latency, self.t0 = fps, latency, None
        self.describe = {"kind": "fake_replay", "npz": str(path), "frames": len(self.ee), "latency_s": latency}

    def start(self, t):
        self.t0 = t
        self.describe["t0"] = t

    def frame(self, t):
        return int(np.clip(round((t - self.t0) * self.fps), 0, len(self.ee) - 1))

    def infer(self, obs, raw, rtc=None):
        i = self.frame(raw.get("tau_obs", raw["t_obs"]))     # 计划时钟：臂落后时回放也跟着等
        idx = np.clip(np.arange(i + 1, i + 1 + HORIZON), 0, len(self.ee) - 1)
        time.sleep(self.latency)           # 模拟推理耗时，让「跳过已经过去的几步」真的发生
        if "first_handover" not in self.describe:
            # 执行器把第一段从收到时刻的第 0 步走起（推理期间臂停着），回放的时钟也从那一刻算，
            # 不然录制的时间轴比臂超前一个推理延迟，之后每段都对不上
            self.t0 = self.describe["first_handover"] = time.perf_counter()
        return np.concatenate([self.ee[idx], self.cmd[idx, None]], axis=1), self.latency


# ───────────────────────────────────────────────────────── 臂的两条路

class WeightedBackend:
    """采集同款：CartesianMapper 的 weighted 逆解 → env.step_joints（实时关节流）。"""

    name = "weighted"

    def __init__(self, env, mapper, hold_posture):
        self.env, self.mapper, self.hold_posture = env, mapper, hold_posture
        self.n = env.config.arm.num_joints
        self.stats = {"box_clamped": 0}

    def engage(self):
        # 和 collect 接合时一样：锚点 = 跟踪器设定点的正运动学，_last_q / _last_target 从这儿起步
        self.mapper.engage(self.hold_posture(self.env.arm, self.n))

    def command(self, T_flange, gripper):
        m = self.mapper
        target = T_flange @ m._tool                      # 法兰目标 → 受控点（指尖）目标
        p = np.clip(target[:3, 3], m._ws_lo, m._ws_hi)   # 工作空间盒，同 CartesianMapper.compute
        if not np.allclose(p, target[:3, 3]):
            self.stats["box_clamped"] += 1
        target[:3, 3] = p
        q_now = self.hold_posture(self.env.arm, self.n)
        # 私有方法：死区 + 每帧限幅 + ik_weighted，和采集每一拍完全同一段代码（compute 去掉主臂那半）
        q = m._solve_weighted(target, q_now)
        self.env.step_joints(q, gripper, observe=False)
        d = m._diag or {}
        return q, d.get("pos_error", np.nan), d.get("rot_error", np.nan)

    def hold(self, gripper):
        self.env.step_joints(self.hold_posture(self.env.arm, self.n), gripper, observe=False)

    def stop(self):
        hold = getattr(self.env.arm, "hold_until_rest", None)
        if hold is not None:
            hold(timeout_s=5.0)

    def report(self):
        return self.mapper.report().rstrip() + f"\n    工作空间盒截过 {self.stats['box_clamped']} 拍"


class SdkBackend:
    """collect 的 --ik sdk：MoveLStreamer 排队式 MoveL，控制柜解逆解。关节空间的保护不起作用，只剩工作空间盒。"""

    name = "sdk"

    def __init__(self, env, mapper, streamer):
        self.env, self.mapper, self.streamer = env, mapper, streamer
        self.stats = {"box_clamped": 0, "movel_errors": 0}
        self._warned = 0.0

    def engage(self):
        pass                                # streamer.step 在臂空闲时自己 start()，从臂当前位姿起步

    def command(self, T_flange, gripper):
        m = self.mapper
        target = T_flange @ m._tool
        p = np.clip(target[:3, 3], m._ws_lo, m._ws_hi)
        if not np.allclose(p, target[:3, 3]):
            self.stats["box_clamped"] += 1
        target[:3, 3] = p
        try:
            self.streamer.step(True, target @ m._tool_inv)
        except RuntimeError as exc:         # 同 collect.step_arm：先停追加，下一拍空闲了再起步
            self.streamer.pause()
            self.stats["movel_errors"] += 1
            now = time.monotonic()
            if now - self._warned > 1.0:
                self._warned = now
                print(f"\n  ⚠ MoveL：{exc}")
        self.env.hand.set_opening(float(np.clip(gripper, 0.0, 1.0)))
        return None, np.nan, np.nan

    def hold(self, gripper):
        self.streamer.step(False, None)
        self.env.hand.set_opening(float(np.clip(gripper, 0.0, 1.0)))

    def stop(self):
        if not self.streamer.halt():
            print("  ⚠ MoveL 队列 8 秒内没停下")

    def report(self):
        return "    " + self.streamer.report() + f"\n    工作空间盒截过 {self.stats['box_clamped']} 拍"


class FakeMoveLController:
    """--mock-arm --ik sdk 用的假控制柜：只实现 MoveLStreamer 调到的 SDK 接口。每个点至少 70 ms、
    按设定速度直线走（真机 10-31：每点 ~64~90 ms，和长短几乎无关），走空了就 idle、要 moveStart 才再走。"""

    MIN_SEGMENT_S = 0.07

    def __init__(self, T0):
        from atom_robot.ar5.transforms import matrix_to_rpy, rpy_to_matrix
        self._rpy, self._from_rpy = matrix_to_rpy, rpy_to_matrix
        self.T, self.queue, self.running = T0.copy(), [], False
        self.lock = threading.Lock()
        self._seg = None
        self.sdk = SimpleNamespace(
            OperationState=SimpleNamespace(idle="idle", moving="moving"),
            CoordinateType=SimpleNamespace(flangeInBase="flangeInBase"),
            MotionControlMode=SimpleNamespace(NrtCommandMode="NrtCommandMode"),
            OperateMode=SimpleNamespace(automatic="automatic"),
            CartesianPosition=lambda: SimpleNamespace(trans=[0.0] * 3, rpy=[0.0] * 3, elbow=0.0,
                                                      hasElbow=False, confData=[0] * 8),
            MoveLCommand=lambda cp: SimpleNamespace(cp=cp, speed=600.0, zone=100.0),
            PyString=str)
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            time.sleep(0.004)
            with self.lock:
                if not self.running:
                    continue
                now = time.monotonic()
                if self._seg is None:
                    if not self.queue:
                        self.running = False
                        continue
                    goal, speed = self.queue.pop(0)
                    dur = max(self.MIN_SEGMENT_S, float(np.linalg.norm(goal[:3, 3] - self.T[:3, 3])) / (speed / 1000.0))
                    self._seg = (self.T.copy(), goal, now, dur)
                A, B, t0, dur = self._seg
                a = min(1.0, (now - t0) / dur)
                self.T = slerp_T(A, B, a)
                if a >= 1.0:
                    self._seg = None

    def close(self):
        self._stop = True

    def pose6(self):
        with self.lock:
            return T_to_pose6(self.T)

    # ── MoveLStreamer 调到的接口：最后一个参数是 ec 字典 ──
    def operationState(self, ec):
        with self.lock:
            return "moving" if (self.running and (self.queue or self._seg)) else "idle"

    def cartPosture(self, coord, ec):
        with self.lock:
            T = self.T.copy()
        return SimpleNamespace(trans=T[:3, 3].tolist(), rpy=self._rpy(T[:3, :3]).tolist(), elbow=0.0,
                               confData=[0] * 8)

    def setMotionControlMode(self, m, ec): pass
    def setOperateMode(self, m, ec): pass
    def setPowerState(self, on, ec): pass
    def powerState(self, ec): return "PowerState.on"
    def toolset(self, ec): return SimpleNamespace(end=SimpleNamespace(trans=[0.0] * 3, rpy=[0.0] * 3))
    def setDefaultConfOpt(self, v, ec): pass
    def setMaxCacheSize(self, v, ec): pass
    def setDefaultSpeed(self, v, ec): pass
    def setDefaultZone(self, v, ec): pass

    def moveReset(self, ec):
        with self.lock:
            self.queue, self._seg, self.running = [], None, False

    def moveAppend(self, cmds, name, ec):
        with self.lock:
            for c in cmds:
                T = np.eye(4)
                T[:3, :3] = self._from_rpy(np.asarray(c.cp.rpy, dtype=np.float64))
                T[:3, 3] = c.cp.trans
                self.queue.append((T, float(c.speed)))

    def moveStart(self, ec):
        with self.lock:
            if self.running and (self.queue or self._seg):
                ec["ec"] = -20
            elif not self.queue:
                ec["ec"] = 768
            else:
                self.running = True


# ───────────────────────────────────────────────────────── 执行

class LiveView:
    """实时画面：俯视 | 腕部 + 一行状态。窗口在 wall_viewer.py 独立进程里（同 DexGraspVLA- deploy_ar5 的
    ViewerFeed），这边一个线程 15 Hz 用 `cameras.peek()` 取最新帧写共享内存 —— peek 不推进 read() 的帧序号、
    不污染 stale_reads（采集预览同理）；窗口卡了、被关了都碰不到发臂指令的回路。"""

    def __init__(self, cameras, log_path):
        import subprocess
        from multiprocessing import shared_memory

        import wall_viewer as wv
        self.cameras, self.wv = cameras, wv
        self.shm = shared_memory.SharedMemory(create=True, size=wv.SIZE)
        self.hdr, self.top, self.wrist = wv.views(self.shm.buf)
        self.hdr[:] = 0
        self.hdr[1] = 1                       # alive
        self.ex = None                        # 开跑后挂上 Executor，状态栏才有段号、延迟、夹爪
        # 新会话：终端里的 Ctrl-C 不传给它（PYTHONPATH 已在 bootstrap 清过）。
        # 输出进记录目录：opencv 自带的 Qt 没字体，每开一次窗口刷几行 QFontDatabase 警告，别占操作员的终端
        with open(log_path, "w", encoding="utf-8") as log:
            self.proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name("wall_viewer.py")),
                                          self.shm.name], stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        self.thread = threading.Thread(target=self.feed, name="wall-live-view", daemon=True)
        self.thread.start()

    def status(self):
        ex = self.ex
        if ex is None or not ex.running or not ex.ticks:
            return (0.0,) + (np.nan,) * 6
        row = ex.ticks[-1]
        return (1.0, row["t"] - ex.t0, row["chunk"], row["s"], ex.last_latency, row["grip_pred"], row["grip_cmd"])

    def feed(self):
        import cv2
        size = (self.wv.W, self.wv.H)
        while self.hdr[1] > 0 and self.proc.poll() is None:      # 窗口关了就不写了
            started = time.monotonic()
            try:
                frames = self.cameras.peek()
                if all(k in frames for k in CAMS):
                    self.hdr[0] += 1                  # 奇数 = 正在写
                    for dst, k in ((self.top, "cam_high"), (self.wrist, "cam_front")):
                        img = frames[k][..., :3]
                        np.copyto(dst, img if img.shape[1::-1] == size else cv2.resize(img, size))
                    self.hdr[2:9] = self.status()
                    self.hdr[0] += 1                  # 偶数 = 写完了
            except Exception as exc:  # noqa: BLE001  显示坏了就不显示，执行照常
                print(f"\n  ⚠ 实时画面停了：{type(exc).__name__}: {exc}", flush=True)
                return
            time.sleep(max(0.0, 1.0 / 15 - (time.monotonic() - started)))

    def close(self):
        self.hdr[1] = 0
        self.thread.join(timeout=1.0)
        try:
            self.proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            self.proc.terminate()
        del self.hdr, self.top, self.wrist    # 先放掉对共享内存的引用，close 才不报 BufferError
        self.shm.close()
        self.shm.unlink()


class Executor:
    def __init__(self, env, backend, policy, args, prompt, log_dir, mapper):
        self.env, self.backend, self.policy, self.args = env, backend, policy, args
        self.prompt, self.log_dir, self.mapper = prompt, log_dir, mapper
        self.fps = args.fps
        self.hw = threading.Lock()           # 臂的 SDK 调用：回路和取观测的线程不同时进
        self.state = threading.Lock()
        self.plan = None                     # 当前在走的：Chunk，或切段渐变中的 Fade
        self.want = threading.Event()        # 回路叫后台线程去要下一段
        self.requested_for = None            # 已经为哪一段发过请求
        self.running = True
        self.gripper_cmd = 1.0               # 起始张开
        self.grip_pending, self.grip_pending_t = None, 0.0   # 待确认的夹爪指令和它开始的时刻
        self.errors = 0
        self.ticks = []
        self.chunk_log = open(log_dir / "chunks.jsonl", "w", encoding="utf-8")
        self.next_id = 0
        self.jumps = 0
        self.t0 = None                       # 开跑时刻；实时画面状态栏用
        self.last_latency = np.nan
        self.latencies = []                  # 每段 t_recv - t_obs，RTC 的推理延迟取它的 p95
        self.stall_win = deque()             # 失速检测：最近 STALL_S 秒的 (时刻, 实测位置, 目标位置)
        # 观测对齐用：回路每拍读的臂位姿 (读的前后中点, 位姿6)，留最近 ~1 s。只有 wall_execute 自己的参数里有
        # obs_align；wall_rlt_online 的 RLExecutor 继承这个类、参数里没有它 → 不对齐，行为不变
        self.obs_align = bool(getattr(args, "obs_align", False))
        self.arm_hist = deque(maxlen=60)
        self.arm_hist_lock = threading.Lock()
        # 计划时钟（PlanClock）。wall_rlt_online 的 RLExecutor 没有 --lag-* 参数、有自己的 run/fetch_once，按真实时间走
        lag = None if getattr(args, "no_lag_wait", True) else (
            (args.lag_slow[0] / 1000, math.radians(args.lag_slow[1])),
            (args.lag_stop[0] / 1000, math.radians(args.lag_stop[1])))
        self.clock = PlanClock(time.perf_counter(), *(lag or (None, None)))
        self.lag_stats = {"slowed_ticks": 0, "min_rate": 1.0, "waited_s": 0.0}

    # ── 观测：和采集记进 npz 的是同一套量 ──
    def observe(self):
        from PIL import Image
        from wall_rlt.data.ar5 import rotvec_to_rot6d
        env = self.env
        if self.args.no_camera:
            images = {k: np.zeros((480, 640, 3), np.uint8) for k in CAMS}
            stamps = {}
        else:
            images = env.cameras.read()
            stamps = dict(getattr(env.cameras, "last_stamps", None) or {})
            missing = [k for k in CAMS if k not in images]
            if missing:
                raise RuntimeError(f"相机少了 {missing}（有的是 {list(images)}）")
        t_cam = stamps.get(SYNC_CAM, time.perf_counter())
        with self.hw:
            t_arm, q, ee = env.read_arm_stamped()
        stamped = getattr(env.hand, "get_opening_stamped", None)
        opening = stamped()[0] if stamped is not None else env.hand.get_opening()
        align = self.align_to(t_cam, t_arm, ee, opening, images, stamps) if self.obs_align else {"mode": "off"}
        ee_obs = align.pop("ee", ee)
        opening_obs = align.pop("opening", opening)
        state = np.concatenate([ee_obs[:3], rotvec_to_rot6d(ee_obs[None, 3:6])[0], [opening_obs]])
        obs = {"instruction": self.prompt, "state": [float(v) for v in state],
               "images": {k: Image.fromarray(np.ascontiguousarray(images[k][..., :3]).astype(np.uint8), "RGB")
                          for k in CAMS}}
        # raw["ee"] / raw["opening"] 是对齐后的（Chunk 的起点、OpenWAM 等包装的状态都从这里取）；现读的另存
        raw = {"t_obs": t_cam, "t_arm": t_arm, "q": q, "ee": ee_obs, "opening": opening_obs,
               "ee_now": ee, "opening_now": opening, "align": align,
               "gripper_cmd": self.gripper_cmd, "state": state}
        return obs, raw

    def align_to(self, t_cam, t_arm, ee, opening, images, stamps):
        """把臂位姿、夹爪开度、顶部图对到腕部图的拍摄时刻 t_cam（同采集 sync_align）。images 原地换顶部图。
        返回记进 chunks.jsonl 的对齐信息，外加 ee / opening 两个对齐后的值（observe 取走）。"""
        info = {"mode": "aligned", "arm_dt_ms": round((t_arm - t_cam) * 1000, 2)}
        with self.arm_hist_lock:
            self.arm_hist.append((t_arm, np.asarray(ee, np.float64)))
            hist = sorted(self.arm_hist, key=lambda h: h[0])
        ee_obs, how = interp_pose6(hist, t_cam)
        if ee_obs is None:                  # 回路还没跑起来（第一段）或者读数断了：退回现读，记下来
            ee_obs, how = np.asarray(ee, np.float64), "now"
            info["mode"] = "partial"
        info["arm"] = how
        info["arm_shift_mm"] = round(float(np.linalg.norm(np.asarray(ee)[:3] - ee_obs[:3])) * 1000, 2)
        info["ee"] = ee_obs

        since = getattr(self.env.hand, "openings_since", None)
        grip = interp_scalar(since(t_cam - 0.3), t_cam) if since is not None else None
        if grip is None:
            grip = opening
            info["mode"] = "partial"
            info["grip"] = "now"
        else:
            info["grip"] = "interp"
        info["grip_shift"] = round(float(opening - grip), 4)
        info["opening"] = grip

        frames_since = getattr(self.env.cameras, "frames_since", None) if not self.args.no_camera else None
        if frames_since is not None:
            lat = cam_latency()
            for name in CAMS:
                if name == SYNC_CAM or name not in stamps:
                    continue
                cands = frames_since(name, None)          # 缓存里最近 8 帧 (序号, 时刻, 帧)
                if not cands:
                    continue
                off = lat.get(name, 0.0)
                seq, st, frame = min(cands, key=lambda c: abs(c[1] - off - t_cam))
                info[f"{name}_latest_dt_ms"] = round((stamps[name] - off - t_cam) * 1000, 2)
                info[f"{name}_dt_ms"] = round((st - off - t_cam) * 1000, 2)
                if abs(st - off - t_cam) > 1.0 / 30 + 1e-3:                    # 缓存里没有挨得上的：照用最新的，记下来
                    info["mode"] = "partial"
                else:
                    images[name] = frame
        return info

    def rtc_prefix(self, t_obs):
        """RTC 前缀：当前计划从观测时刻往后、每步（t_obs+(k+1)/fps）的目标，到计划走完为止；t_obs 是计划时钟上的
        观测时刻。推理延迟取最近延迟的 p95 折成步数（同 LeRobot rollout/inference/rtc.py）。"""
        with self.state:
            plan = self.plan
        if plan is None:
            return None
        steps = min(HORIZON, int(math.floor((plan.end - t_obs) * self.fps)))
        if steps < 1:
            return None
        rows = [plan.at(t_obs + (k + 1) / self.fps) for k in range(steps)]
        delay = int(math.ceil(np.percentile(self.latencies[-20:], 95) * self.fps)) if self.latencies else 0
        return {"prefix_targets": [T_to_pose6(T).tolist() + [float(g)] for T, g, _ in rows],
                "inference_delay": min(delay, steps), "method": self.args.rtc_method}

    def fetch_once(self):
        obs, raw = self.observe()
        with self.state:
            raw["tau_obs"] = tau_obs = self.clock.at(raw["t_obs"])     # 段的时间轴都在计划时钟上
        rtc = self.rtc_prefix(tau_obs) if getattr(self.policy, "rtc", False) else None
        t_send = time.perf_counter()
        targets, timing = self.policy.infer(obs, raw, rtc=rtc)
        t_recv = time.perf_counter()
        self.last_latency = t_recv - raw["t_obs"]
        self.latencies.append(self.last_latency)
        cid = self.next_id
        self.next_id += 1
        jump = None
        with self.state:
            tau_recv = self.clock.at(t_recv)
            old = self.plan
            # 推理期间臂停着的时长：旧计划在观测后就走完了（第一段没有旧计划 = 一直停着）。
            # 新段的时间轴往后推这么久，臂实际没走的那几步不跳（LeRobot ActionQueue 同理按实际消耗的步数丢）
            held = tau_recv - tau_obs if old is None else max(0.0, tau_recv - max(tau_obs, old.end))
            chunk = Chunk(cid, tau_obs + held, pose6_to_T(raw["ee"]), targets, self.fps)
            if old is None:
                self.plan = chunk
            else:
                jump = T_gap(old.at(tau_recv)[0], chunk.at(tau_recv)[0])
                self.plan = Fade(old, chunk, tau_recv, max(tau_recv + self.args.blend, old.end))
        skipped = max(0, int(math.floor(chunk.step(tau_recv))) + 1)
        if old is not None and held > 0:
            print(f"\n  ⚠ 第 {cid} 段来晚了：上一段已走完，臂停了 {held:.2f} s"
                  f"（推理 {t_recv - raw['t_obs']:.2f} s；调大 --queue-threshold 或降延迟）")
        if jump and (jump[0] > JUMP_WARN_M or jump[1] > JUMP_WARN_RAD):
            self.jumps += 1
            print(f"\n  ⚠ 第 {cid} 段和上一段在切换时刻差 {jump[0] * 1000:.0f} mm / {math.degrees(jump[1]):.1f}°"
                  f"（在两段重叠的 {max(self.args.blend, old.end - tau_recv):.2f} s 内渐变过去）")
        rec = {"chunk": cid, "t_obs": raw["t_obs"], "t_arm": raw["t_arm"], "t_send": t_send, "t_recv": t_recv,
               "tau_obs": tau_obs, "tau_recv": tau_recv,
               "latency_s": t_recv - raw["t_obs"], "server_timing_s": timing, "steps_skipped": skipped,
               "held_s": held,
               "rtc_prefix_steps": None if rtc is None else len(rtc["prefix_targets"]),
               "rtc_delay": None if rtc is None else rtc["inference_delay"],
               "switch_gap_m": None if jump is None else jump[0],
               "switch_gap_rad": None if jump is None else jump[1],
               "state": [float(v) for v in raw["state"]], "q_obs": raw["q"].tolist(),
               "align": raw.get("align"),
               "targets": targets.tolist()}
        self.chunk_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.chunk_log.flush()
        return chunk

    def worker(self):
        while self.running:
            if not self.want.wait(timeout=0.1):
                continue
            self.want.clear()
            try:
                self.fetch_once()
                self.errors = 0
            except Exception as exc:          # noqa: BLE001  一段要失败了臂接着走 / 停在当前段末尾，不炸回路
                self.errors += 1
                print(f"\n  ⚠ 要下一段失败（第 {self.errors} 次）：{type(exc).__name__}: {exc}")
                if self.errors >= self.args.max_errors:
                    print("  ✗ 连续失败太多，停")
                    self.running = False
                    return
                with self.state:
                    self.requested_for = None    # 让回路下一拍再叫一次

    def stalled(self, now, ee, T):
        """最近 STALL_S 秒实测动了多少、指令变了多少（米）；够判失速才返回，不然 None。"""
        w = self.stall_win
        w.append((now, ee[:3].copy(), T[:3, 3].copy()))
        while now - w[0][0] > STALL_S:
            w.popleft()
        if now - w[0][0] < 0.95 * STALL_S:
            return None
        measured, commanded = np.array([x[1] for x in w]), np.array([x[2] for x in w])
        moved = float(np.linalg.norm(measured - measured[0], axis=1).max())
        changed = float(np.linalg.norm(commanded - commanded[0], axis=1).max())
        return (moved, changed) if moved < STALL_MOVE_M and changed > STALL_CMD_M else None

    def power_state(self):
        try:
            return str(self.env.arm._call(self.env.arm._robot.powerState))
        except Exception as exc:  # noqa: BLE001  只为报原因，查不到不影响停
            return f"查不到（{type(exc).__name__}）"

    def gripper_from(self, value, now):
        want = 0.0 if value < CLOSE_BELOW else 1.0 if value > OPEN_ABOVE else self.gripper_cmd
        if want == self.gripper_cmd:
            self.grip_pending = None
        elif self.grip_pending != want:
            self.grip_pending, self.grip_pending_t = want, now
        elif now - self.grip_pending_t >= GRIP_CONFIRM_S:
            self.gripper_cmd, self.grip_pending = want, None
        return self.gripper_cmd

    def run(self):
        args = self.args
        period = 1.0 / args.control_hz
        self.worker_thread = threading.Thread(target=self.worker, daemon=True)
        self.worker_thread.start()
        start = self.t0 = time.perf_counter()
        self.policy.start(start)
        with self.hw:
            self.backend.engage()
        self.want.set()
        self.requested_for = -1
        next_t = start
        waited_note = False
        while self.running:
            now = time.perf_counter()
            if args.max_seconds and now - start > args.max_seconds:
                print(f"\n  到 {args.max_seconds:.0f}s，停")
                break
            if getattr(self.env.arm, "latched", False):
                print("\n  ✗ 臂刹车锁死（latch），停。原因看上面有没有 `goals stopped arriving`")
                break
            with self.state:
                tau = self.clock.advance(now)        # 计划时钟：上一拍臂落后多少定这一拍走多快
                if isinstance(self.plan, Fade) and tau >= self.plan.t_end:
                    self.plan = self.plan.new        # 渐变完了，只剩新段（不让 Fade 一层层套下去）
                plan = self.plan
            row = {"t": now, "tau": tau, "rate": np.nan, "chunk": -1, "s": np.nan, "target": np.full(6, np.nan),
                   "q": np.full(7, np.nan), "pos_err": np.nan, "rot_err": np.nan, "grip_pred": np.nan}
            with self.hw:
                t_read = time.perf_counter()
                ee_now = np.asarray(self.env.arm.get_ee_pose(), dtype=np.float64)
                if self.obs_align:                   # 观测对齐要拿图的拍摄时刻前后的读数来插值
                    with self.arm_hist_lock:
                        self.arm_hist.append(((t_read + time.perf_counter()) / 2, ee_now.copy()))
                if plan is None:
                    row["rate"] = self.clock.set_rate(None)
                    self.stall_win.clear()
                    self.backend.hold(self.gripper_cmd)
                    if self.requested_for is None:       # 第一段就没要到：再叫一次
                        self.requested_for = -1
                        self.want.set()
                    if not waited_note and now - start > 2.0:
                        waited_note = True
                        print("  … 还在等第一段目标")
                else:
                    T, g, s = plan.at(tau)
                    # 目标离臂太远就不追：weighted 会按每帧 8 mm 冲过去，sdk 会按 MoveL 600 mm/s 冲过去。
                    # 真模型的目标是相对当前状态的，不会差这么多；回放测试从 0 位起步、或模型输出离谱时才会。
                    # 臂只是跟不上（抓起后快速抬升）由计划时钟等臂，差距停在 --lag-stop 附近，到不了这里
                    gap = T_gap(pose6_to_T(ee_now), T)
                    if gap[0] > args.max_offset_m or gap[1] > math.radians(args.max_offset_deg):
                        print(f"\n  ✗ 目标离臂当前位姿 {gap[0] * 1000:.0f} mm / {math.degrees(gap[1]):.1f}°，超过 "
                              f"--max-offset-m {args.max_offset_m * 1000:.0f} mm / --max-offset-deg "
                              f"{args.max_offset_deg:.0f}°：原地保持并停。回放测试要先把臂摆到那条的首帧姿势")
                        self.backend.hold(self.gripper_cmd)
                        self.running = False
                        break
                    row["rate"] = rate = self.clock.set_rate(gap)
                    if rate < 1.0:
                        self.lag_stats["slowed_ticks"] += 1
                        self.lag_stats["min_rate"] = min(self.lag_stats["min_rate"], rate)
                    stall = None if args.mock_arm else self.stalled(now, ee_now, T)
                    if stall:
                        print(f"\n  ✗ 臂 {STALL_S:.0f} s 只动了 {stall[0] * 1000:.1f} mm、指令却变了 {stall[1] * 1000:.0f} mm："
                              f"控制柜把臂停了（powerState={self.power_state()}；碰撞停或急停）。停。\n"
                              "    急停要手动松开；之后在 SO101 下 PYTHONPATH= python -m atom_robot.ar5.unseize "
                              "--recover --home（臂会回 0 位）")
                        self.running = False
                        break
                    grip = self.gripper_from(g, now)
                    q, pe, re_ = self.backend.command(T, grip)
                    row.update(chunk=plan.id, s=s, target=T_to_pose6(T), pos_err=pe, rot_err=re_, grip_pred=g)
                    if q is not None:
                        row["q"] = np.asarray(q, dtype=np.float64)
                    # 同 LeRobot 异步推理的 chunk_size_threshold：最新一段只剩 --queue-threshold 步就拍下一帧去要，
                    # 推理期间臂接着走这段剩下的，新段回来时和它重叠几步、在重叠里渐变过去
                    if (HORIZON - 1 - s <= args.queue_threshold and self.requested_for != plan.id) \
                            or self.requested_for is None:
                        self.requested_for = plan.id
                        self.want.set()
                row["ee"] = ee_now                  # 本拍下发前读的实测位姿
            row["grip_cmd"] = self.gripper_cmd
            self.ticks.append(row)
            if args.verbose and len(self.ticks) % int(args.control_hz) == 0:
                print(f"\r  段 {row['chunk']:>4}  步 {row['s']:6.1f}  夹爪 {self.gripper_cmd:.0f}  "
                      f"逆解残差 {row['pos_err'] * 1000 if np.isfinite(row['pos_err']) else float('nan'):5.1f} mm",
                      end="", flush=True)
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()        # 卡过就重新对齐，不追拍
        self.running = False

    def save(self):
        # 停的那一刻后台线程可能还在要下一段：等它回来再关 chunks.jsonl，不然它写已关的文件报一条假的「要下一段失败」
        worker = getattr(self, "worker_thread", None)
        if worker is not None:
            worker.join(timeout=self.args.timeout + 1.0)
        self.chunk_log.close()
        with self.state:
            self.lag_stats["waited_s"] = float(self.clock.real[-1] - self.clock.tau[-1])   # 计划比真实时间慢了多少
        if not self.ticks:
            return
        keys = ("t", "tau", "rate", "chunk", "s", "pos_err", "rot_err", "grip_pred", "grip_cmd")
        arrays = {k: np.array([r.get(k, np.nan) for r in self.ticks], dtype=np.float64) for k in keys}
        for k in ("target", "q", "ee"):
            arrays[k] = np.stack([r.get(k, np.full(6, np.nan)) for r in self.ticks]).astype(np.float64)
        np.savez(self.log_dir / "ticks.npz", **arrays)


# ───────────────────────────────────────────────────────── 入口

def load_task(task_dir):
    meta = json.loads((user_path(task_dir) / "dataset.json").read_text(encoding="utf-8"))
    return meta.get("prompt"), meta.get("grip_force"), meta.get("speed")


def main():
    so101, wall = bootstrap()
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.cartesian_mapping import CartesianMapConfig, CartesianMapper
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.end_effector import add_end_effector_args, configure_end_effector
    from atom_robot.ar5.joint_mapping import JointMapConfig
    from atom_robot.ar5.teleop_joint import hold_posture

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--server", help="WALL+RLT 推理服务，如 http://127.0.0.1:8080")
    src.add_argument("--fake-policy", help="不要服务：hold 或 replay:<episode.npz>")
    ap.add_argument("--ik", choices=("weighted", "sdk"), default="weighted",
                    help="weighted = 采集同款本地逆解 + 实时关节流（默认）；sdk = 控制柜 MoveL 队列")
    ap.add_argument("--ori-cost", type=float, default=COLLECT_ORI_COST,
                    help=f"weighted 的姿态权重，默认 {COLLECT_ORI_COST:g} = wall_collect.py 固定的采集值")
    ap.add_argument("--prompt", help="语言指令，要和训练数据逐字一样")
    ap.add_argument("--task-dir", help="采集的任务目录：从它的 dataset.json 取 prompt、夹持力、速度")
    ap.add_argument("--mock-arm", action="store_true", help="模拟臂 / 夹爪 / 相机，不碰硬件")
    ap.add_argument("--no-camera", action="store_true", help="不读相机，发全黑图（只验链路）")
    ap.add_argument("--no-view", action="store_true", help="不开实时画面窗口（俯视 | 腕部 + 状态，独立进程）")
    ap.add_argument("--no-video", action="store_true",
                    help="不录视频（默认开跑起录：cam_high.mp4、cam_front.mp4 + video_frames.npz，格式同采集）")
    ap.add_argument("--fps", type=float, default=30.0, help="模型时间轴（契约 fps），默认 30")
    ap.add_argument("--control-hz", type=float, default=None, help="回路频率，默认用 collect 的 control_hz（50）")
    ap.add_argument("--queue-threshold", type=int, default=20,
                    help="最新一段只剩这么多步就去要下一段（LeRobot chunk_size_threshold / rtc_queue_threshold），默认 20；"
                         "要大于推理延迟的步数（30 Hz：jpeg 约 0.27 s ≈ 8 步，--rtc-method guidance 约 0.5 s ≈ 15 步），"
                         "不然旧段走完了新段还没到，臂会停一下")
    ap.add_argument("--no-rtc", action="store_true",
                    help="不用 RTC：新段不在去噪时贴旧段剩下的，只在重叠里渐变")
    ap.add_argument("--rtc-method", choices=("inpaint", "guidance"), default="inpaint",
                    help="inpaint（默认）：每个去噪步把前缀拉向加噪的旧段，和不开 RTC 一样快；"
                         "guidance：LeRobot 的梯度引导，推理慢一倍，11-05 离线比 inpaint 贴得松")
    ap.add_argument("--blend", type=float, default=0.1,
                    help="切段渐变的最短秒数，默认 0.1；正常是在新旧两段重叠的那几步里渐变完")
    ap.add_argument("--timeout", type=float, default=5.0, help="一次推理请求的超时秒数")
    ap.add_argument("--image-encoding", choices=("png", "jpeg"), default="jpeg",
                    help="传图编码，默认 jpeg：真实画面 PNG 编码俯视 56 ms、腕部 87 ms，jpeg 各约 1 ms"
                         "（训练图像本来就是从 mp4 解出来的，有损）")
    ap.add_argument("--fake-latency", type=float, default=0.23,
                    help="--fake-policy replay 模拟的推理耗时，默认 0.23 s（5070 Ti 实测中位）")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="跑多久自动停，0 = 直到 Ctrl-C")
    ap.add_argument("--max-errors", type=int, default=5, help="连续这么多次要不到下一段就停")
    # 11-05 用户定 5 cm → 8 cm：示范法兰速度 p99 280~344 mm/s，臂追目标滞后 0.1~0.15 s，正常就差 40~50 mm
    # （pick 实验 36 条每条最大差中位 45 mm，unseen 02 到 51 mm 被误停）；臂的实际速度由每拍 8 mm 和关节限速管。
    # 11-10 用户定 8 cm → 12 cm：bolt try_02 抓起后快速抬升，关节限速（0.0137 rad/拍）顶了 31 拍，落后 80 mm 被误停
    ap.add_argument("--max-offset-m", type=float, default=0.12,
                    help="当前目标离臂实测位姿超过这么远（米）就原地保持并停，默认 0.12")
    ap.add_argument("--max-offset-deg", type=float, default=20.0,
                    help="当前目标离臂实测姿态超过这么多度就原地保持并停")
    # 11-12 加：bolt2_30k trial_02 抓起后快速抬升，计划照真实时间跑在臂前面，9.3 s 差到 18° 被上面的保护停了。
    # 默认值按 trial_01（成功、全程跟得上）的实测差距定：p90 10 mm / 2.2°、p99 32 mm / 4.8°
    ap.add_argument("--lag-slow", type=float, nargs=2, default=(15.0, 3.0), metavar=("MM", "DEG"),
                    help="当前目标领先臂实测位姿超过这么多（mm 或 °，任一）就放慢计划时钟，默认 15 3")
    ap.add_argument("--lag-stop", type=float, nargs=2, default=(40.0, 8.0), metavar=("MM", "DEG"),
                    help=f"领先到这么多计划只按 {PlanClock.FLOOR:g} 倍速走、等臂追上，默认 40 8")
    ap.add_argument("--no-obs-align", action="store_true",
                    help="观测不对齐：两路最新的图 + 现读的臂（11-13 之前的行为）。默认按采集的对法对到腕部图的拍摄时刻")
    ap.add_argument("--no-lag-wait", action="store_true",
                    help="计划时钟不等臂，按真实时间走（11-12 之前的行为）")
    ap.add_argument("--deadband-mm", type=float, default=1.0, help="末端死区（位置），默认 1 mm（collect 遥操是 3 mm）")
    ap.add_argument("--deadband-deg", type=float, default=0.5, help="末端死区（姿态），默认 0.5°（collect 遥操是 1.7°）")
    ap.add_argument("--autostart", action="store_true", help="不等回车（回 0 位、开跑都不问）")
    ap.add_argument("--no-home", action="store_true", help="开跑前不回 0 位")
    ap.add_argument("--log-dir", type=Path, default=None, help="本次记录目录（默认 ~/wall_exec_logs/<时刻>_<ik>）")
    ap.add_argument("--verbose", action="store_true", help="每秒打一行状态")
    add_end_effector_args(ap)
    args = ap.parse_args()
    args.obs_align = not args.no_obs_align
    args.end_effector = "omnipicker"                  # 采集固定 OmniPicker
    if not 1 <= args.queue_threshold < HORIZON:
        raise SystemExit(f"✗ --queue-threshold 要在 1~{HORIZON - 1}")
    if not all(0 <= a < b for a, b in zip(args.lag_slow, args.lag_stop)):
        raise SystemExit("✗ --lag-slow 的两个数都要小于 --lag-stop 对应的数")

    # 语言指令、夹持力、速度：采集时的 dataset.json 优先，命令行可覆盖
    prompt, force, speed = (load_task(args.task_dir) if args.task_dir else (None, None, None))
    policy_note = None
    if args.fake_policy:
        kind, _, arg = args.fake_policy.partition(":")
        if kind == "hold":
            policy = HoldPolicy()
        elif kind == "replay" and arg:
            policy = ReplayPolicy(user_path(arg), args.fps, args.fake_latency)
            prompt = prompt or policy.task
            policy_note = arg
        else:
            raise SystemExit("✗ --fake-policy 只能是 hold 或 replay:<episode.npz>")
    prompt = args.prompt or prompt
    if not prompt:
        raise SystemExit("✗ 要语言指令：--prompt 或 --task-dir（和训练数据逐字一样）")
    if args.omnipicker_force is None and force is not None:
        args.omnipicker_force = int(force)
    if args.omnipicker_speed is None and speed is not None:
        args.omnipicker_speed = int(speed)

    log_dir = (user_path(args.log_dir) if args.log_dir else Path.home() / "wall_exec_logs" /
               f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{args.ik}{'_mock' if args.mock_arm else ''}")
    if log_dir.exists():
        raise SystemExit(f"✗ {log_dir} 已存在，换一个（不覆盖旧记录）")

    if args.server:
        policy = RemotePolicy(args.server, args.timeout, args.image_encoding, rtc=not args.no_rtc)

    config = AR5Config()
    collect_cfg = config.collect
    collect_cfg.space, collect_cfg.cart_ik = "cartesian", args.ik
    collect_cfg.__post_init__()
    control_hz = args.control_hz or collect_cfg.control_hz
    args.control_hz = control_hz
    config.mock = args.mock_arm
    config.use_hand = True
    configure_end_effector(config, args)
    config.arm.use_realtime = args.ik != "sdk"         # 同 collect：sdk 整场不开实时流
    if args.no_camera:
        config.cameras.sources = {}
    elif set(CAMS) - set(config.cameras.sources):
        raise SystemExit(f"✗ config 里的相机 {list(config.cameras.sources)} 缺 {CAMS}")

    # 映射器两条路都要：0 位（centre_joints）、正运动学、指尖偏移、工作空间盒，都按采集时的构造
    cart_cfg = CartesianMapConfig(position_scale=collect_cfg.scale, ik=args.ik,
                                  tcp_z=collect_cfg.cartesian_tcp_z(),
                                  rotation_scale=collect_cfg.cartesian_rot_scale(),
                                  centre_joints=collect_cfg.cartesian_centre_joints)
    cart_cfg.orientation_cost = args.ori_cost
    # 末端死区：每拍目标变化低于它就不更新。collect 的 3 mm / 1.7° 是给主臂遥操去抖的，
    # 执行策略时会把慢速段变成 3 mm 一级的台阶、插孔分辨率只有 3 mm；11-10 用户定 1 mm / 0.5°
    cart_cfg.deadband_m, cart_cfg.deadband_rad = args.deadband_mm / 1000, math.radians(args.deadband_deg)
    mapper = CartesianMapper(config.arm, config.arm.workspace_min, config.arm.workspace_max, config=cart_cfg,
                             joint_cfg=JointMapConfig.load(**collect_cfg.mapping_overrides))
    # mapper.configure() 里和逆解有关的那半（cartesian_mapping.py configure）；另一半是主臂行程，执行时没有主臂
    lo = np.asarray(config.arm.joint_min, dtype=np.float64) + config.arm.joint_limit_margin
    hi = np.asarray(config.arm.joint_max, dtype=np.float64) - config.arm.joint_limit_margin
    mapper._kin.joint_min, mapper._kin.joint_max = lo, hi
    mapper._kin.joint_mid = (lo + hi) / 2.0
    mapper._kin.joint_range = np.maximum(hi - lo, 1e-9)
    mapper._posture = np.clip(mapper.centre, lo, hi)

    lock = None
    if not args.mock_arm:
        from atom_robot.ar5.collect import acquire_single_instance_lock
        lock = acquire_single_instance_lock()
        if lock is None:
            raise SystemExit("✗ /tmp/atom_ar5_collect.lock 被占着：collect 或另一个执行器还开着，同一台臂不能两边驱动")
        from atom_robot.ar5.unseize import diagnose, power_back_on
        st = diagnose(config.arm, quiet=True)
        if st["physical_stop"]:
            raise SystemExit(f"✗ {st['why']}（急停 / 安全门是物理的，软件清不掉）")
        if st["reachable"] and (st["seized"] or st.get("idle_off")):
            print(f"  臂现在起不来：{st['why']}；先自动上电（只使能伺服，臂不动）")
            if power_back_on() != 0:
                raise SystemExit("✗ 上电没成功")

    log_dir.mkdir(parents=True)
    env = AR5Env(config)
    env.connect()
    fake_ctl = view = None
    try:
        if not (args.no_view or args.no_camera or not os.environ.get("DISPLAY")):
            view = LiveView(env.cameras, log_dir / "viewer.log")   # 回 0 位之前就开：回车前先看一眼两路画面对不对
            print("  实时画面窗口已打开（独立进程；关掉窗口不影响执行）")
        env.arm.clear_latch()
        start_q = mapper.centre.copy()
        if isinstance(policy, ReplayPolicy) and args.mock_arm:
            start_q = policy.q0.copy()                   # 模拟时从那条的首帧姿势起步，残差只反映逆解和时间轴
        if args.mock_arm:
            # 模拟臂原本只按关节更新位置、不更新姿态；换成正运动学，观测和逆解用同一套运动学
            arm = env.arm
            arm.kinematics = mapper._kin
            arm._joints = start_q.copy()
            with arm._setpoint_lock:
                arm._q_cmd, arm._q_goal = start_q.copy(), start_q.copy()
            arm.get_ee_pose = lambda: T_to_pose6(mapper._kin.fk(arm._joints))
        if args.ik == "sdk":
            from atom_robot.ar5.movel_stream import MoveLStreamer
            if args.mock_arm:
                fake_ctl = FakeMoveLController(mapper._kin.fk(start_q))
                robot, sdk = fake_ctl, fake_ctl.sdk
                env.arm.get_ee_pose = fake_ctl.pose6
            else:
                robot, sdk = env.arm._robot, env.arm.sdk
            streamer = MoveLStreamer(robot, sdk, speed=collect_cfg.sdk_speed, zone=collect_cfg.sdk_zone,
                                     elbow=collect_cfg.sdk_elbow, min_step_mm=collect_cfg.sdk_deadband_mm,
                                     min_rotation_deg=collect_cfg.sdk_rot_deadband_deg,
                                     backlog_ms=collect_cfg.sdk_backlog_ms, batch=collect_cfg.sdk_batch,
                                     merge_mm=collect_cfg.sdk_merge_mm)
            if not streamer.configure():
                raise SystemExit("✗ MoveL 排队模式没配起来（臂没动过）")
            backend = SdkBackend(env, mapper, streamer)
            print("  ⚠ --ik sdk：控制柜解逆解，跳变门 / 限速 / 地板检查不起作用，只剩工作空间盒和控制柜自己的保护")
        else:
            if not args.mock_arm and not getattr(env.arm, "_rt_active", False):
                raise SystemExit("✗ 臂的实时通道没起来（python -m atom_robot.ar5.unseize 看一眼）")
            backend = WeightedBackend(env, mapper, hold_posture)

        run = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
               "ik": args.ik, "mock_arm": args.mock_arm, "prompt": prompt, "fps": args.fps,
               "control_hz": control_hz, "queue_threshold": args.queue_threshold, "blend_s": args.blend,
               "image_encoding": args.image_encoding,
               "rtc_method": None if args.no_rtc or not args.server else args.rtc_method,
               "max_offset_m": args.max_offset_m, "max_offset_deg": args.max_offset_deg,
               "obs_align": None if not args.obs_align else {"master": SYNC_CAM, "cam_latency_s": cam_latency(),
                                                             "edge_s": ALIGN_EDGE_S},
               "lag_wait": None if args.no_lag_wait else {"slow_mm_deg": list(args.lag_slow),
                                                          "stop_mm_deg": list(args.lag_stop),
                                                          "floor_rate": PlanClock.FLOOR},
               "deadband_mm": args.deadband_mm, "deadband_deg": args.deadband_deg,
               "gripper": {"close_below": CLOSE_BELOW, "open_above": OPEN_ABOVE, "confirm_s": GRIP_CONFIRM_S},
               "policy": policy.describe, "so101": str(so101), "wall_rlt_root": str(wall),
               "tcp_z": cart_cfg.tcp_z, "ori_cost": cart_cfg.orientation_cost, "workspace": [list(config.arm.workspace_min), list(config.arm.workspace_max)],
               "omnipicker": {"grip_force": getattr(config.omnipicker, "grip_force", None),
                              "speed": getattr(config.omnipicker, "speed", None)},
               "start_q": start_q.tolist()}
        (log_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")

        print(f"\n  WALL 执行器：{args.ik}，{'模拟臂' if args.mock_arm else '真臂'}，"
              f"策略 {policy.describe['kind']}{'（' + policy_note + '）' if policy_note else ''}")
        print(f"  指令：{prompt}\n  记录：{log_dir}")
        if not args.no_home:
            if not args.autostart:
                input("\n  回车：臂回 0 位（collect 的 centre_joints）并张开夹爪 —— **臂会动**，Ctrl-C 取消 ")
            if args.ik == "sdk" and not args.mock_arm:
                streamer.halt()
            ok = env.goto_joints(start_q, timeout_s=20.0)
            print("  已到起始姿势" if ok is not False else "  ⚠ 20 s 内没到起始姿势，自己看一眼")
        env.hand.set_opening(1.0)
        if not args.autostart:
            input("  回车开跑（Ctrl-C 停） ")
        ex = Executor(env, backend, policy, args, prompt, log_dir, mapper)
        if view is not None:
            view.ex = ex
        rec = None
        if not (args.no_video or args.no_camera):
            from wall_recorder import Recorder    # 独立线程，不碰回路；写坏了只停录像
            rec = Recorder(env.cameras, log_dir, CAMS, fps=config.cameras.fps, codec=collect_cfg.video_codec).start()
        try:
            ex.run()
        except KeyboardInterrupt:
            print("\n  Ctrl-C，停")
        finally:
            ex.running = False
            with ex.hw:
                backend.hold(ex.gripper_cmd)
                backend.stop()
            ex.save()
            if rec is not None:
                run["video"] = rec.stop()
            run.update(policy=policy.describe, ended=datetime.datetime.now().isoformat(timespec="seconds"),
                       chunks=ex.next_id, switch_jump_warnings=ex.jumps, lag_stats=dict(ex.lag_stats),
                       env_stats=dict(env.stats),
                       backend_stats=dict(backend.stats))
            (log_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
            print("\n" + backend.report())
            if hasattr(env.arm, "rt_stats"):
                print(f"    实时流：{env.arm.rt_stats()}")
            print(f"    env 计数：{env.stats}")
            if not args.no_lag_wait:
                ls = ex.lag_stats
                print(f"    臂跟不上时计划放慢 {ls['slowed_ticks']} 拍（最慢 {ls['min_rate']:.2f} 倍速），"
                      f"整场比真实时间慢 {ls['waited_s']:.2f} s")
            if rec is not None:
                print("    录像：" + "，".join(f"{k} {c['frames']} 帧（丢 {c['dropped']}，{c['fps_actual']} fps）"
                                           for k, c in run["video"]["cams"].items()))
            print(f"    段 {ex.next_id}，切段跳变告警 {ex.jumps} 次；记录在 {log_dir}")
    finally:
        if view is not None:
            view.close()
        if fake_ctl is not None:
            fake_ctl.close()
        env.close()
        if lock is not None:
            lock.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:          # 回车提示时按 Ctrl-C：臂已在 finally 里关好、锁已放
        print("\n  取消了")
        sys.exit(130)

"""WALL-RLT Stage 2 真机在线训练 / 评估：WALL 抓取搬运 → 到套筒附近自动交接 → actor 控制 → 松手 → 打标签 → 更新。

决策（WALL-OSS-CRP/WALL_RLT_Stage2_Research_Plan.md 第 7 节，2026-11-10 用户定）：
* D5/D15：从拿螺栓开始；每次向 WALL 请求时检查，法兰离示范松手点 < 3 cm 且实测开度 0.12~0.30（夹着螺栓）
  就交接给 actor，之后不切回；RL 阶段最长 20 s（截断），WALL 阶段 30 s 没交接 = WALL 阶段失败（不入库）。
* D1：C=12，固定执行起始步 D=11。交接后每 C 步拍一次（--queue-threshold 自动设为 32-C），actor 的块
  = 这次窗口的第 11~22 步；中间 11 步是上一块的尾巴，作为 RTC 前缀（全部硬约束）发给 WALL；
  新块在第 D 步之前切换完（不像 WALL 阶段那样渐变到旧段走完），actor 的行才是真正执行的。
* D6：actor 自带噪声为 0；动作空间层按物理单位加探索（每块一个线性增长的偏移，2 mm / 0.5°）。
* D14：夹爪交给 actor，不受限幅；松手 = 人工按键 o 为主（人工干预）+ 每块 0.08 随机试松手；
  warmup 期间只开松手探索、不加位姿噪声。
* D16：回车开始；o 人工松手；s 成功；f 失败；r 丢弃；Ctrl-C 急停；空格不做任何事（采集里空格是夹紧）。
* D9/D13：warmup、critic-only、batch 等在配置里；训练 200 条，第 50/100/150/200 条后评估（--mode eval）。

一条 episode：回车 → 臂回 0 位、夹爪张开 → WALL（异步 + RTC，同 wall_execute.py）→ 交接 → actor →
松手（actor / 随机 / o 键）→ RL 立即停在这一步、臂停 0.5 s 后上抬 5 cm → 按 s / f → 入库 + 更新 → 回 0 位。
截断：当前块走完后上抬，等回车（先扶住螺栓，回车后夹爪张开）。WALL 阶段失败、r 丢弃：原地保持，同样等回车。

复用 wall_execute.py 的执行部件（Chunk / Fade / 逆解后端 / 偏离保护 / 失速检测 / 实时画面），不改它。
配置是 WALL-OSS-CRP/configs/rlt_online/*.yaml；有未决（null）的参数就拒绝运行。启动时核对：推理服务的
data_contract_hash 必须等于边界表所用契约的指纹（边界、交接点、归一化都来自那份数据）。

用法（dexgraspvla 环境；先开 bolt 推理服务）：
  PY=~/miniforge3/envs/dexgraspvla/bin/python
  $PY tools/wall_rlt_online.py --server http://127.0.0.1:8080 --task-dir ~/atom_episodes/wall_unlocked_v2/bolt \\
      --config <填好的 yaml> --run-dir ~/wall_rlt_runs/<名字> --rl-deadband-mm <mm> --rl-deadband-deg <度>
  续训：同一 --run-dir 加 --resume；评估：--mode eval --policy-from <run-dir>/trainer/latest_online_state.pt --episodes 20
  模拟臂干跑：--mock-arm --mock-handover-after 3 --mock-keys "o@2" --mock-episodes 3 --autostart
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wall_execute as wx  # noqa: E402

ENTER_KEYS = ("\n", "\r")
LIFT_M, LIFT_WAIT_S, LIFT_S = 0.05, 0.5, 1.5


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ───────────────────────────────────────────────────────── 会话：模型、训练器、动作空间

class Session:
    def __init__(self, args, wall_root, remote):
        import torch
        import yaml
        from rlt_online import evo
        from rlt_online.action_space import ActionBounds, ChunkActionCodec, ProprioNormalizer
        from rlt_online.collector import attach_chunk_collector
        from rlt_online.episode import ProtocolConfig
        from rlt_online.exploration import ChunkExplorer, ExplorationConfig
        from rlt_online.policy import WallChunkACPolicy, load_configs
        from wall_rlt.data.contract import DataContract

        self.torch, self.evo, self.mode = torch, evo, args.mode
        run_dir = args.run_dir
        payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        payload["online"]["save_dir"] = str(run_dir / "trainer")
        resolved = run_dir / "config.resolved.yaml"
        if resolved.exists() and args.mode == "train" and args.resume:
            old = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            if old != payload:
                raise SystemExit(f"✗ --resume 但配置和这个 run 当初的不一样（{resolved}）：续训必须同一套配置")
        resolved.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        try:
            self.policy_cfg, self.online_cfg, self.meta = load_configs(resolved)
        except ValueError as exc:
            raise SystemExit(f"✗ 配置还有没定的参数，不能跑：{exc}")
        meta = self.meta
        self.C, self.D = self.policy_cfg.chunk_length, int(meta["execution_offset"])
        if float(meta["fps"]) != args.fps:
            raise SystemExit(f"✗ 配置 fps {meta['fps']} ≠ 执行器 {args.fps}")

        def wall_path(p):
            p = Path(p)
            return p if p.is_absolute() else wall_root / p
        bounds = ActionBounds.from_json(wall_path(meta["action_bounds"]))
        contract = DataContract.load(bounds.provenance["contract"])
        if contract.fingerprint() != remote.contract_hash:
            raise SystemExit(f"✗ 推理服务的 data_contract_hash {remote.contract_hash[:12]}… ≠ 边界表所用契约 "
                             f"{contract.fingerprint()[:12]}…（{bounds.provenance['contract']}）：模型和数据派生的参数不配套")
        self.codec = ChunkActionCodec(bounds, self.C, self.D)
        self.proprio = ProprioNormalizer(wall_path(meta["normalization"]))
        self.rng_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
        self.explorer = ChunkExplorer(ExplorationConfig.from_meta(meta), self.C, np.random.default_rng(self.rng_seed))
        self.proto_cfg = ProtocolConfig.from_meta(meta)
        self.keys = self.proto_cfg.keys
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = WallChunkACPolicy(self.policy_cfg).to(self.device)
        self.trainer = self.collector = None
        self.stored = 0
        if args.mode == "train":
            self.trainer = evo.OnlineRLTrainer(self.policy, self.online_cfg, self.policy_cfg)
            self.collector = attach_chunk_collector(self.trainer)
            if args.resume:
                self.stored = self.trainer.load_latest_state(run_dir / "trainer" / "latest_online_state.pt")
        elif args.mode == "eval":
            self.load_weights(Path(args.policy_from))
        self.policy.eval()
        self.provenance = {"config": str(Path(args.config).resolve()), "config_sha256": sha256_file(args.config),
                           "bounds_sha256": sha256_file(wall_path(meta["action_bounds"])),
                           "contract_fingerprint": contract.fingerprint(), "seed": self.rng_seed,
                           "C": self.C, "D": self.D, "device": str(self.device)}

    def load_weights(self, path):
        torch = self.torch
        if path.is_file():                        # latest_online_state.pt
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.policy.actor.load_state_dict(state["actor_state_dict"])
            self.policy.critic.load_state_dict(state["critic_state_dict"])
        else:                                     # step_NNNNNN/ (save_pretrained)
            state = torch.load(path / "actor_critic.pt", map_location="cpu", weights_only=True)
            self.policy.actor.load_state_dict(state["actor"])
            self.policy.critic.load_state_dict(state["critic"])
        self.policy.to(self.device)

    def explore_mode(self):
        if self.mode != "train":
            return "eval"
        return "warmup" if self.trainer.warmup_completed_at_episode is None else "train"


# ───────────────────────────────────────────────────────── 一条 episode 的执行器

class RLExecutor(wx.Executor):
    def __init__(self, env, backend, remote, args, prompt, log_dir, mapper, sess, protocol, kb, deadbands):
        super().__init__(env, backend, remote, args, prompt, log_dir, mapper)
        from rlt_online.rollout import ChunkLedger
        self.sess, self.protocol, self.kb, self.deadbands = sess, protocol, kb, deadbands
        self.ledger = ChunkLedger(sess.collector, sess.C, args.fps) if sess.collector is not None else None
        self.human_open = False
        self.release = None                   # (时刻, 来源)
        self.post = None                      # None | released | lifted | truncating | hold_bolt
        self.lift = None
        self.hold_done = False
        self.rl_deadband = False
        self.last_rl = None                   # 最近一个 RL 块的记录（判断松手来源）
        self.prev_rl_obs = None               # 上一次 RL 观测的时刻（算观测间隔几帧）
        self.first_rl_start = None            # 第一块 actor 开始执行的时刻
        self.counts = {"rl_chunks": 0, "late_chunks": 0, "ref_clipped": 0, "exec_clipped": 0, "random_release_chunks": 0}
        self.frames = log_dir / "frames" if args.save_frames else None
        if self.frames:
            self.frames.mkdir()
        self.mock_keys = list(args.mock_keys or [])
        self.messages = []
        # 给操作员看的：离松手点多远、开度多少；没交接时说明原因
        self.ref_point = np.asarray(sess.proto_cfg.reference_point_m, dtype=np.float64)
        self.last_obs = None                  # (时刻, 离参考点 m, 实测开度)
        self.closest = (np.inf, None)         # WALL 阶段最近一次到参考点的距离和当时开度
        self.warned_not_holding = False
        self.tty = sys.stdout.isatty()
        self.status_t = 0.0

    # ── 推理：WALL 阶段同 wall_execute；RL 阶段带 z_rl、D 步硬前缀，再把 actor 的块拼进第 D~D+C-1 步 ──
    def rl_prefix(self, t_obs):
        with self.state:
            plan = self.plan
        if plan is None:
            return None
        steps = min(self.sess.D, int(math.floor((plan.end - t_obs) * self.fps)))
        if steps < 1:
            return None
        rows = [plan.at(t_obs + (k + 1) / self.fps) for k in range(steps)]
        return {"prefix_targets": [wx.T_to_pose6(T).tolist() + [float(g)] for T, g, _ in rows],
                "inference_delay": steps, "method": self.args.rtc_method}

    def actor_chunk(self, r, raw, obs, cid):
        sess, torch = self.sess, self.sess.torch
        C, D = sess.C, sess.D
        rel = np.asarray(r["actions"], dtype=np.float64)
        ref_rel = rel[D:D + C]
        ref_norm, ref_clipped = sess.codec.encode(ref_rel)
        state_vec = np.concatenate([np.asarray(r["z_rl"], dtype=np.float32),
                                    sess.proprio(raw["state"]).astype(np.float32)])
        mode = sess.explore_mode()
        if self.args.mode == "vla-only":
            exec_rel, info = ref_rel.copy(), {"mode": "vla-only", "random_release": False}
        else:
            with torch.no_grad():
                mean = sess.policy.act(torch.from_numpy(state_vec)[None].to(sess.device),
                                       torch.from_numpy(ref_norm.astype(np.float32))[None].to(sess.device),
                                       explore=False)[0].cpu().numpy().astype(np.float64)
            exec_rel, info = sess.explorer(sess.codec.decode(mean), mode=mode)
            info["mean_dev_mm"] = float(np.abs(sess.codec.decode(mean)[:, :3] - ref_rel[:, :3]).max() * 1000)
        if self.human_open:
            exec_rel[:, 9] = 1.0
        exec_norm, exec_clipped = sess.codec.encode(exec_rel)
        targets = np.asarray(r["absolute_targets"], dtype=np.float64).copy()
        targets[D:D + C] = sess.codec.to_absolute(exec_rel, raw["state"])
        targets[D + C:] = targets[D + C - 1]      # 块尾之后停在 actor 最后一步，空档里不跑 WALL 的预测
        if self.frames is not None:
            for k, img in obs["images"].items():
                img.save(self.frames / f"{cid:04d}_{k}.jpg", quality=90)
        self.counts["rl_chunks"] += 1
        self.counts["ref_clipped"] += ref_clipped
        self.counts["exec_clipped"] += exec_clipped
        self.counts["random_release_chunks"] += bool(info.get("random_release"))
        return targets, state_vec, ref_norm.astype(np.float32), exec_norm.astype(np.float32), info

    def fetch_once(self):
        from rlt_online.evo import SOURCE_HUMAN, SOURCE_RL
        from rlt_online.rollout import RLChunk
        args, sess = self.args, self.sess
        obs, raw = self.observe()
        dist = float(np.linalg.norm(raw["ee"][:3] - self.ref_point))
        self.last_obs = (raw["t_obs"], dist, float(raw["opening"]))
        if self.protocol.phase == "wall":
            if dist < self.closest[0]:
                self.closest = (dist, float(raw["opening"]))
            low, high = sess.proto_cfg.holding_opening
            if dist < sess.proto_cfg.radius_m and not low <= raw["opening"] <= high and not self.warned_not_holding \
                    and not args.mock_arm:
                self.warned_not_holding = True
                self.say(f"⚠ 已到套筒附近（离松手点 {dist * 1000:.0f} mm），但夹爪开度 {raw['opening']:.2f} 不在夹持范围 "
                         f"{low:.2f}~{high:.2f}，不会交接：夹爪没夹住螺栓？（夹着螺栓时开度约 0.17~0.22）", bell=True)
        with self.state:
            xyz, opening = raw["ee"][:3], raw["opening"]
            if (args.mock_arm and args.mock_handover_after is not None and self.t0 is not None
                    and raw["t_obs"] - self.t0 >= args.mock_handover_after):
                xyz, opening = sess.proto_cfg.reference_point_m, 0.2      # 只在模拟臂干跑时伪造交接条件
                if self.protocol.phase == "wall":
                    self.gripper_cmd, self.grip_pending = 0.0, None      # 同时当作夹着螺栓
            self.protocol.on_observation(raw["t_obs"], xyz, opening)
            phase, truncating = self.protocol.phase, self.post == "truncating"
        if phase not in ("wall", "rl") and not truncating:
            return None
        rl = phase == "rl" or truncating
        rtc = self.rl_prefix(raw["t_obs"]) if rl else self.rtc_prefix(raw["t_obs"])
        t_send = time.perf_counter()
        r = self.policy.client.infer(obs, image_encoding=args.image_encoding, rtc=rtc, return_features=rl)
        if r["data_contract_hash"] != self.policy.contract_hash:
            raise RuntimeError("这一段的 data_contract_hash 和 /health 不一样：服务中途换了模型")
        targets = np.asarray(r["absolute_targets"], dtype=np.float64)
        t_recv = time.perf_counter()
        self.last_latency = t_recv - raw["t_obs"]
        self.latencies.append(self.last_latency)
        cid = self.next_id
        self.next_id += 1
        rl_rec = None
        if rl:
            targets, state_vec, ref_norm, exec_norm, info = self.actor_chunk(r, raw, obs, cid)
        jump = None
        with self.state:
            old = self.plan
            held = t_recv - raw["t_obs"] if old is None else max(0.0, t_recv - max(raw["t_obs"], old.end))
            chunk = wx.Chunk(cid, raw["t_obs"] + held, wx.pose6_to_T(raw["ee"]), targets, self.fps)
            if rl:
                t_start = chunk.t0 + sess.D / self.fps
                late = max(0, int(math.ceil((t_recv - t_start) * self.fps)))
                self.counts["late_chunks"] += late > 0
                spacing = None
                if self.prev_rl_obs is not None and not truncating:
                    # D1：相邻两次观测要正好隔 C 帧。相机按帧打时间戳，触发时刻和拍到的帧差一个固定时延
                    # （11-10 真机：在第 12 步触发拍到的是第 11 帧）；按实际间隔把触发时刻挪一步
                    # 用误差的滑动平均，平均偏半帧以上才挪（时间戳不按帧对齐时单次误差会来回摆）
                    spacing = int(round((raw["t_obs"] - self.prev_rl_obs) * self.fps))
                    sess.spacing_err = 0.7 * sess.spacing_err + 0.3 * (spacing - sess.C)
                    if sess.spacing_err <= -0.5:
                        sess.trigger_shift, sess.spacing_err = min(sess.trigger_shift + 1, 3), 0.0
                    elif sess.spacing_err >= 0.5:
                        sess.trigger_shift, sess.spacing_err = max(sess.trigger_shift - 1, -3), 0.0
                if not truncating:
                    self.prev_rl_obs = raw["t_obs"]
                    if self.first_rl_start is None:
                        self.first_rl_start = t_start
                source = SOURCE_HUMAN if self.human_open else SOURCE_RL
                record = RLChunk(cid, t_start, state_vec, ref_norm, exec_norm, source,
                                 bool(info.get("random_release")),
                                 {**info, "late_steps": late, "spacing_frames": spacing,
                                  "trigger_shift": sess.trigger_shift})
                if self.ledger is not None:
                    self.ledger.register(record, executes=not truncating)
                self.last_rl = record if not truncating else self.last_rl
                rl_rec = {k: v for k, v in record.info.items()}
                if not truncating:                # 截断时这一段只拿来当 bootstrap 观测，不执行
                    if old is None:
                        self.plan = chunk
                    else:
                        jump = wx.T_gap(old.at(t_recv)[0], chunk.at(t_recv)[0])
                        self.plan = wx.Fade(old, chunk, t_recv, max(t_start, t_recv + (0.05 if late else 0.0)))
            else:
                if old is None:
                    self.plan = chunk
                else:
                    jump = wx.T_gap(old.at(t_recv)[0], chunk.at(t_recv)[0])
                    self.plan = wx.Fade(old, chunk, t_recv, max(t_recv + args.blend, old.end))
        if jump and (jump[0] > wx.JUMP_WARN_M or jump[1] > wx.JUMP_WARN_RAD):
            self.jumps += 1
            print(f"\n  ⚠ 第 {cid} 段切换时刻差 {jump[0] * 1000:.0f} mm / {math.degrees(jump[1]):.1f}°")
        rec = {"chunk": cid, "phase": "rl" if rl else "wall", "t_obs": raw["t_obs"], "t_send": t_send, "t_recv": t_recv,
               "latency_s": t_recv - raw["t_obs"], "held_s": held,
               "rtc_prefix_steps": None if rtc is None else len(rtc["prefix_targets"]),
               "rtc_delay": None if rtc is None else rtc["inference_delay"],
               "switch_gap_m": None if jump is None else jump[0],
               "state": [float(v) for v in raw["state"]], "targets": targets.tolist(), "rl": rl_rec}
        self.chunk_log.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        self.chunk_log.flush()
        return chunk

    # ── 按键 ──
    def next_key(self, now):
        if self.mock_keys and self.protocol.handover_t is not None:
            key, after = self.mock_keys[0]
            if now - self.protocol.handover_t >= after:
                self.mock_keys.pop(0)
                return key
        if self.args.mock_arm and self.post == "hold_bolt":
            return "\n"
        return self.kb.key() if self.kb is not None else None

    def say(self, text, bell=False):
        self.messages.append((time.perf_counter(), text))
        if self.tty:                                      # 覆盖掉状态行再打；响铃 = 要操作员动手
            print(f"\r\x1b[K  {text}{chr(7) if bell else ''}", flush=True)
        else:
            print(f"\n  {text}", flush=True)

    def status_line(self, now, phase):
        """终端底部一行实时状态（每 0.5 s 刷新，不进日志）。"""
        if not self.tty or now - self.status_t < 0.5:
            return
        self.status_t = now
        cfg = self.sess.proto_cfg
        where = ""
        if self.last_obs is not None:
            where = f"离松手点 {self.last_obs[1] * 1000:4.0f} mm  开度 {self.last_obs[2]:.2f}"
        if self.post == "hold_bolt":
            what = "等回车：扶住螺栓后按回车，夹爪会张开"
        elif self.post == "lifted":
            what = "等标签：s 成功  f 失败  r 丢弃"
        elif self.post == "released":
            what = "已松手，上抬中"
        elif self.post == "truncating":
            what = "截断：走完当前这一块"
        elif phase == "rl":
            keys = "o 松手  f 失败  r 丢弃" if self.args.mode == "train" else "f 失败  r 丢弃"
            what = (f"actor {now - self.protocol.handover_t:4.1f}/{cfg.rl_phase_s:.0f} s  "
                    f"[{self.sess.explore_mode()}]  {keys}")
        else:
            what = f"WALL 搬运 {now - self.t0:4.1f}/{cfg.wall_phase_s:.0f} s  r 丢弃"
        print(f"\r\x1b[K  ▸ {what}   {where}", end="", flush=True)

    def handle_key(self, key, now):
        if key is None:
            return
        if len(key) == 1:
            key = key.lower()                            # 大写锁定开着也认
        name = "space" if key == " " else key
        if self.post == "hold_bolt":
            if key in ENTER_KEYS:
                self.hold_done = True
            return
        if key in ENTER_KEYS:
            return
        if self.args.mode != "train" and name == self.sess.keys["release"]:
            return                                       # 评估不许人工松手
        with self.state:
            action = self.protocol.on_key(name, now)
        if action == "command_open":
            self.human_open = True
            if self.ledger is not None:
                self.ledger.mark_human_open(now)
            self.say("o：人工松手")
        elif action == "label":
            if self.release is None and self.ledger is not None:
                self.ledger.close_at(now)                # 没松手就判失败（螺栓掉了 / 卡死）：截到这一步
            self.say({"success": "s：成功", "failure": "f：失败"}[self.protocol.outcome])
        elif action == "discard":
            self.post = "hold_bolt"
            self.say("r：这一条丢弃，不入库。扶住螺栓，按回车后夹爪张开", bell=True)
        elif name in ("s", "f", "o") and self.protocol.phase == "wall":
            self.say(f"{name}：还没交接给 actor，这个键现在不起作用（要丢弃按 r）")
        elif name == "s" and self.protocol.phase == "rl":
            self.say("s：还没松手，不能判成功（螺栓掉了 / 卡死按 f）")

    # ── 回路 ──
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
        prev_grip = self.gripper_cmd
        while self.running:
            now = time.perf_counter()
            if getattr(self.env.arm, "latched", False):
                self.say("✗ 臂刹车锁死（latch），停")
                self.end_reason = "latched"
                break
            self.handle_key(self.next_key(now), now)
            with self.state:
                if self.protocol.phase in ("wall", "rl"):
                    self.protocol.on_tick(now)
                phase, outcome = self.protocol.phase, self.protocol.outcome
                if isinstance(self.plan, wx.Fade) and now >= self.plan.t_end:
                    self.plan = self.plan.new
                plan = self.plan
            if phase == "rl" and not self.rl_deadband:
                self.mapper.config.deadband_m, self.mapper.config.deadband_rad = self.deadbands["rl"]
                self.rl_deadband = True
                keys = "o 人工松手  f 失败  r 丢弃" if args.mode == "train" else "f 失败  r 丢弃"
                self.say(f"交接给 actor（开始后 {self.protocol.handover_t - start:.1f} s）   {keys}")
            if outcome == "truncated" and self.post is None:
                self.post = "truncating"
                self.say(f"RL 阶段到 {self.sess.proto_cfg.rl_phase_s:.0f} s 没松手：截断（走完当前块）")
            elif outcome in ("wall_failure",) and self.post is None:
                self.post = "hold_bolt"
                cfg = self.sess.proto_cfg
                dist, opening = self.closest
                why = ("一直没到套筒附近" if not np.isfinite(dist) else
                       f"最近离松手点 {dist * 1000:.0f} mm（要 < {cfg.radius_m * 1000:.0f} mm），当时开度 "
                       f"{opening:.2f}（要 {cfg.holding_opening[0]:.2f}~{cfg.holding_opening[1]:.2f}）")
                self.say(f"{cfg.wall_phase_s:.0f} s 没交接：WALL 阶段失败（{why}）。扶住螺栓，按回车后夹爪张开",
                         bell=True)
            elif outcome in ("success", "failure") and self.post in ("released", "lifted", None):
                if self.gripper_cmd < 0.5:
                    self.post = "hold_bolt"
                    self.say("夹爪还夹着：扶住螺栓，按回车后夹爪张开", bell=True)
                else:
                    break
            if self.post == "hold_bolt" and self.hold_done:
                break
            if self.ledger is not None and phase == "rl":
                self.ledger.advance(now)
            if self.post == "truncating" and (self.ledger is None or self.ledger.truncate(now)):
                self.post, self.lift = "released", None
                self.say("截断已入库；上抬后扶住螺栓，按回车后夹爪张开", bell=True)
                self.after_lift = "hold_bolt"
            self.status_line(now, phase)

            row = {"t": now, "chunk": -1, "s": np.nan, "target": np.full(6, np.nan), "q": np.full(7, np.nan),
                   "pos_err": np.nan, "rot_err": np.nan, "grip_pred": np.nan}
            with self.hw:
                ee_now = np.asarray(self.env.arm.get_ee_pose(), dtype=np.float64)
                if self.post in ("released", "lifted", "hold_bolt"):
                    self.post_motion(now, ee_now, row)
                elif plan is None:
                    self.backend.hold(self.gripper_cmd)
                    if self.requested_for is None:
                        self.requested_for = -1
                        self.want.set()
                else:
                    T, g, s = plan.at(now)
                    gap = wx.T_gap(wx.pose6_to_T(ee_now), T)
                    if gap[0] > args.max_offset_m or gap[1] > math.radians(args.max_offset_deg):
                        self.say(f"✗ 目标离臂 {gap[0] * 1000:.0f} mm / {math.degrees(gap[1]):.1f}°，超过偏离保护：保持并停")
                        self.backend.hold(self.gripper_cmd)
                        self.end_reason = "offset_guard"
                        break
                    stall = None if args.mock_arm else self.stalled(now, ee_now, T)
                    if stall:
                        self.say(f"✗ 臂 {wx.STALL_S:.0f} s 只动 {stall[0] * 1000:.1f} mm、指令变 {stall[1] * 1000:.0f} mm："
                                 f"控制柜停了（powerState={self.power_state()}）")
                        self.end_reason = "stall"
                        break
                    if self.human_open and phase == "rl":
                        self.gripper_cmd, self.grip_pending = 1.0, None      # 人工松手不等确认
                        grip = 1.0
                    elif self.post == "truncating":
                        grip = self.gripper_cmd                              # 截断后走完当前块，夹爪不再变
                    else:
                        grip = self.gripper_from(g, now)
                    q, pe, re_ = self.backend.command(T, grip)
                    row.update(chunk=plan.id, s=s, target=wx.T_to_pose6(T), pos_err=pe, rot_err=re_, grip_pred=g)
                    if q is not None:
                        row["q"] = np.asarray(q, dtype=np.float64)
                    if phase == "rl" and self.release is None and (self.human_open or (prev_grip < 0.5 <= grip)):
                        active = self.ledger.active(now) if self.ledger is not None else None
                        if self.human_open:
                            source = "human"
                        elif args.mode == "vla-only" or self.first_rl_start is None or now < self.first_rl_start:
                            source = "wall"              # actor 的第一块还没开始执行：是 WALL 自己松的
                        else:
                            source = "random" if active is not None and active.random_release else "actor"
                        self.on_release(now, source, ee_now)
                    requesting = phase in ("wall", "rl") or self.post == "truncating"
                    if phase == "rl" or self.post == "truncating":
                        due = s >= self.sess.C - 1 + self.sess.trigger_shift       # D1：每 C 帧拍一次
                    else:
                        due = wx.HORIZON - 1 - s <= args.queue_threshold          # WALL 阶段同 wall_execute
                    if requesting and ((due and self.requested_for != plan.id) or self.requested_for is None):
                        self.requested_for = plan.id
                        self.want.set()
                row["ee"] = ee_now
            prev_grip = self.gripper_cmd
            row["grip_cmd"] = self.gripper_cmd
            self.ticks.append(row)
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()
        self.running = False

    def on_release(self, now, source, ee_now):
        with self.state:
            self.protocol.on_release(now, source)
        self.release = (now, source)
        rows = self.ledger.close_at(now) if self.ledger is not None else None
        self.release_rows = rows
        self.post, self.lift, self.after_lift = "released", None, "lifted"
        self.say(f"松手（{ {'human': '人工 o', 'random': '随机尝试', 'actor': 'actor', 'wall': 'WALL'}[source] }），"
                 f"交接后 {now - self.protocol.handover_t:.1f} s。上抬后看螺栓在不在套筒里：s 成功  f 失败  r 丢弃",
                 bell=True)

    def post_motion(self, now, ee_now, row):
        if self.post == "released":
            if self.lift is None:
                T0 = wx.pose6_to_T(ee_now)
                T1 = T0.copy()
                T1[2, 3] += LIFT_M
                self.lift = (now + LIFT_WAIT_S, T0, T1)
            t_lift, T0, T1 = self.lift
            if now < t_lift:
                self.backend.hold(self.gripper_cmd)
                return
            a = min(1.0, (now - t_lift) / LIFT_S)
            T = wx.slerp_T(T0, T1, a)
            q, pe, re_ = self.backend.command(T, self.gripper_cmd)
            row.update(target=wx.T_to_pose6(T), pos_err=pe, rot_err=re_)
            if a >= 1.0:
                self.post = getattr(self, "after_lift", "lifted")
        else:
            self.backend.hold(self.gripper_cmd)


# ───────────────────────────────────────────────────────── 入口

def parse_args():
    from atom_robot.ar5.end_effector import add_end_effector_args
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("--server", required=True, help="bolt 推理服务，如 http://127.0.0.1:8080")
    ap.add_argument("--task-dir", required=True, help="采集任务目录：prompt、夹持力、速度取它的 dataset.json")
    ap.add_argument("--config", required=True, help="Stage 2 配置 yaml（WALL-OSS-CRP/configs/rlt_online/）")
    ap.add_argument("--run-dir", type=Path, required=True, help="这次训练 / 评估的目录（新建；续训用同一个加 --resume）")
    ap.add_argument("--mode", choices=("train", "eval", "vla-only"), default="train",
                    help="train：在线训练；eval：actor 均值、不加噪声、不许按 o、不入库；vla-only：交接后仍是 WALL（对照组）")
    ap.add_argument("--resume", action="store_true", help="train：从 <run-dir>/trainer/latest_online_state.pt 接着训")
    ap.add_argument("--policy-from", help="eval：latest_online_state.pt 或 step_NNNNNN 目录")
    ap.add_argument("--episodes", type=int, default=None,
                    help="这次跑多少条（train 默认跑到配置的 budget.train_episodes；eval / vla-only 必须给）")
    # 11-10 用户定 RL 阶段 1 mm / 0.5°：死区以下的目标变化不下发，3 mm 会吃掉 actor 的毫米级修正和探索
    ap.add_argument("--rl-deadband-mm", type=float, default=1.0, help="RL 阶段（交接后）的末端死区，默认 1 mm")
    ap.add_argument("--rl-deadband-deg", type=float, default=0.5, help="RL 阶段的末端死区，默认 0.5°")
    ap.add_argument("--deadband-mm", type=float, default=3.0, help="WALL 阶段死区，默认 3 mm（bolt 现在用的）")
    ap.add_argument("--deadband-deg", type=float, default=1.7, help="WALL 阶段死区，默认 1.7°")
    ap.add_argument("--seed", type=int, default=None, help="探索噪声的随机种子（默认随机，记进 run.json）")
    ap.add_argument("--no-save-frames", dest="save_frames", action="store_false",
                    help="RL 阶段每次请求的两路画面默认存成 jpg（审计、以后补算中间帧特征）")
    ap.add_argument("--mock-arm", action="store_true", help="模拟臂 / 夹爪 / 相机，不碰硬件")
    ap.add_argument("--mock-handover-after", type=float, default=None, help="模拟臂：开始后这么多秒伪造交接条件")
    ap.add_argument("--mock-keys", default="", help='模拟臂：交接后按时注入按键，如 "o@2,s@5"')
    ap.add_argument("--mock-episodes", type=int, default=None, help="模拟臂：跑几条（不等回车）")
    ap.add_argument("--autostart", action="store_true", help="不等回车（只给模拟臂用）")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--control-hz", type=float, default=None)
    ap.add_argument("--rtc-method", choices=("inpaint", "guidance"), default="inpaint")
    ap.add_argument("--blend", type=float, default=0.1)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--image-encoding", choices=("png", "jpeg"), default="jpeg")
    ap.add_argument("--max-errors", type=int, default=5)
    ap.add_argument("--max-offset-m", type=float, default=0.12)
    ap.add_argument("--max-offset-deg", type=float, default=20.0)
    ap.add_argument("--ori-cost", type=float, default=wx.COLLECT_ORI_COST)
    ap.add_argument("--verbose", action="store_true")
    add_end_effector_args(ap)
    args = ap.parse_args()
    args.end_effector, args.ik, args.no_camera, args.no_rtc, args.max_seconds = "omnipicker", "weighted", False, False, 0.0
    # bootstrap 会 chdir 到 SO101：命令行给的相对路径一律按启动时所在的目录解析
    args.run_dir = wx.user_path(args.run_dir)
    args.config = str(wx.user_path(args.config))
    args.task_dir = str(wx.user_path(args.task_dir))
    if args.policy_from:
        args.policy_from = str(wx.user_path(args.policy_from))
    keys = []
    for item in filter(None, (s.strip() for s in args.mock_keys.split(","))):
        k, _, t = item.partition("@")
        keys.append(("\n" if k == "enter" else k, float(t)))
    args.mock_keys = keys
    if (args.mock_handover_after is not None or keys or args.mock_episodes or args.autostart) and not args.mock_arm:
        raise SystemExit("✗ --mock-* / --autostart 只能和 --mock-arm 一起用")
    if args.mode == "eval" and not args.policy_from:
        raise SystemExit("✗ --mode eval 要 --policy-from")
    if args.mode != "train" and args.episodes is None and not args.mock_episodes:
        raise SystemExit("✗ eval / vla-only 要 --episodes")
    return args


def open_hardware(args, so101):
    """wall_execute.main 里建 env / 映射器 / weighted 后端的那段（只走 weighted，不走 sdk / 回放）。"""
    from atom_robot.ar5.ar5_env import AR5Env
    from atom_robot.ar5.cartesian_mapping import CartesianMapConfig, CartesianMapper
    from atom_robot.ar5.config import AR5Config
    from atom_robot.ar5.end_effector import configure_end_effector
    from atom_robot.ar5.joint_mapping import JointMapConfig
    from atom_robot.ar5.teleop_joint import hold_posture

    config = AR5Config()
    collect_cfg = config.collect
    collect_cfg.space, collect_cfg.cart_ik = "cartesian", "weighted"
    collect_cfg.__post_init__()
    args.control_hz = args.control_hz or collect_cfg.control_hz
    config.mock, config.use_hand = args.mock_arm, True
    configure_end_effector(config, args)
    config.arm.use_realtime = True
    cart_cfg = CartesianMapConfig(position_scale=collect_cfg.scale, ik="weighted", tcp_z=collect_cfg.cartesian_tcp_z(),
                                  rotation_scale=collect_cfg.cartesian_rot_scale(),
                                  centre_joints=collect_cfg.cartesian_centre_joints)
    cart_cfg.orientation_cost = args.ori_cost
    cart_cfg.deadband_m, cart_cfg.deadband_rad = args.deadband_mm / 1000, math.radians(args.deadband_deg)
    mapper = CartesianMapper(config.arm, config.arm.workspace_min, config.arm.workspace_max, config=cart_cfg,
                             joint_cfg=JointMapConfig.load(**collect_cfg.mapping_overrides))
    lo = np.asarray(config.arm.joint_min, dtype=np.float64) + config.arm.joint_limit_margin
    hi = np.asarray(config.arm.joint_max, dtype=np.float64) - config.arm.joint_limit_margin
    mapper._kin.joint_min, mapper._kin.joint_max = lo, hi
    mapper._kin.joint_mid = (lo + hi) / 2.0
    mapper._kin.joint_range = np.maximum(hi - lo, 1e-9)
    mapper._posture = np.clip(mapper.centre, lo, hi)
    lock = None
    if not args.mock_arm:
        from atom_robot.ar5.collect import acquire_single_instance_lock
        from atom_robot.ar5.unseize import diagnose, power_back_on
        lock = acquire_single_instance_lock()
        if lock is None:
            raise SystemExit("✗ /tmp/atom_ar5_collect.lock 被占着：collect 或执行器还开着，同一台臂不能两边驱动")
        st = diagnose(config.arm, quiet=True)
        if st["physical_stop"]:
            raise SystemExit(f"✗ {st['why']}（急停 / 安全门是物理的）")
        if st["reachable"] and (st["seized"] or st.get("idle_off")):
            print(f"  臂现在起不来：{st['why']}；先自动上电（只使能伺服，臂不动）")
            if power_back_on() != 0:
                raise SystemExit("✗ 上电没成功")
    env = AR5Env(config)
    env.connect()
    env.arm.clear_latch()
    start_q = mapper.centre.copy()
    if args.mock_arm:
        arm = env.arm
        arm.kinematics = mapper._kin
        arm._joints = start_q.copy()
        with arm._setpoint_lock:
            arm._q_cmd, arm._q_goal = start_q.copy(), start_q.copy()
        arm.get_ee_pose = lambda: wx.T_to_pose6(mapper._kin.fk(arm._joints))
    elif not getattr(env.arm, "_rt_active", False):
        raise SystemExit("✗ 臂的实时通道没起来（python -m atom_robot.ar5.unseize 看一眼）")
    backend = wx.WeightedBackend(env, mapper, hold_posture)
    return env, backend, mapper, cart_cfg, start_q, lock


def wait_enter(kb, text, mock):
    print(f"\n  {text}", flush=True)
    if mock:
        return True
    while True:
        k = kb.key() if kb is not None else None
        if k in ENTER_KEYS:
            return True
        if k in ("q", "Q"):
            return False
        time.sleep(0.02)


OUTCOME_NAMES = {"success": "成功", "failure": "失败", "truncated": "截断", "wall_failure": "WALL 没交接",
                 "discarded": "丢弃", "aborted": "中断", "latched": "锁死", "stall": "失速", "offset_guard": "偏离保护"}
RELEASE_NAMES = {"human": "人工 o", "random": "随机尝试", "actor": "actor", "wall": "WALL", None: "没松手"}


def read_tally(results):
    """这个 run 目录 results.tsv 里各结果的条数（续训时接着数）。"""
    tally = {}
    for line in results.read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) > 2:
            tally[cols[2]] = tally.get(cols[2], 0) + 1
    return tally


def progress_text(sess):
    """训练进度：warmup 还差什么 / 只训 critic 还剩几条 / actor 在学。"""
    tr = sess.trainer
    if tr is None:
        return None
    cfg = tr.cfg
    if tr.warmup_completed_at_episode is None:
        succ, fail = tr.replay_buffer.count_outcomes()
        return (f"warmup（actor = WALL，不加位姿噪声）：入库 {sess.stored}/{cfg.warmup_episodes} 条  "
                f"成功 {succ}/{cfg.min_warmup_successes}  失败 {fail}/{cfg.min_warmup_failures}  "
                f"transition {len(tr.replay_buffer)}/{cfg.min_warmup_transitions}")
    left = tr.warmup_completed_at_episode + cfg.critic_only_episodes - sess.stored
    if left > 0:
        return f"只训 critic（actor 还没开始改）：还剩 {left} 条"
    return "actor 在学（加探索噪声）"


def print_settings(args, sess, prompt, target):
    m, x = sess.meta, sess.explorer.config
    pc, oc = sess.policy_cfg, sess.online_cfg
    mode = {"train": "训练", "eval": "评估（actor 不加噪声、不许按 o、不入库）", "vla-only": "对照组（交接后仍是 WALL）"}
    print(f"\n  ══ WALL-RLT 在线{mode[args.mode]} · {'模拟臂' if args.mock_arm else '真臂'} ══")
    print(f"  指令      {prompt}")
    print(f"  目录      {args.run_dir}")
    print(f"  进度      已入库 {sess.stored} 条，这次最多 {target} 条；训练预算 {m['budget']['train_episodes']} 条，"
          f"第 {'/'.join(map(str, m['budget']['evaluate_after']))} 条后停下评估")
    print(f"  交接      离松手点 < {sess.proto_cfg.radius_m * 1000:.0f} mm 且开度 {sess.proto_cfg.holding_opening[0]:.2f}"
          f"~{sess.proto_cfg.holding_opening[1]:.2f}；WALL 阶段 {sess.proto_cfg.wall_phase_s:.0f} s、"
          f"RL 阶段 {sess.proto_cfg.rl_phase_s:.0f} s")
    print(f"  动作块    C={sess.C}、从第 {sess.D} 步执行；死区 WALL {args.deadband_mm:g} mm/{args.deadband_deg:g}°、"
          f"RL {args.rl_deadband_mm:g} mm/{args.rl_deadband_deg:g}°")
    if args.mode == "train":
        print(f"  探索      每块偏移 {x.position_std_m * 1000:g} mm / {x.rotation_std_deg:g}°；随机试松手每块 "
              f"{x.random_release_prob:g}；夹爪 BC 权重 {pc.actor_bc_dim_weights[9]:g}")
        print(f"  训练      warmup {oc.warmup_episodes} 条（≥{oc.min_warmup_successes} 成 ≥{oc.min_warmup_failures} 败"
              f" ≥{oc.min_warmup_transitions} transition）→ 只训 critic {oc.critic_only_episodes} 条 → actor；"
              f"batch {oc.batch_size}、UTD {pc.utd_ratio}")
        state = progress_text(sess)
        if state:
            print(f"  现在      {state}")
    keys = "回车 开始/继续   o 人工松手   s 成功   f 失败   r 丢弃   q 结束（在回车提示处）   Ctrl-C 急停"
    if args.mode != "train":
        keys = keys.replace("o 人工松手   ", "")
    print(f"  按键      {keys}（大小写都行）")
    if not args.mock_arm:
        print("  提醒      实时画面窗口弹出后点回这个终端：按键只从终端读，焦点在画面窗口上按键不响应")


def main():
    so101, wall = wx.bootstrap()
    args = parse_args()
    from atom_robot.ar5.collect import RawKeyboard

    prompt, force, speed = wx.load_task(args.task_dir)
    if args.omnipicker_force is None and force is not None:
        args.omnipicker_force = int(force)
    if args.omnipicker_speed is None and speed is not None:
        args.omnipicker_speed = int(speed)
    if args.run_dir.exists() and not (args.resume or args.mode != "train"):
        raise SystemExit(f"✗ {args.run_dir} 已存在：续训加 --resume，或换一个目录")
    if args.mode != "train" and args.run_dir.exists() and any(args.run_dir.glob("ep_*")):
        raise SystemExit(f"✗ {args.run_dir} 里已有 episode：评估每次用新目录")
    if args.resume and not (args.run_dir / "trainer" / "latest_online_state.pt").is_file():
        raise SystemExit(f"✗ --resume 但 {args.run_dir}/trainer/latest_online_state.pt 不存在")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    remote = wx.RemotePolicy(args.server, args.timeout, args.image_encoding, rtc=True)
    sess = Session(args, wall, remote)
    args.queue_threshold = 20                             # WALL 阶段同 wall_execute 默认；RL 阶段按 C 帧触发
    sess.trigger_shift = 0 if args.mock_arm else 1        # 真机相机晚一帧，见 RLExecutor.fetch_once
    sess.spacing_err = 0.0
    budget = int(sess.meta["budget"]["train_episodes"])
    target = args.mock_episodes or args.episodes or (budget - sess.stored)
    deadbands = {"wall": (args.deadband_mm / 1000, math.radians(args.deadband_deg)),
                 "rl": (args.rl_deadband_mm / 1000, math.radians(args.rl_deadband_deg))}

    env, backend, mapper, cart_cfg, start_q, lock = open_hardware(args, so101)
    view = None
    results = args.run_dir / "results.tsv"
    if not results.exists():
        results.write_text("episode\tmode\toutcome\thandover_s\trl_phase_s\trelease\trl_rows\tupdates\tnote\n",
                           encoding="utf-8")
    run_meta = {"started": datetime.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
                "mode": args.mode, "prompt": prompt, "server": remote.describe, "session": sess.provenance,
                "deadbands": {"wall_mm_deg": [args.deadband_mm, args.deadband_deg],
                              "rl_mm_deg": [args.rl_deadband_mm, args.rl_deadband_deg]},
                "queue_threshold": args.queue_threshold, "mock_arm": args.mock_arm}
    (args.run_dir / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print_settings(args, sess, prompt, target)
    tally = read_tally(results)
    eval_points = [int(n) for n in sess.meta["budget"]["evaluate_after"]]
    from rlt_online.episode import EpisodeProtocol
    kb_ctx = RawKeyboard()
    kb = kb_ctx.__enter__() if not args.mock_arm else None
    done_here = 0
    ex = None
    try:
        if not (args.no_view or args.mock_arm or not os.environ.get("DISPLAY")):
            view = wx.LiveView(env.cameras, args.run_dir / "viewer.log")
        while done_here < target:
            index = len(list(args.run_dir.glob("ep_*")))
            ep_dir = args.run_dir / f"ep_{index:04d}"
            if not wait_enter(kb, f"── 第 {index + 1} 条（ep_{index:04d}）──  螺栓插回蓝色座、场景摆好后按回车"
                                  f"（臂会回 0 位并张开夹爪，**臂会动**）；q 结束", args.mock_arm):
                break
            ok = env.goto_joints(start_q, timeout_s=20.0)
            if ok is False:
                print("  ⚠ 20 s 内没到起始姿势，自己看一眼")
            env.hand.set_opening(1.0)
            mapper.config.deadband_m, mapper.config.deadband_rad = deadbands["wall"]
            ep_dir.mkdir()
            base = sess.trainer.start_episode(sess.stored) if sess.trainer is not None else None
            protocol = EpisodeProtocol(sess.proto_cfg, t_start=time.perf_counter())
            ex = RLExecutor(env, backend, remote, args, prompt, ep_dir, mapper, sess, protocol, kb, deadbands)
            if view is not None:
                view.ex = ex
            try:
                ex.run()
            finally:
                ex.running = False
                with ex.hw:
                    backend.hold(ex.gripper_cmd)
                    backend.stop()
                ex.save()
            outcome = protocol.outcome or ex.__dict__.get("end_reason") or "aborted"
            stats = None
            note = ex.__dict__.get("end_reason") or ""
            counted = False
            if sess.trainer is not None:
                rl_rows = ex.ledger.rows_added if ex.ledger is not None else 0
                if outcome in ("success", "failure"):
                    sess.collector.flush_episode(outcome == "success")
                if protocol.stores_transitions and rl_rows > 0:
                    stats = sess.trainer.maybe_update(sess.stored, base)
                    sess.stored += 1
                    counted = True
                    sess.trainer.save_latest_state(sess.stored)
                elif protocol.stores_transitions:
                    note = "no_rl_rows（actor 的块还没执行就结束了：不计入预算）"
            summary = {**protocol.summary(), "end_reason": ex.__dict__.get("end_reason"), "counts": ex.counts,
                       "release_rows": ex.__dict__.get("release_rows"),
                       "ledger_rows": None if ex.ledger is None else ex.ledger.rows_added,
                       "stored_index": sess.stored - 1 if counted else None,
                       "explore_mode_end": sess.explore_mode(),
                       "update": None if stats is None else {k: v for k, v in stats.items() if k != "loss"} | {"loss": stats["loss"]},
                       "messages": ex.messages}
            (ep_dir / "episode.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                                 encoding="utf-8")
            with open(results, "a", encoding="utf-8") as f:
                f.write(f"{index}\t{args.mode}\t{outcome}\t{summary['handover_s']}\t{summary['rl_phase_s']}\t"
                        f"{summary['release_source']}\t{summary['ledger_rows']}\t"
                        f"{None if stats is None else stats['actual_updates']}\t{note}\n")
            tally[outcome] = tally.get(outcome, 0) + 1
            parts = [f"第 {index + 1} 条：{OUTCOME_NAMES.get(outcome, outcome)}"]
            if summary["handover_s"] is not None:
                parts.append(f"开始后 {summary['handover_s']:.1f} s 交接")
            if summary["rl_phase_s"] is not None:
                parts.append(f"actor {summary['rl_phase_s']:.1f} s")
            if summary["handover_s"] is not None:
                parts.append(f"松手：{RELEASE_NAMES.get(summary['release_source'], summary['release_source'])}")
            if stats is not None:
                parts.append(f"更新 {stats['actual_updates']} 次")
            if note.startswith("no_rl_rows"):
                parts.append("actor 还没动就结束了，不计入预算")
            print("\n  " + "，".join(parts))
            print("  累计：" + "  ".join(f"{OUTCOME_NAMES.get(k, k)} {v}" for k, v in tally.items())
                  + (f"   已入库 {sess.stored}/{budget} 条" if sess.trainer is not None else ""))
            state = progress_text(sess)
            if state:
                print(f"  进度：{state}")
            if ex.__dict__.get("end_reason") in ("latched", "stall", "offset_guard"):
                print("  ✗ 安全停机，这次会话到此为止（unseize --recover 之后再开）")
                break
            env.hand.set_opening(1.0)
            env.goto_joints(start_q, timeout_s=20.0)
            done_here += 1
            if sess.trainer is not None and stats is not None and sess.stored in eval_points and not args.mock_arm:
                print(f"\n  ★ 已入库 {sess.stored} 条，到了评估点：这次会话先停。评估（20 条示例，每次新目录）：\n"
                      f"    {sys.executable} {Path(__file__).resolve()} --server {args.server} --task-dir {args.task_dir} "
                      f"--config {args.config} --run-dir {args.run_dir}_eval{sess.stored} --mode eval "
                      f"--policy-from {args.run_dir}/trainer/latest_online_state.pt --episodes 20")
                break
            if sess.trainer is not None and sess.stored >= budget:
                print(f"\n  ★ 训练预算 {budget} 条用完了")
                break
    except KeyboardInterrupt:
        print("\n  Ctrl-C：停（这一条不入库）")
        if ex is not None:
            ex.running = False
            with ex.hw:
                backend.hold(ex.gripper_cmd)
                backend.stop()
    finally:
        if sess.trainer is not None:
            sess.trainer.save_latest_state(sess.stored)
            sess.trainer.close()
        if kb is not None:
            kb_ctx.__exit__(None, None, None)
        if view is not None:
            view.close()
        env.close()
        if lock is not None:
            lock.close()
    print(f"\n  结束：这次 {done_here} 条，已入库共 {sess.stored} 条；记录在 {args.run_dir}")
    if args.mode == "train" and not args.mock_arm and sess.stored < budget:
        print(f"  接着训：\n    {sys.executable} {Path(__file__).resolve()} --server {args.server} --task-dir {args.task_dir} "
              f"--config {args.config} --run-dir {args.run_dir} --resume")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  取消了")
        sys.exit(130)

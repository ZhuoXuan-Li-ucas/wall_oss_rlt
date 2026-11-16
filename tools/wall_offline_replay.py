"""WALL 官方模型真实观测离线回放：把 top_error 采集数据逐帧喂给官方 WALL-OSS-0.5，对比预测和数据里的末端运动。

照 算法/DexGraspVLA-/offline_replay.py（C1）改：参数、输出（summary.json、preds.npz、几条的曲线图）、
「原地不动」对照组都一样。只读数据和模型，不碰任何硬件，在 sm_120 推理镜像里跑。

和 DexGraspVLA 版不同的地方：
- 官方 WALL 没在 AR5 上训过，动作是 26 维（双臂末端 + 底盘、升降、头），没有关节。这里用它的单臂接口
  （harrix 的 encode_proprio / decode_chunk，LIBERO 评测走的就是这套）：右臂末端位置 3 + 6D 姿态 6 + 夹爪 1，
  其余槽位 mask 掉。所以比的是末端运动，看的是零样本表现，不是训练效果。
- 建模型、调预测和官方检查器 verify_full_model.py 完全一样（fake_inference.py 的 _predict_raw_action），
  只是把合成观测换成真实观测。
- 输出是相对当前帧的末端位姿（wall-x lerobot loader 的 _to_relative_action：位置相减、姿态 R_abs·R_stateᵀ）。
  第 k 步按数据第 t+1+k 帧算（lerobot 的 action[t] 是下一帧要到的目标，和 DexGraspVLA 的 action 同一个约定）。
- generate_flow_action 不用传进去的 noise，自己 torch.randn，所以每次预测前 torch.manual_seed(seed)：
  每个回放点用同一份噪声，同样的输入能复现。

数据约定（算法/DexGraspVLA-/SO101/数据集格式说明.md）：
- ee_pose = 法兰在 AR5 基座系下的位置（米）+ 轴角。harrix 把 eef_axisangle 当 ZYX 欧拉角 (roll, pitch, yaw) 用
  （R = Rz·Ry·Rx），所以先把轴角换成这种欧拉角再传；开头核对两边算出的 6D 一致。
- cam_high（俯视）→ face_view，cam_front（腕部）→ right_wrist_view，和 WALL-RLT 项目的 camera_mapping 一致；
  视频用 PyAV 解成 rgb24（WALL-RLT 的 npz_source.py 也是这样）。
- 夹爪开度 0 合 ~ 1 开。默认线性映射到所选 norm_key 夹爪统计的 [min, min+delta]（假设对方也是数大 = 张开，
  没法核实），预测值按同一映射还原。--gripper-map raw 则原样传。

对照组「原地不动」：32 步都等于当前位姿和开度。
开头几个回放点另做三项检查：同种子重跑（应完全一样）、换种子、把两张图换成全黑（看模型用没用图）。

用法（宿主机，source 本轮 session.env 后）：
  mkdir -p "$WALL_RUN/replay"
  docker run --rm --pull=never --network none --gpus 'device=0' \\
    --user "$(id -u):$(id -g)" $WALL_CONTAINER_ENV --workdir /opt/walloss \\
    -v "$WALL_RUN/models:/opt/walloss/models:ro" \\
    -v "$HOME/atom_episodes/top_error:/target-data:ro" \\
    -v "$WALL_ROOT/tools:/tools:ro" -v "$WALL_RUN/replay:/out" \\
    "$WALL_INFERENCE_IMAGE_ID" \\
    python /tools/wall_offline_replay.py --data /target-data/black_sleeve --name black_sleeve_x2 --stride 30
输出在 $WALL_RUN/replay/<--name>/，目录已存在就拒绝运行（不覆盖旧结果）。
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # 镜像里没有中文字体，图上用英文
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

CKPT = "/opt/walloss/models/wall-oss-0.5"
FAKE_INFERENCE = "/opt/walloss/wall-x/scripts/fake_inference.py"
CHECK_STEPS = (7, 15, 31)  # 汇总里单独列第 8、16、32 步


def build_adapter(norm_key):
    """照 verify_full_model.py 建 adapter，EvalConfig 一模一样。"""
    import wall_x._vendor.harrix.adapters  # noqa: F401  注册 adapter
    from wall_x._vendor.harrix.adapters.registry import build_adapter as _build
    from wall_x._vendor.harrix.eval_config import EvalConfig, LiberoEnvParams, autofill_from_checkpoint
    cfg = EvalConfig()
    cfg.model.checkpoint_path = CKPT
    cfg.model.norm_key = norm_key
    cfg.model.cam_names = ["face_view", "right_wrist_view"]
    cfg.model.architecture = "qwen2_5"
    cfg.model.action_mode = "flow"
    cfg.env.libero = LiberoEnvParams(num_trials_per_task=1, task_indices=[0])
    cfg = autofill_from_checkpoint(cfg)
    return _build(cfg)


def load_helper():
    spec = importlib.util.spec_from_file_location("walloss_fake_inference", FAKE_INFERENCE)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def slices(layout):
    """按 dof_config / agent_pos_config 的顺序给每个槽位切片，键去掉 follow_ / master_ 前缀。"""
    out, start = {}, 0
    for key, dim in layout.items():
        out[key.replace("follow_", "").replace("master_", "")] = slice(start, start + int(dim))
        start += int(dim)
    return out


def decode_frames(path, wanted):
    """顺序解码整段视频，只留 wanted 里的帧。返回 {帧号: rgb24} 和总帧数。"""
    frames, n = {}, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if n in wanted:
                frames[n] = frame.to_ndarray(format="rgb24")
            n += 1
    return frames, n


def to_euler_zyx(rotvec):
    """轴角 → harrix 用的 (roll, pitch, yaw)，R = Rz(yaw)·Ry(pitch)·Rx(roll)，即 scipy 的外旋 xyz。"""
    return R.from_rotvec(rotvec).as_euler("xyz")


def check_euler_convention(ee):
    from wall_x._vendor.x2robot_utils.geometry import euler_to_matrix_zyx_6d_nb
    six = euler_to_matrix_zyx_6d_nb(to_euler_zyx(ee[:, 3:]))
    ref = R.from_rotvec(ee[:, 3:]).as_matrix()[:, :2].reshape(-1, 6)  # 6D = 旋转矩阵前两行
    return float(np.abs(six - ref).max())


def make_obs(ee, grip, top, wrist):
    return {"eef_pos": ee[:3].astype(np.float32),
            "eef_axisangle": to_euler_zyx(ee[3:]).astype(np.float32),  # 名字是 harrix 的，内容是 ZYX 欧拉角
            "gripper": np.array([grip], np.float32),
            "face_view": top, "wrist_view": wrist}


def predict(adapter, helper, obs, instruction, seed):
    torch.manual_seed(seed)  # generate_flow_action 自己 torch.randn，不用 payload 里的 noise
    with torch.inference_mode():
        action = helper._predict_raw_action(adapter, {"observation": obs, "instruction": instruction, "noise": None})
    return np.asarray(action, dtype=np.float64)  # (32, 26)，已反归一化


def ground_truth(ee, opening, t, H):
    """第 k 步 = 第 t+1+k 帧相对第 t 帧；越过末尾的用最后一帧补（和训练时一样）。"""
    idx = np.minimum(np.arange(t + 1, t + 1 + H), len(ee) - 1)
    pos = ee[idx, :3] - ee[t, :3]
    rot = (R.from_rotvec(ee[idx, 3:]) * R.from_rotvec(ee[t, 3:]).inv()).as_rotvec()
    return pos, rot, opening[idx]


def angle_deg(rv_a, rv_b):
    """两组轴角之间的夹角（度），形状 (..., 3) → (...)。"""
    a, b = rv_a.reshape(-1, 3), rv_b.reshape(-1, 3)
    return np.degrees((R.from_rotvec(a).inv() * R.from_rotvec(b)).magnitude()).reshape(rv_a.shape[:-1])


def plot_episode(path, ep, ee, opening, ts, pred_abs, pred_open):
    """黑线：数据里的法兰位置和夹爪开度；红线：每个回放时刻预测的 32 步（叠到当前位姿上）。"""
    T = len(ee)
    fig, axes = plt.subplots(4, 1, figsize=(12, 7.5), sharex=True)
    for j, name in enumerate(("x (mm)", "y (mm)", "z (mm)")):
        axes[j].plot(np.arange(T), ee[:, j] * 1000, "k-", lw=1)
        for t, p in zip(ts, pred_abs):
            axes[j].plot(np.arange(t + 1, t + 1 + len(p)), p[:, j] * 1000, "r-", lw=0.9)
        axes[j].set_ylabel(name, fontsize=8)
    axes[3].plot(np.arange(T), opening, "k-", lw=1)
    for t, g in zip(ts, pred_open):
        axes[3].plot(np.arange(t + 1, t + 1 + len(g)), g, "r-", lw=0.9)
    axes[3].set_ylabel("gripper open", fontsize=8)
    axes[3].set_xlabel("frame (30 Hz)")
    for ax in axes:
        ax.tick_params(labelsize=7)
    axes[0].set_title(f"episode {ep}: black = recorded flange pose, red = WALL 32-step prediction from each replay frame",
                      fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_error_by_step(path, pos_model, pos_hold, rot_model, rot_hold):
    k = np.arange(1, pos_model.shape[0] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(k, pos_model, "r-", label="WALL")
    axes[0].plot(k, pos_hold, "k--", label="hold still")
    axes[0].set_ylabel("position error (mm)")
    axes[1].plot(k, rot_model, "r-", label="WALL")
    axes[1].plot(k, rot_hold, "k--", label="hold still")
    axes[1].set_ylabel("rotation error (deg)")
    for ax in axes:
        ax.set_xlabel("step in chunk")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="一个任务的目录：dataset.json + episode_XXXX.npz + 两路 mp4")
    ap.add_argument("--name", required=True, help="输出目录名，放在 --out-root 下")
    ap.add_argument("--out-root", default="/out")
    ap.add_argument("--episodes", default="all", help="逗号分隔的编号（按文件名排序后的序号），或 all")
    ap.add_argument("--stride", type=int, default=30, help="每条隔多少帧回放一次")
    ap.add_argument("--plot-episodes", default="0", help="画曲线图的条，逗号分隔")
    ap.add_argument("--norm-key", default="x2_normal")
    ap.add_argument("--gripper-map", choices=("range", "raw"), default="range")
    ap.add_argument("--instruction", default=None, help="默认用 npz 里的 task")
    ap.add_argument("--check-points", type=int, default=5, help="开头几个回放点做复现 / 换种子 / 全黑图检查")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=False)
    data = Path(a.data)
    npzs = sorted(data.glob("episode_*.npz"))  # 按文件名排序，不假设编号连续
    eps = list(range(len(npzs))) if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    plot_eps = {int(x) for x in a.plot_episodes.split(",") if x != ""}
    meta = json.loads((data / "dataset.json").read_text())

    t0 = time.perf_counter()
    adapter = build_adapter(a.norm_key)
    load_s = time.perf_counter() - t0
    helper = load_helper()
    tc = adapter._train_config
    H, D = adapter._action_horizon, adapter._action_dim
    act = slices(tc["dof_config"])
    s_pos, s_rot, s_grip = (act[k] for k in ("right_ee_cartesian_pos_relative", "right_ee_rotation_6D_relative",
                                             "right_gripper"))
    prop = slices(tc["agent_pos_config"])
    key = adapter._norm_key
    pmin = adapter._normalizer_propri.min[key].float().cpu().numpy()
    pdelta = adapter._normalizer_propri.delta[key].float().cpu().numpy()
    gmin, gdelta = float(pmin[prop["right_gripper"]][0]), float(pdelta[prop["right_gripper"]][0])
    if a.gripper_map == "range":
        to_model, from_model = (lambda g: gmin + g * gdelta), (lambda g: (g - gmin) / gdelta)
    else:
        to_model = from_model = lambda g: g
    print(f"模型 {CKPT}：动作 {H} 步 × {D} 维，加载 {load_s:.1f} s；norm_key {key}；"
          f"夹爪 {a.gripper_map}（{key} 统计 {gmin:.3f} ~ {gmin + gdelta:.3f}）")
    print(f"数据 {data}：{len(npzs)} 条，回放 {len(eps)} 条、每 {a.stride} 帧一次")

    from wall_x._vendor.x2robot_utils.geometry import so3_to_matrix_batch_nb

    torch.cuda.reset_peak_memory_stats()
    rec = {k: [] for k in ("ep", "t", "raw", "pred_pos", "pred_rot", "pred_open", "gt_pos", "gt_rot", "gt_open",
                           "cur_open", "ee")}
    times, checks, euler_err = [], [], 0.0
    pos_lo, pos_hi = pmin[prop["right_ee_cartesian_pos"]], (pmin + pdelta)[prop["right_ee_cartesian_pos"]]
    for ep in eps:
        z = np.load(npzs[ep], allow_pickle=True)
        ee, opening = z["ee_pose"].astype(np.float64), z["state"][:, 7].astype(np.float64)
        instruction = a.instruction or str(z["task"].item())
        T = len(ee)
        euler_err = max(euler_err, check_euler_convention(ee))
        ts = list(range(0, T, a.stride))
        stem = npzs[ep].name[:-len(".npz")]
        top, n_top = decode_frames(data / f"{stem}_cam_high.mp4", set(ts))
        wrist, n_wrist = decode_frames(data / f"{stem}_cam_front.mp4", set(ts))
        if n_top != T or n_wrist != T:
            raise ValueError(f"{stem}：视频帧数 {n_top}/{n_wrist} 和 npz 的 {T} 对不上")
        ep_abs, ep_open = [], []
        for t in ts:
            obs = make_obs(ee[t], to_model(opening[t]), top[t], wrist[t])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            raw = predict(adapter, helper, obs, instruction, a.seed)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t1)
            if raw.shape != (H, D) or not np.isfinite(raw).all():
                raise ValueError(f"{stem} 第 {t} 帧：输出形状 {raw.shape}，有限 {np.isfinite(raw).all()}")
            pos = raw[:, s_pos]
            rot = R.from_matrix(so3_to_matrix_batch_nb(raw[:, s_rot])).as_rotvec()
            grip = from_model(raw[:, s_grip][:, 0])
            if len(checks) < a.check_points:
                same = predict(adapter, helper, obs, instruction, a.seed)
                other = predict(adapter, helper, obs, instruction, a.seed + 1)
                blank = predict(adapter, helper, {**obs, "face_view": np.zeros_like(top[t]),
                                                  "wrist_view": np.zeros_like(wrist[t])}, instruction, a.seed)
                mm = lambda x: float(np.linalg.norm(x[:, s_pos] - pos, axis=1).mean() * 1000)
                checks.append({"ep": ep, "t": t, "repeat_max_abs": float(np.abs(same - raw)[:, :].max()),
                               "seed_change_mm": mm(other), "blank_images_mm": mm(blank)})
            gp, gr, go = ground_truth(ee, opening, t, H)
            for k, v in (("ep", ep), ("t", t), ("raw", raw), ("pred_pos", pos), ("pred_rot", rot), ("pred_open", grip),
                         ("gt_pos", gp), ("gt_rot", gr), ("gt_open", go), ("cur_open", opening[t]), ("ee", ee[t])):
                rec[k].append(v)
            ep_abs.append(ee[t, :3] + pos)
            ep_open.append(grip)
        if ep in plot_eps:
            plot_episode(out / f"episode_{ep:04d}.png", ep, ee, opening, ts, ep_abs, ep_open)
        print(f"  ep{ep:3d}  {len(ts):3d} 次  累计 {len(times)}  单次中位 {np.median(times) * 1000:.0f} ms", flush=True)

    r = {k: np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v) for k, v in rec.items()}
    np.savez_compressed(out / "preds.npz", **r)

    pos_m = np.linalg.norm(r["pred_pos"] - r["gt_pos"], axis=-1) * 1000  # (N, H) mm
    pos_h = np.linalg.norm(r["gt_pos"], axis=-1) * 1000
    rot_m = angle_deg(r["pred_rot"], r["gt_rot"])
    rot_h = angle_deg(np.zeros_like(r["gt_rot"]), r["gt_rot"])
    grip_m = np.abs(r["pred_open"] - r["gt_open"])
    grip_h = np.abs(r["cur_open"][:, None] - r["gt_open"])
    end_gt, end_pred = r["gt_pos"][:, -1], r["pred_pos"][:, -1]
    moving = np.linalg.norm(end_gt, axis=1) >= 0.01  # 32 步内真走了 1 cm 以上的点才算方向
    cos = np.sum(end_gt * end_pred, 1) / (np.linalg.norm(end_gt, axis=1) * np.linalg.norm(end_pred, axis=1) + 1e-12)
    in_range = (r["ee"][:, :3] >= pos_lo) & (r["ee"][:, :3] <= pos_hi)
    warm = times[1:] if len(times) > 1 else times
    summary = {
        "scope": "official WALL-OSS-0.5 zero-shot on recorded AR5 observations; offline, no robot",
        "data": str(data), "task": meta.get("prompt"), "episodes": len(eps), "predictions": len(times),
        "stride": a.stride, "seed": a.seed, "norm_key": key, "gripper_map": a.gripper_map,
        "gripper_model_range": [gmin, gmin + gdelta], "action_shape": [H, D], "all_finite": True,
        "euler_convention_max_err": euler_err,
        "model_load_s": load_s, "first_predict_ms": times[0] * 1000,
        "predict_ms_median": float(np.median(warm) * 1000), "predict_ms_p90": float(np.percentile(warm, 90) * 1000),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "state_pos_in_norm_range_xyz": in_range.mean(0).tolist(),
        "pos_err_mm_model": float(pos_m.mean()), "pos_err_mm_hold": float(pos_h.mean()),
        "rot_err_deg_model": float(rot_m.mean()), "rot_err_deg_hold": float(rot_h.mean()),
        "grip_err_model": float(grip_m.mean()), "grip_err_hold": float(grip_h.mean()),
        **{f"pos_err_mm_step{k + 1}": [float(pos_m[:, k].mean()), float(pos_h[:, k].mean())] for k in CHECK_STEPS},
        **{f"rot_err_deg_step{k + 1}": [float(rot_m[:, k].mean()), float(rot_h[:, k].mean())] for k in CHECK_STEPS},
        "end_moving_points": int(moving.sum()),
        "end_dir_cos_median": float(np.median(cos[moving])) if moving.any() else None,
        "end_dir_cos_positive_frac": float((cos[moving] > 0).mean()) if moving.any() else None,
        "end_disp_mm_median_pred": float(np.median(np.linalg.norm(end_pred, axis=1)) * 1000),
        "end_disp_mm_median_gt": float(np.median(np.linalg.norm(end_gt, axis=1)) * 1000),
        "checks": checks,
        "pos_err_mm_by_step_model": pos_m.mean(0).tolist(), "pos_err_mm_by_step_hold": pos_h.mean(0).tolist(),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    plot_error_by_step(out / "error_by_step.png", pos_m.mean(0), pos_h.mean(0), rot_m.mean(0), rot_h.mean(0))

    print(f"\n回放 {len(times)} 次；首次 {summary['first_predict_ms']:.0f} ms，之后中位 {summary['predict_ms_median']:.0f} ms、"
          f"p90 {summary['predict_ms_p90']:.0f} ms；显存峰值 {summary['peak_allocated_gib']:.2f} GiB")
    print(f"欧拉角约定核对最大差 {euler_err:.1e}；当前位置落在 {key} 统计范围内的比例 xyz {np.round(in_range.mean(0), 2)}")
    print(f"{'':14s}{'模型':>10s}{'不动':>10s}")
    print(f"{'位置 mm':14s}{summary['pos_err_mm_model']:10.1f}{summary['pos_err_mm_hold']:10.1f}")
    for k in CHECK_STEPS:
        m, h = summary[f"pos_err_mm_step{k + 1}"]
        print(f"{'  第' + str(k + 1) + '步':14s}{m:10.1f}{h:10.1f}")
    print(f"{'姿态 °':14s}{summary['rot_err_deg_model']:10.2f}{summary['rot_err_deg_hold']:10.2f}")
    print(f"{'夹爪开度':14s}{summary['grip_err_model']:10.3f}{summary['grip_err_hold']:10.3f}")
    if moving.any():
        print(f"第 32 步方向（真走了 ≥1 cm 的 {int(moving.sum())} 个点）：余弦中位 {summary['end_dir_cos_median']:.2f}，"
              f"同向比例 {summary['end_dir_cos_positive_frac']:.2f}；位移中位 预测 {summary['end_disp_mm_median_pred']:.1f} mm"
              f" / 数据 {summary['end_disp_mm_median_gt']:.1f} mm")
    for c in checks:
        print(f"检查 ep{c['ep']} t{c['t']}：同种子重跑最大差 {c['repeat_max_abs']:.2e}；换种子位置平均变 {c['seed_change_mm']:.1f} mm；"
              f"两张图换全黑平均变 {c['blank_images_mm']:.1f} mm")
    print(f"输出：{out}")


if __name__ == "__main__":
    main()

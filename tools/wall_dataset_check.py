"""WALL+RLT 解锁关节数据采完后的检查：先跑 WALL+RLT 自己的 audit.py，再查解锁数据专有的几项。

在 sm_120 推理镜像里跑（镜像里有 /opt/wall-rlt 的 wall_rlt 包、PyAV、scipy），不用 GPU，只读数据。

第一步原样调用 wall_rlt.data.audit.audit_npz，也就是 WALL+RLT 数据准备流程里那一步：格式、二值夹爪指令、
时间线 timestamp == t_cam_front、held 全 False、prompt 一致、上一帧到这一帧的位姿增量、视频帧数 / 30 fps / 640x480。
产物（audit.json、episodes.json、normalization.json、data_contract.inspection.json）和训练准备用的一样。

第二步是 audit.py 不管、但解锁数据要管的：
- 数据根目录有 wall_collection.json（wall_collect.py 写的参数记录）；没有就不是按 WALL 解锁参数采的
- 没有 teleop=cartesian_sdk_trial（--ik sdk 试验数据不能进数据集）
- j3、j5 真的动了：一条里摆动 ≥ 1° 才算。全部都没动 = 锁关节数据，判不通过
- 关节离 AR5 软限位多远、相邻两帧关节最大变化（逆解跳解会表现成一帧突变）、实测帧率
- 各关节、夹爪开度、条长的范围，写数据说明时用

用法（宿主机，source 本轮 session.env 后）：
  mkdir -p "$WALL_ROOT/data-audit"
  docker run --rm --pull=never --network none --user "$(id -u):$(id -g)" $WALL_CONTAINER_ENV \\
    -v <数据根目录>:/target-data:ro -v "$WALL_ROOT/tools:/tools:ro" -v "$WALL_ROOT/data-audit:/out" \\
    "$WALL_INFERENCE_IMAGE_ID" python /tools/wall_dataset_check.py --root /target-data --output /out/<名字>
输出目录已存在就拒绝运行（不覆盖旧结果）。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("WALL_RLT_ROOT", "/opt/wall-rlt"))
from wall_rlt.data.audit import audit_npz, write_json  # noqa: E402

RECORD = "wall_collection.json"
# AR5 软限位，弧度（SO101/数据集格式说明.md 4.1）
LOWER = np.array([-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -0.8727, -0.8727])
UPPER = np.array([3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 0.8727, 0.8727])
MOVING_DEG = 1.0      # j3 或 j5 一条里摆动超过这么多才算解锁
STEP_WARN_RAD = 0.1   # 相邻两帧关节变化超过它就提示（30 Hz 下约 170°/s），可能是逆解跳解
MARGIN_WARN_DEG = 5.0
FPS_TOL = 0.5


def episode_checks(path):
    with np.load(path, allow_pickle=False) as z:
        q, opening = z["state"][:, :7].astype(np.float64), z["state"][:, 7].astype(np.float64)
        return {"frames": len(q), "teleop": str(z["teleop"]) if "teleop" in z.files else None,
                "fps_measured": float(z["fps_measured"]) if "fps_measured" in z.files else None,
                "duration_s": float(z["timestamp"][-1] - z["timestamp"][0]),
                "ptp_deg": np.degrees(np.ptp(q, axis=0)).round(3).tolist(),
                "min_deg": np.degrees(q.min(0)).round(2).tolist(), "max_deg": np.degrees(q.max(0)).round(2).tolist(),
                "limit_margin_deg": np.degrees(np.minimum(q - LOWER, UPPER - q).min(0)).round(2).tolist(),
                "max_step_rad": float(np.abs(np.diff(q, axis=0)).max()) if len(q) > 1 else 0.0,
                "opening_range": [float(opening.min()), float(opening.max())]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="数据根目录：<任务名>/episode_XXXX.npz + 两路 mp4 + dataset.json")
    ap.add_argument("--output", type=Path, required=True, help="结果目录，必须在数据根目录外、且还不存在")
    ap.add_argument("--exclude-group", action="append", default=[], help="不检查的任务目录，可重复")
    ap.add_argument("--seed", type=int, default=20261029, help="audit.py 的划分种子，默认和 WALL+RLT 一样")
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"✗ {a.output} 已存在，换个名字（不覆盖旧结果）")

    audit = audit_npz(a.root, a.output, seed=a.seed, exclude_groups=a.exclude_group)
    root = a.root.resolve()
    manifest = json.loads((a.output / "episodes.json").read_text())
    record_path = root / RECORD
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None

    eps = []
    for entry in manifest["episodes"]:
        try:
            eps.append({"episode_id": entry["episode_id"], "task_group": entry["task_group"], "npz": entry["npz"],
                        **episode_checks(root / entry["npz"])})
        except (OSError, KeyError, ValueError) as exc:  # audit.py 已经记过结构错误，这里只跳过
            eps.append({"episode_id": entry["episode_id"], "npz": entry["npz"], "error": f"{type(exc).__name__}: {exc}"})
    ok = [e for e in eps if "error" not in e]

    issues, warnings = [], []
    if audit["structural_status"] != "PASS" or audit["semantic_status"] != "PASS":
        issues.append(f"WALL+RLT audit.py 没通过：结构 {audit['structural_status']}、语义 {audit['semantic_status']}，见 audit.json")
    if record is None:
        issues.append(f"根目录没有 {RECORD}：不是 wall_collect.py 采的（可能是解锁前、已作废的数据）")
    sdk = [e["npz"] for e in ok if e["teleop"]]
    if sdk:
        issues.append(f"{len(sdk)} 条带 teleop 标记（--ik sdk 试验数据），不能进数据集：{sdk[:5]}")
    moving = [e for e in ok if max(e["ptp_deg"][2], e["ptp_deg"][4]) >= MOVING_DEG]
    if ok and not moving:
        issues.append(f"j3、j5 在所有 {len(ok)} 条里摆动都不到 {MOVING_DEG}°：是锁关节的数据")
    elif len(moving) < len(ok):
        warnings.append(f"{len(ok) - len(moving)} 条 j3、j5 摆动都不到 {MOVING_DEG}°："
                        f"{[e['npz'] for e in ok if e not in moving][:10]}")
    jumps = [(e["npz"], round(e["max_step_rad"], 3)) for e in ok if e["max_step_rad"] > STEP_WARN_RAD]
    if jumps:
        warnings.append(f"{len(jumps)} 条相邻两帧关节变化超过 {STEP_WARN_RAD} rad（可能是逆解跳解）：{jumps[:10]}")
    near = [(e["npz"], min(e["limit_margin_deg"])) for e in ok if min(e["limit_margin_deg"]) < MARGIN_WARN_DEG]
    if near:
        warnings.append(f"{len(near)} 条有关节离软限位不到 {MARGIN_WARN_DEG}°：{near[:10]}")
    fps_off = [(e["npz"], e["fps_measured"]) for e in ok if e["fps_measured"] and abs(e["fps_measured"] - 30) > FPS_TOL]
    if fps_off:
        warnings.append(f"{len(fps_off)} 条实测帧率偏离 30 超过 {FPS_TOL}：{fps_off[:10]}")

    def agg(key, fn):
        return fn(np.array([e[key] for e in ok]), axis=0).round(2).tolist() if ok else None
    summary = {
        "scope": "WALL+RLT audit.py + 解锁关节数据专有检查；只读原始 npz / mp4，不碰硬件",
        "root": str(root), "status": "PASS" if not issues else "ISSUES", "issues": issues, "warnings": warnings,
        "wall_audit": {k: audit[k] for k in ("structural_status", "semantic_status", "episodes", "frames", "groups",
                                             "split_counts")},
        "collection_record": {k: record[k] for k in ("created", "purpose", "fixed_args", "settings")} | {
            "sessions": len(record["sessions"])} if record else None,
        "episodes_checked": len(ok), "episodes_j3_or_j5_moving": len(moving),
        "joint_min_deg": agg("min_deg", np.min), "joint_max_deg": agg("max_deg", np.max),
        "joint_ptp_deg_median": agg("ptp_deg", np.median),
        "limit_margin_deg_min": agg("limit_margin_deg", np.min),
        "max_step_rad": max((e["max_step_rad"] for e in ok), default=None),
        "opening_range": [min(e["opening_range"][0] for e in ok), max(e["opening_range"][1] for e in ok)] if ok else None,
        "frames_per_episode": [min(e["frames"] for e in ok), int(np.median([e["frames"] for e in ok])),
                               max(e["frames"] for e in ok)] if ok else None,
        "fps_measured_range": [min((e["fps_measured"] for e in ok if e["fps_measured"]), default=None),
                               max((e["fps_measured"] for e in ok if e["fps_measured"]), default=None)],
        "episodes": eps,
    }
    write_json(a.output / "wall_check.json", summary)

    print(f"WALL+RLT audit.py：结构 {audit['structural_status']}、语义 {audit['semantic_status']}；"
          f"{audit['episodes']} 条 / {audit['frames']} 帧；分组 {audit['groups']}")
    print(f"参数记录 {RECORD}：{'有，' + str(len(record['sessions'])) + ' 次会话' if record else '没有'}")
    print(f"j3 或 j5 摆动 ≥ {MOVING_DEG}° 的条：{len(moving)} / {len(ok)}；"
          f"各关节摆动中位（°）{summary['joint_ptp_deg_median']}")
    print(f"关节范围（°）min {summary['joint_min_deg']}\n             max {summary['joint_max_deg']}")
    print(f"离软限位最近（°）{summary['limit_margin_deg_min']}；相邻帧关节最大变化 {summary['max_step_rad']} rad；"
          f"夹爪开度 {summary['opening_range']}；条长（帧）min/中位/max {summary['frames_per_episode']}")
    for w in warnings:
        print(f"⚠ {w}")
    for i in issues:
        print(f"✗ {i}")
    print(f"结论 {summary['status']}；输出 {a.output}")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())

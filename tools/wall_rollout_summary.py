"""一批 wall_execute rollout 的用时汇总，和 bolt2 示教对比。

读 <批次目录>/trial_*/（ticks.npz、run.json，有录像时还有 cam_*.mp4）和 results.tsv（次号、1/0、备注），
写 summary.csv / summary.md 到批次目录。

时间从执行开始（第二次回车后第一拍）算：
  抓取 = 夹爪指令第一次合拢；松手 = 抓取后第一次张开（bolt：螺栓已插进套筒）；插入用时 = 松手 − 抓取。
示教的同样三项来自 --demo（逐条复查的 json：grasp_s / release_s / dur_s，从开始录制算），
默认剔掉训练时不用的 24、36、57。

  python tools/wall_rollout_summary.py ~/wall_exec_logs/bolt2_30k_sdk_baseline_20261113
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import numpy as np


def trial_row(d: Path, res: dict) -> dict:
    i = d.name.split("_", 1)[1]
    z, run = np.load(d / "ticks.npz"), json.loads((d / "run.json").read_text(encoding="utf-8"))
    t = z["t"] - z["t"][0]
    g = z["grip_cmd"]
    closes = [float(t[k]) for k in range(1, len(g)) if g[k - 1] >= .5 > g[k]]
    opens = [float(t[k]) for k in range(1, len(g)) if g[k - 1] < .5 <= g[k]]
    grasp = closes[0] if closes else None
    release = next((o for o in opens if grasp is not None and o > grasp), None)
    video = run.get("video") or {}
    cams = video.get("cams", {})
    r = res.get(i, [])
    return {"trial": i, "success": r[1] if len(r) > 1 else "", "note": r[2] if len(r) > 2 else "",
            "start": run["created"], "grasp_s": grasp, "release_s": release,
            "insert_s": None if release is None else release - grasp,
            "run_s": float(t[-1]), "hit_max_seconds": bool(t[-1] >= 59.9),
            "grip_toggles_after_release": max(0, len(closes) - 1),
            "chunks": run.get("chunks"), "lag_waited_s": (run.get("lag_stats") or {}).get("waited_s"),
            "switch_jumps": run.get("switch_jump_warnings"), "ik": run.get("ik"), "obs_align": bool(run.get("obs_align")),
            "video": " ".join(f"{k}:{c['frames']}帧/丢{c['dropped']}" for k, c in cams.items()) if cams else "无"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", type=Path)
    ap.add_argument("--demo", type=Path, default=Path.home() / "wall_exec_logs/bolt2_demo_timing.json")
    ap.add_argument("--demo-drop", default="episode_0024,episode_0036,episode_0057")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    res_path = a.batch / "results.tsv"
    res = {r[0]: r for r in csv.reader(open(res_path, encoding="utf-8"), delimiter="\t")} if res_path.exists() else {}
    rows = [trial_row(d, res) for d in sorted(a.batch.glob("trial_*")) if (d / "ticks.npz").exists()]
    if not rows:
        raise SystemExit(f"{a.batch} 下没有 trial_*/ticks.npz")
    with open(a.batch / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fmt = lambda v: "-" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))  # noqa: E731
    ok = {"1": "成功", "0": "失败"}
    md = [f"# {a.title or a.batch.name}", "",
          "时间从执行开始（第二次回车后第一拍）算；抓取 = 夹爪指令第一次合拢，松手 = 抓取后第一次张开；"
          "插入用时 = 松手 − 抓取。", "",
          "| 次 | 结果 | 抓取 s | 松手 s | 插入用时 s | 整条 s | 松手后夹爪又合 | 等臂累计 s | 视频 | 备注 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['trial']} | {ok.get(r['success'], r['success'] or '-')} | {fmt(r['grasp_s'])} | "
                  f"{fmt(r['release_s'])} | {fmt(r['insert_s'])} | {fmt(r['run_s'])}"
                  f"{'（到 60 s 上限）' if r['hit_max_seconds'] else ''} | {r['grip_toggles_after_release']} 次 | "
                  f"{fmt(r['lag_waited_s'])} | {r['video']} | {r['note'] or ''} |")
    succ = [r for r in rows if r["success"] == "1" and r["release_s"] is not None]
    if succ:
        md += ["", f"成功 {len(succ)}/{len(rows)}；成功的松手用时中位 {st.median(r['release_s'] for r in succ):.1f} s，"
                   f"插入用时中位 {st.median(r['insert_s'] for r in succ):.1f} s。"]
    if a.demo.exists():
        drop = set(a.demo_drop.split(",")) if a.demo_drop else set()
        demo = [e for e in json.loads(a.demo.read_text(encoding="utf-8")) if e["ep"] not in drop and e.get("release_s")]
        md += ["", f"示教（{len(demo)} 条，从开始录制算）：", "", "| | 最短 | 中位 | 最长 |", "|---|---|---|---|"]
        series = {"抓取 s": [e["grasp_s"] for e in demo], "松手 s": [e["release_s"] for e in demo],
                  "插入用时 s": [e["release_s"] - e["grasp_s"] for e in demo], "整条 s": [e["dur_s"] for e in demo]}
        for name, v in series.items():
            md.append(f"| {name} | {min(v):.1f} | {st.median(v):.1f} | {max(v):.1f} |")
    (a.batch / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

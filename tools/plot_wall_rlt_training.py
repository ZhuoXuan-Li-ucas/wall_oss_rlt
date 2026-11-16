"""Loss curves for a WALL-RLT Stage 1 run, plus the dev comparison of its checkpoints.

Reads the tensorboard events that RLinf writes during training
(``<project>/outputs/wall-rlt/tensorboard/``: ``train/loss``, ``train/vla_loss``,
``train/rlt_loss``, ``train/learning_rate``, ``train/grad_norm`` per step) and,
optionally, the per-checkpoint dev evaluations (``step_<N>.json`` with an
``aggregate`` block, the format of the 11-05 bolt comparison).

Writes, into --out:
  training_curves.png   training losses (raw + moving average, log scale), lr, grad norm,
                        dashed lines at the checkpoints
  dev_checkpoints.png   dev metrics per checkpoint (only with --eval-dir)
  summary.csv / .md     per checkpoint: train losses averaged over the last --window steps
                        before it, and the dev metrics

Needs only tensorboard and matplotlib (both in the training overlay .venv-training).
It reads files only; it never touches checkpoints.

  python plot_wall_rlt_training.py --tb-dir outputs/wall-rlt/tensorboard \
      --eval-dir logs/checkpoint-comparison-dev --checkpoints 10000 20000 30000 40000 --out logs/loss-curves
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

LOSS_TAGS = ("train/loss", "train/vla_loss", "train/rlt_loss")
EXTRA_TAGS = ("train/learning_rate", "train/grad_norm")
DEV_KEYS = (("vla_loss", "dev VLA (action) loss"), ("rlt_loss", "dev RLT reconstruction loss"),
            ("position_mean_mm", "position error mean (mm)"), ("rotation_mean_deg", "rotation error mean (deg)"),
            ("gripper_error_rate", "gripper error rate"))


def load_scalars(tb_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All scalar series from every event file under tb_dir; later files win on duplicate steps
    (a resumed run re-logs from its checkpoint)."""
    files = sorted(tb_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no tensorboard event files under {tb_dir}")
    series: dict[str, dict[int, float]] = {}
    for f in files:
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            d = series.setdefault(tag, {})
            for e in acc.Scalars(tag):
                d[int(e.step)] = float(e.value)
    out = {}
    for tag, d in series.items():
        steps = np.array(sorted(d), dtype=np.int64)
        out[tag] = (steps + 1, np.array([d[s] for s in sorted(d)], dtype=float))   # RLinf logs step index 0-based
    print(f"{len(files)} event file(s); tags: {sorted(out)}")
    return out


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < 2 or window <= 1:
        return y
    w = min(window, len(y))
    c = np.cumsum(np.insert(y, 0, 0.0))
    head = c[1:w] / np.arange(1, w)                       # growing window at the start
    tail = (c[w:] - c[:-w]) / w
    return np.concatenate([head, tail])


def load_dev(eval_dir: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(eval_dir.glob("step_*.json")):
        d = json.loads(f.read_text())
        step = int(d.get("checkpoint_step") or f.stem.split("_")[1])
        agg = d.get("aggregate", d)
        out[step] = {k: agg.get(k) for k, _ in DEV_KEYS} | {"task_macro_vla_loss": d.get("task_macro_vla_loss")}
    return out


def plot_training(s, checkpoints, window, path: Path, title: str):
    tags = [t for t in LOSS_TAGS if t in s]
    extra = [t for t in EXTRA_TAGS if t in s]
    n = len(tags) + len(extra)
    if not n:
        raise SystemExit(f"none of {LOSS_TAGS + EXTRA_TAGS} found")
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, tag in zip(axes, tags + extra):
        x, y = s[tag]
        ax.plot(x, y, color="0.75", lw=0.4, label="per step")
        ax.plot(x, moving_average(y, window), color="C0", lw=1.6, label=f"moving average ({window} steps)")
        if tag in LOSS_TAGS and (y > 0).all():
            ax.set_yscale("log")
        for c in checkpoints:
            ax.axvline(c, color="C3", ls="--", lw=0.8)
        ax.set_ylabel(tag.replace("train/", ""))
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(title)
    axes[-1].set_xlabel("training step (dashed = saved checkpoints)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dev(dev, path: Path, title: str):
    steps = sorted(dev)
    fig, axes = plt.subplots(1, len(DEV_KEYS), figsize=(3.3 * len(DEV_KEYS), 3.2))
    for ax, (key, label) in zip(axes, DEV_KEYS):
        v = [dev[s].get(key) for s in steps]
        ok = [(s, x) for s, x in zip(steps, v) if x is not None]
        if ok:
            ax.plot([s for s, _ in ok], [x for _, x in ok], "o-", color="C1")
            best = min(ok, key=lambda t: t[1])
            ax.plot([best[0]], [best[1]], "*", ms=14, color="C2")
        ax.set_title(label, fontsize=9)
        ax.set_xticks(steps)
        ax.set_xticklabels([f"{s // 1000}k" for s in steps])
        ax.grid(alpha=0.3)
    fig.suptitle(title + "  (star = lowest)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_rows(s, dev, checkpoints, window):
    rows = []
    for c in checkpoints:
        row = {"step": c}
        for tag in LOSS_TAGS:
            if tag in s:
                x, y = s[tag]
                m = (x > c - window) & (x <= c)
                row[f"train_{tag.split('/')[1]}_last{window}"] = float(y[m].mean()) if m.any() else None
        for key, _ in DEV_KEYS:
            row[f"dev_{key}"] = dev.get(c, {}).get(key)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tb-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, help="directory with step_<N>.json dev evaluations")
    ap.add_argument("--checkpoints", type=int, nargs="*", default=None,
                    help="checkpoint steps (default: the steps found in --eval-dir)")
    ap.add_argument("--window", type=int, default=500, help="moving-average window in steps")
    ap.add_argument("--title", default="WALL-RLT stage 1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = load_scalars(a.tb_dir)
    dev = load_dev(a.eval_dir) if a.eval_dir else {}
    checkpoints = sorted(a.checkpoints if a.checkpoints else dev)
    last = max(int(v[0][-1]) for v in s.values())
    print(f"logged steps up to {last}; checkpoints {checkpoints}")
    plot_training(s, checkpoints, a.window, a.out / "training_curves.png", a.title)
    if dev:
        plot_dev(dev, a.out / "dev_checkpoints.png", a.title + " - dev split")
    rows = summary_rows(s, dev, checkpoints, a.window)
    if rows:
        keys = list(rows[0])
        with open(a.out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        fmt = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
        md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        md += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
        (a.out / "summary.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()

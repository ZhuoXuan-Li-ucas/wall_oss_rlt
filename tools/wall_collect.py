"""WALL+RLT 解锁关节数据的采集入口：固定参数调 SO101 的 collect.py，记下参数和代码版本。

2026-10-31 用户定：解锁前采的数据（j3、j5 锁死的 top_error 等）全部作废，新数据用笛卡尔遥操、7 个轴全动。
这里**不复制采集代码**（工作区约定 AR5 驱动只有 atom_robot 一份），只做四件事：

1. 固定会改变数据含义的参数：OmniPicker、笛卡尔、weighted 逆解、位置 x0.8、转动 x0.2、姿态权重 0.5
   （11-02 用户真机调定：转动 0.4 时腕两轴顶限位、姿态怪，0.3 撞软限位后抱死；位置 0.6 嫌小），
   夹爪自转 x0.6（11-03 用户要夹爪能转大角度：主臂 wrist_roll 单独直接给夹爪自身轴，不乘 0.2；
   不单独算时拧主臂手腕大半变成从臂夹爪「歪」。先给 1:1，用户真机试过 0.7 后定 0.6），再加 --preview 开预览窗口。
   11-03 加自转之前在 ~/atom_episodes/wall_unlocked 采的 2 条是另一套参数，本入口不再往那儿续。起始姿势用 `config.py` 的 cartesian_centre_joints（夹爪斜向下），
   指尖偏移、控制和录制频率也用 `config.py` 当时的值（130 mm、50 Hz / 30 Hz）。
   这些参数不许从命令行改，只放行不影响数据含义的几项（条数、时长上限、深度、诊断记录、
   夹爪力和速度初值、主臂串口和标定名）。
2. 每次启动先跑 `collect.py --show-config`（不碰硬件），把生效参数记进数据根目录的
   `wall_collection.json`。同一个根目录参数变了就拒绝：一个数据集只能有一套参数，
   否则同样的画面会对应不同的关节标签。要换参数就换一个 --out。
3. 不往已有旧数据的目录里写：根目录下已经有 episode 但没有 `wall_collection.json` 的，视为解锁前的数据，拒绝。
4. 记下 DexGraspVLA- 仓库的提交号、SO101 下没提交的改动（文件名 + diff 的 sha256），
   以后能查出这批数据是哪版采集代码采的。

npz / mp4 / dataset.json 的格式和原来完全一样，WALL+RLT 的 audit.py 直接能读。采完用同目录的
wall_dataset_check.py 检查。

用法（在哪个目录跑都行；按键和流程与 collect.py 的 OmniPicker 模式相同）：
  python3 tools/wall_collect.py --dry-run                  # 只打印命令和生效参数，不写文件、不碰硬件
  python3 tools/wall_collect.py                            # 存到默认的 ~/atom_episodes/wall_unlocked_v2/<任务名>/
  python3 tools/wall_collect.py --out <新根目录> -n 50

找工作区根目录的方法和工作区约定一样：从本文件往上找同时有 算法/ 和 数据_硬件/ 的目录，或者设环境变量 YUHANG_WS。
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RECORD = "wall_collection.json"
FIXED_ARGS = ["--end-effector", "omnipicker", "--space", "cartesian", "--ik", "weighted",
              "--scale", "0.8", "--rot-scale", "0.2", "--spin-scale", "0.6", "--ori-cost", "0.5", "--preview"]
# 放行的 collect.py 参数：不影响数据含义。值 = 是否带一个参数值
PASSTHROUGH = {"-n": True, "--num-episodes": True, "--episode-time": True,
               "--depth": False, "--cart-log": False, "--omnipicker-force": True, "--omnipicker-speed": True,
               "--leader-port": True, "--calib-id": True}
# --show-config 输出里决定数据含义的行，按行首标签取；同一个根目录必须完全一样
SETTING_LABELS = ("控制频率", "录制帧率", "遥操空间", "逆解", "映射", "主臂滤波", "相机", "手", "主臂", "臂", "视频",
                  "OmniPicker 张开上限", "起始姿势")
# 这几行 collect 只在配了才打印（起始姿势：config.cartesian_centre_joints 为 None 时没有），缺了不算格式变了
OPTIONAL_LABELS = ("起始姿势",)
DEFAULT_OUT = Path.home() / "atom_episodes" / "wall_unlocked_v2"   # v1 是加夹爪自转之前那套参数
DEPTH_MB_PER_EPISODE = 225  # SO101 数据集格式说明 4.7：600 帧一条约 225 MB


def workspace_root():
    env = os.environ.get("YUHANG_WS")
    if env:
        return Path(env).expanduser().resolve()
    for d in Path(__file__).resolve().parents:
        if (d / "算法").is_dir() and (d / "数据_硬件").is_dir():
            return d
    raise SystemExit("找不到工作区根目录（同时有 算法/ 和 数据_硬件/ 的目录），设环境变量 YUHANG_WS")


def split_passthrough(extra):
    """只放行 PASSTHROUGH 里的参数，别的一律拒绝并说明原因。"""
    out, i = [], 0
    while i < len(extra):
        flag = extra[i]
        name = flag.split("=", 1)[0]
        if name not in PASSTHROUGH:
            raise SystemExit(f"✗ 不放行 {flag}：WALL 采集的参数是固定的（{' '.join(FIXED_ARGS)}，其余用 config.py），"
                             f"能加的只有 {', '.join(PASSTHROUGH)}")
        out.append(flag)
        if PASSTHROUGH[name] and "=" not in flag:
            if i + 1 >= len(extra):
                raise SystemExit(f"✗ {flag} 后面要跟一个值")
            out.append(extra[i + 1])
            i += 1
        i += 1
    return out


def settings_from(show_config):
    """从 --show-config 的输出里取决定数据含义的行：{标签: 整行}。"""
    found = {}
    for line in show_config.splitlines():
        s = line.strip()
        if s.startswith("OmniPicker "):
            # 这一行的夹持力、速度只是初值（每个任务实际用的在 dataset.json）；张开上限决定「全开」记成多少
            found.update({"OmniPicker 张开上限": t.split("（")[0] for t in s.split() if t.startswith("张开上限=")})
            continue
        for label in SETTING_LABELS:
            if s.startswith(label + " ") and label not in found:
                # --cart-log 是放行的诊断开关，collect 把它打在逆解那行末尾；不算数据含义，去掉再比
                found[label] = " ".join(s.split()).replace(" 逐帧诊断记录 开", "")
    missing = [l for l in SETTING_LABELS if l not in found and l not in OPTIONAL_LABELS]
    if missing:
        raise SystemExit(f"✗ --show-config 的输出里找不到 {missing}，collect.py 的打印格式可能改了，先核对再采")
    return found


def git_state(repo):
    def run(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    head = (run("rev-parse", "HEAD") or "").strip() or None
    dirty = [l[3:] for l in (run("status", "--porcelain", "--", "SO101") or "").splitlines()]
    diff = run("diff", "HEAD", "--", "SO101") or ""
    return {"repo": str(repo), "head": head, "dirty_so101_files": dirty,
            "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}


def check_root(out, settings):
    """返回已有记录（没有就 None）；旧数据目录或参数对不上就退出。"""
    record_path = out / RECORD
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        old = record["settings"]
        diff = [k for k in SETTING_LABELS if old.get(k) != settings.get(k)]
        if diff:
            lines = "\n".join(f"    {k}\n      原来：{old.get(k)}\n      现在：{settings.get(k)}" for k in diff)
            raise SystemExit(f"✗ {out} 已经按另一套参数采过，不能混成一个数据集：\n{lines}\n"
                             f"  要按新参数采就换一个 --out；要接着采就先把 config.py 改回原来的值")
        return record
    old_eps = sorted(out.glob("*/episode_*.npz"))
    if old_eps:
        raise SystemExit(f"✗ {out} 里已经有 {len(old_eps)} 条数据，但没有 {RECORD}，不是本入口采的"
                         f"（可能是解锁前、已作废的数据），不往里写。换一个 --out")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False,
                                 epilog=f"其余参数原样交给 collect.py，只放行：{', '.join(PASSTHROUGH)}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"数据根目录，每个任务一个子目录（默认 {DEFAULT_OUT}）")
    ap.add_argument("--python", default=os.environ.get("WALL_COLLECT_PYTHON",
                                                       str(Path.home() / "miniforge3/envs/dexgraspvla/bin/python")))
    ap.add_argument("--dry-run", action="store_true", help="只打印命令和生效参数，不写文件、不碰硬件")
    a, extra = ap.parse_known_args()

    ws = workspace_root()
    repo = ws / "算法" / "DexGraspVLA-"
    so101 = repo / "SO101"
    if not (so101 / "atom_robot" / "ar5" / "collect.py").is_file():
        raise SystemExit(f"✗ 找不到 {so101}/atom_robot/ar5/collect.py")
    if not Path(a.python).is_file():
        raise SystemExit(f"✗ 找不到 Python {a.python}（采集要用 dexgraspvla 环境，用 --python 或 WALL_COLLECT_PYTHON 指定）")
    out = a.out.expanduser().resolve()
    cmd = [a.python, "-m", "atom_robot.ar5.collect", *FIXED_ARGS, "--out", str(out), *split_passthrough(extra)]
    env = {**os.environ, "PYTHONPATH": ""}  # 本机 ~/.bashrc 加载的 ROS 会污染 conda 环境

    shown = subprocess.run([*cmd, "--show-config"], cwd=so101, env=env, capture_output=True, text=True)
    if shown.returncode != 0:
        raise SystemExit(f"✗ collect.py --show-config 失败（exit {shown.returncode}）：\n{shown.stdout}{shown.stderr}")
    show_text = shown.stdout + shown.stderr
    settings = settings_from(show_text)
    print(show_text.rstrip())
    print(f"\n  WALL 采集：工作目录 {so101}\n  命令 PYTHONPATH= {' '.join(cmd)}")
    if "--depth" in extra:
        free_gb = shutil.disk_usage(out if out.exists() else out.parent).free / 2**30
        print(f"  ⚠ --depth 每条约 {DEPTH_MB_PER_EPISODE} MB（600 帧），这块盘还剩 {free_gb:.0f} GB；WALL+RLT 训练不用深度")

    record = check_root(out, settings) if out.exists() else None
    if a.dry_run:
        print(f"  （--dry-run：没写 {out / RECORD}，没启动采集）")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    record = record or {"schema_version": 1, "created": now, "purpose": "WALL+RLT 解锁关节数据：笛卡尔遥操、weighted 逆解、7 个轴全动",
                        "fixed_args": FIXED_ARGS, "settings": settings, "sessions": []}
    record["sessions"].append({"started": now, "argv": cmd[1:], "collect_git": git_state(repo),
                               "show_config": show_text})
    tmp = out / (RECORD + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(out / RECORD)
    print(f"  参数和代码版本记在 {out / RECORD}（第 {len(record['sessions'])} 次会话）\n")
    sys.stdout.flush()
    os.chdir(so101)
    os.execve(cmd[0], cmd, env)  # 交出终端，按键直接到 collect.py


if __name__ == "__main__":
    sys.exit(main())

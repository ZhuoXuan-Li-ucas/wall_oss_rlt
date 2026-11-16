"""顶部相机（cam_high，Gemini 335）复位：实时对准旧数据集 wall_unlocked_v2 的画面，边看边拧相机支架。

11-10 顶部相机位置变了，要让它回到采集时的位置（数据集和训练好的模型都按那个视角）。旧数据只有 RGB 视频，
这里的参照 = 若干条 episode 第 0 帧（臂都在采集起始姿势）的逐像素中位图。150 条第 0 帧彼此只差 ≤1.6 px、≤0.2°，
所以复位目标就是这个量级。

每帧做两件事：
1. 2D：参照图和实时画面 SIFT 匹配 + RANSAC 相似变换 → 画面要挪多少像素、转多少度、缩放多少。直接对应看到的画面。
2. 3D（点云）：实时画面的匹配点用对齐到彩色的实时深度（335 出厂对齐，1 mm/单位）抬成相机坐标系里的点，
   和参照图里的对应像素做 PnP（RANSAC + LM 精修）→ 采集时相机相对现在相机的位姿 = 相机要平移多少 mm、转多少度。
   内参用相机自报的（OBSENSOR，640x480）；参照和实时是同一台相机、同一分辨率。
取最近 5 帧的中位数显示。对准判据（11-10 用户定 1 px）：中心偏移 < 1 px、转角 < 0.1°、缩放误差 < 0.15%，
内点 ≥ 15，连续 10 帧；对准时窗口底部出现绿色 ALIGNED、终端响一声提示，并自动存一张 *_aligned 快照。

场景变了怎么办（台面上东西挪过、加了杂物）：匹配只该用没动过的参照物。
* 先让臂回采集起始姿势（参照帧里臂就在那儿；臂座钉在台上，回零后的夹爪就是很好的 3D 参照物）：
    cd ~/yuhang_workspace/算法/DexGraspVLA-/SO101 && PYTHONPATH= ~/miniforge3/envs/dexgraspvla/bin/python -m atom_robot.ar5.unseize --recover --home
* --select-roi：在参照图上框出确定没动过的区域（胶带、台面边、回零的夹爪……），可以框多块，存进 --out，下次自动用；
  不框就用整张图。
* 叠加的参照轮廓（默认绿色）不依赖匹配，特征点少的时候直接用眼睛把胶带 / 板边对上。

相机坐标系（3D 读数）：x = 画面右，y = 画面下，z = 镜头朝向（顶部相机就是朝下往台面）。
  "move" = 相机光心要往哪挪（mm）；"rot" = 绕这三个轴要转多少度（右手定则）。
  画面效果：绕 y 转 ≈ 画面左右平移，绕 x 转 ≈ 画面上下平移，绕 z 转 = 画面旋转；沿 z 挪 = 画面缩放。

窗口左边是实时画面 + 对准叠加，右边是数据集画面（11-10 用户：只复位 bolt 场景 —— 螺栓座、套筒、白板照着摆）。
窗口按键（先点一下窗口）：o 换叠加方式（默认半透明，调到夹爪没有重影；还有轮廓 / 差分 / 只看参照）  s 存一张快照到 --out  q 或 Esc 退出；
--tasks 给了多个任务时 t 切换右边的数据集画面（对准参照也跟着换）。
窗口里只有 ASCII（cv2 没中文字形），中文读数每秒在终端打一行。

用法（dexgraspvla 环境；不能和采集、执行器同时开相机）：
  PY=~/miniforge3/envs/dexgraspvla/bin/python
  $PY tools/wall_camera_align.py --select-roi      # 重新框参照物（每块按空格/回车确认，框完 Esc，再在预览里回车存）
  $PY tools/wall_camera_align.py --add-roi         # 在存下的框上接着加
  $PY tools/wall_camera_align.py                   # 之后：直接用存下的框
"""
import argparse
import datetime
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

CAM = "cam_high"
W, H = 640, 480
# 对准判据（11-10 用户：调到 1 px 以内）：中心偏移 < 1 px；转角 0.1°、缩放 0.15% 在画面边缘也只差约 0.5 px。
# 读数要可信：至少 15 个内点；连续 10 帧（约 1 s）都满足才算，单帧碰巧落进去不算
OK_SHIFT_PX, OK_ROT_DEG, OK_SCALE, OK_MIN_INLIERS, OK_FRAMES = 1.0, 0.1, 0.0015, 15, 10
OVERLAYS = ("edges", "blend", "diff", "reference")


def workspace_root():
    for d in Path(__file__).resolve().parents:
        if (d / "算法").is_dir() and (d / "数据_硬件").is_dir():
            return d
    raise SystemExit("找不到工作区根目录（同时有 算法/ 和 数据_硬件/ 的目录）")


def task_reference(data, task, episodes=None):
    """某个任务若干条 episode 第 0 帧的中位图（BGR）+ 说明。episodes = "40-49"，不给就取最后 10 条。"""
    files = sorted((Path(data).expanduser() / task).glob(f"episode_*_{CAM}.mp4"))
    if episodes:
        first, last = (int(x) for x in episodes.split("-"))
        files = [f for f in files if first <= int(f.name[8:12]) <= last]
    else:
        files = files[-10:]
    frames = []
    for f in files:
        cap = cv2.VideoCapture(str(f))
        ok, frame = cap.read()
        cap.release()
        if ok:
            frames.append(frame)
    if not frames:
        raise SystemExit(f"✗ {Path(data).expanduser() / task} 里没有可用的 {CAM} 视频")
    span = f"{files[0].name[8:12]}-{files[-1].name[8:12]}"
    return np.median(np.stack(frames), axis=0).astype(np.uint8), task, span, len(frames)


def roi_mask(rects):
    if not rects:
        return None
    mask = np.zeros((H, W), np.uint8)
    for x, y, w, h in rects:
        mask[y:y + h, x:x + w] = 255
    return mask


def open_camera():
    """只开顶部相机 + 对齐深度，读相机自报的内参。"""
    so101 = workspace_root() / "算法" / "DexGraspVLA-" / "SO101"
    sys.path.insert(0, str(so101))
    cwd = os.getcwd()
    os.chdir(so101)
    try:
        from atom_robot.ar5.config import AR5Config
        from atom_robot.ar5.drivers.cameras import OpenCVCameras
        cfg = AR5Config().cameras
        cfg.sources = {CAM: cfg.sources[CAM]}
        cfg.depth_sources = (CAM,)
        cams = OpenCVCameras(cfg)
        cams.connect()
    finally:
        os.chdir(cwd)
    info = cams.depth_info.get(CAM, {})
    if not info.get("aligned_to_color"):
        cams.disconnect()
        raise SystemExit(f"✗ {CAM} 的深度没对齐彩色（{info}），3D 读数会错")
    cap = cams._grabbers[CAM]._cap
    K = np.array([[cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FX), 0, cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_CX)],
                  [0, cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FY), cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_CY)],
                  [0, 0, 1.0]])
    if not np.isfinite(K).all() or K[0, 0] <= 0:
        cams.disconnect()
        raise SystemExit("✗ 相机没报内参")
    return cams, K, float(info["mm_per_unit"]) / 1000.0


class Estimator:
    def __init__(self, ref, mask, K, m_per_unit):
        self.sift = cv2.SIFT_create(4000)
        self.matcher = cv2.BFMatcher()
        self.K, self.m_per_unit = K, m_per_unit
        self.ref_kp, self.ref_des = self.sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), mask)

    def depth_at(self, depth, pts):
        """每个点 3x3 邻域里非零深度的中位数（米），没有就 0。"""
        out = np.zeros(len(pts))
        for i, (u, v) in enumerate(np.round(pts).astype(int)):
            patch = depth[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2]
            patch = patch[patch > 0]
            if patch.size:
                out[i] = float(np.median(patch)) * self.m_per_unit
        return out

    def __call__(self, live, depth):
        result = {"matches": 0}
        kp, des = self.sift.detectAndCompute(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), None)
        if des is None or self.ref_des is None or len(kp) < 8:
            return result
        good = [a for a, b in self.matcher.knnMatch(self.ref_des, des, k=2) if a.distance < 0.75 * b.distance]
        result["matches"] = len(good)
        if len(good) < 8:
            return result
        p_ref = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        p_live = np.float32([kp[m.trainIdx].pt for m in good])
        # 2D：实时 → 参照 的相似变换；画面中心要挪多少、转多少、缩放多少
        A, inl = cv2.estimateAffinePartial2D(p_live, p_ref, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if A is not None and inl.sum() >= 6:
            c = np.array([W / 2, H / 2])
            result.update(inliers2d=int(inl.sum()), shift=A[:, :2] @ c + A[:, 2] - c,
                          rot=float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), scale=float(np.hypot(A[0, 0], A[1, 0])),
                          inlier_pts=p_live[inl.ravel() == 1])
        # 3D：实时点云里的匹配点 ↔ 参照图像素，PnP 解出采集时相机相对现在相机的位姿
        z = self.depth_at(depth, p_live)
        ok = (z > 0.2) & (z < 3.0)
        if ok.sum() >= 8:
            K = self.K
            X = np.stack([(p_live[:, 0] - K[0, 2]) * z / K[0, 0], (p_live[:, 1] - K[1, 2]) * z / K[1, 1], z], 1)[ok]
            found, rvec, tvec, pin = cv2.solvePnPRansac(X, p_ref[ok], K, None, reprojectionError=2.0,
                                                        iterationsCount=500, flags=cv2.SOLVEPNP_SQPNP)
            if found and pin is not None and len(pin) >= 12:        # 内点太少的 3D 解会离谱（11-10 实测 7 点解出 134 mm）
                idx = pin.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(X[idx], p_ref[ok][idx], K, None, rvec, tvec)
                R, _ = cv2.Rodrigues(rvec)
                proj, _ = cv2.projectPoints(X[idx], rvec, tvec, K, None)
                result.update(inliers3d=len(idx),
                              move_mm=(-R.T @ tvec.ravel()) * 1000,                  # 参照相机光心，在现在相机坐标系里
                              rot_deg=np.degrees(cv2.Rodrigues(R.T)[0].ravel()),     # 现在相机转到参照相机
                              reproj_px=float(np.median(np.linalg.norm(proj.reshape(-1, 2) - p_ref[ok][idx], axis=1))))
        return result


def directions(move, rot):
    """3D 读数换成拧支架的说法。正负号约定按合成数据核对过（已知偏移渲染 → 还原）：
    绕 x 正 = 光轴往画面上方偏，绕 y 正 = 光轴往画面右侧偏，绕 z 正 = 顺着镜头看顺时针（画面内容逆时针转）。"""
    x, y, z = move
    rx, ry, rz = rot
    return (f"挪：{'右' if x >= 0 else '左'} {abs(x):.1f} / {'下' if y >= 0 else '上'} {abs(y):.1f} / "
            f"{'靠近台面' if z >= 0 else '远离台面'} {abs(z):.1f} mm；"
            f"转：顺着镜头看{'顺' if rz >= 0 else '逆'}时针 {abs(rz):.2f}° / 镜头朝画面{'上方抬' if rx >= 0 else '下方压'} "
            f"{abs(rx):.2f}° / 镜头朝画面{'右' if ry >= 0 else '左'}转 {abs(ry):.2f}°")


def median_of(history, key):
    vals = [h[key] for h in history if key in h]
    return np.median(np.stack(vals), axis=0) if vals else None


def compose(live, ref, ref_edges, mode, est, aligned, rects=(), ref_label="dataset"):
    if mode == "edges":
        view = live.copy()
        view[ref_edges > 0] = (0, 255, 0)
        for x, y, w, h in rects:                                    # 框选的参照物区域
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 255), 1)
    elif mode == "blend":
        view = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
    elif mode == "diff":
        view = cv2.applyColorMap(cv2.absdiff(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY),
                                             cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)), cv2.COLORMAP_INFERNO)
    else:
        view = ref.copy()
    for u, v in est.get("inlier_pts", []):
        cv2.circle(view, (int(u), int(v)), 3, (255, 0, 255), 1)
    if aligned:                                                     # 对准了：实时画面底部一条醒目的绿条
        cv2.rectangle(view, (0, H - 44), (W, H), (40, 170, 40), -1)
        cv2.putText(view, "ALIGNED  < 1 px  - camera matches dataset", (12, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    dataset = ref.copy()                                            # 右边：数据集画面，对着它复位场景
    for panel, label in ((view, f"live + {mode}"), (dataset, ref_label)):
        cv2.rectangle(panel, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    view = np.hstack([view, dataset])
    bar = np.zeros((92, 2 * W, 3), np.uint8)
    colour = (60, 220, 60) if aligned else (0, 200, 255)
    lines = [f"[{mode}]  matches {est.get('matches', 0)}  2D inliers {est.get('inliers2d', 0)}  "
             f"3D inliers {est.get('inliers3d', 0)}" + ("   ALIGNED" if aligned else
             f"   target: shift<{OK_SHIFT_PX:g}px rot<{OK_ROT_DEG:g}deg scale<{OK_SCALE * 100:g}% "
             f"(>={OK_MIN_INLIERS} inliers, {OK_FRAMES} frames)")]
    if "shift" in est:
        lines.append(f"image -> ref: shift ({est['shift'][0]:+.1f}, {est['shift'][1]:+.1f}) px  "
                     f"rot {est['rot']:+.2f} deg  scale {est['scale']:.4f}")
    if "move_mm" in est:
        m, r = est["move_mm"], est["rot_deg"]
        lines.append(f"camera move x {m[0]:+.1f}  y {m[1]:+.1f}  z {m[2]:+.1f} mm   (reproj {est['reproj_px']:.2f} px)")
        lines.append(f"camera rot  x {r[0]:+.2f}  y {r[1]:+.2f}  z {r[2]:+.2f} deg   x=right y=down z=view")
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour if i == 0 else (255, 255, 255), 1)
    return np.vstack([view, bar])


def draw_rois(ref, rects, help_text):
    view = ref.copy()
    for i, (x, y, w, h) in enumerate(rects):
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(view, str(i + 1), (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.rectangle(view, (0, H - 24), (W, H), (0, 0, 0), -1)
    cv2.putText(view, help_text, (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    return view


def select_rois(ref, path, rects=()):
    """框参照物区域。cv2.selectROIs 每块都要按空格/回车确认，没确认就按 Esc 的那块会丢（11-10 桌角就这么丢的），
    所以框完先出预览：回车存，a 接着加，u 撤销最后一块，r 全部重框。"""
    rects = list(rects)
    adding = not rects
    while True:
        if adding:
            print("  拖一个框 → 按空格或回车确认这一块（每块都要确认！）→ 接着拖下一块；全部框完按 Esc", flush=True)
            new = cv2.selectROIs("drag a box, SPACE/ENTER keeps it (every box!), ESC when done",
                                 draw_rois(ref, rects, "existing boxes in red"), showCrosshair=False)
            cv2.destroyAllWindows()
            rects += [tuple(int(v) for v in r) for r in new if r[2] > 0 and r[3] > 0]
            adding = False
        print(f"  现在 {len(rects)} 块：{rects}；预览窗口里 回车=保存  a=接着加  u=撤销最后一块  r=全部重框", flush=True)
        title = "check boxes: ENTER save | a add | u undo last | r redo all"
        cv2.imshow(title, draw_rois(ref, rects, "ENTER save | a add more | u undo last | r redo all"))
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key in (13, 10):
            break
        if key == ord("a"):
            adding = True
        elif key == ord("u") and rects:
            rects.pop()
        elif key == ord("r"):
            rects, adding = [], True
    path.write_text(json.dumps({"camera": CAM, "rects_xywh": rects}, indent=1), encoding="utf-8")
    print(f"  存了 {len(rects)} 块到 {path}")
    return rects


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="~/atom_episodes/wall_unlocked_v2", help="旧数据集根目录")
    ap.add_argument("--tasks", default="bolt",
                    help="右边显示哪个任务的数据集画面，默认 bolt；逗号分隔给多个时窗口里按 t 切")
    ap.add_argument("--ref-episodes", help="取第几条到第几条的第 0 帧做中位图，如 40-49；默认每个任务最后 10 条")
    ap.add_argument("--ref-image", help="直接给一张参照图（BGR 图片文件），给了就不读视频、不能切任务")
    ap.add_argument("--out", default="~/wall_exec_logs/camera_align", help="框选区域、快照、读数存这里")
    ap.add_argument("--select-roi", action="store_true", help="重新框没动过的区域（覆盖存下的框，下次自动用）")
    ap.add_argument("--add-roi", action="store_true", help="在存下的框上接着加（比如补框桌角、桌缝）")
    ap.add_argument("--full-image", action="store_true", help="不用存下的框，整张图都拿来匹配")
    ap.add_argument("--overlay", choices=OVERLAYS, default="blend",
                    help="一开始的叠加方式，默认 blend（半透明：调到夹爪没有重影）；窗口里按 o 切换")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if args.ref_image:
        img = cv2.imread(str(Path(args.ref_image).expanduser()))
        if img is None:
            raise SystemExit(f"✗ 读不了 {args.ref_image}")
        refs = [(img, Path(args.ref_image).name, "", 1)]
    else:
        refs = [task_reference(args.data, t.strip(), args.ref_episodes) for t in args.tasks.split(",") if t.strip()]
    for img, task, span, n in refs:
        print(f"  数据集画面：{task} episode {span} 第 0 帧中位图（{n} 张）")
    roi_path = out / f"roi_{CAM}.json"
    saved = ([tuple(r) for r in json.loads(roi_path.read_text(encoding="utf-8"))["rects_xywh"]]
             if roi_path.exists() else [])
    if args.select_roi or args.add_roi:
        rects = select_rois(refs[0][0], roi_path, saved if args.add_roi else [])
    elif not args.full_image:
        rects = saved
    else:
        rects = []
    print(f"  匹配区域：{'整张图' if not rects else f'{len(rects)} 块框选（{roi_path}）'}")
    mask = roi_mask(rects)

    cams, K, m_per_unit = open_camera()
    print(f"  {CAM} 内参 fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}；深度对齐彩色")
    state = {}

    def activate(i):
        """换参照（数据集画面）：重算参照特征和轮廓，读数历史清掉。"""
        ref, task, span, n = refs[i]
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
        if mask is not None:
            edges &= mask                                           # 只画框选区域里的参照轮廓，别被杂物线条淹掉
        state.update(i=i, ref=ref, edges=edges, est=Estimator(ref, mask, K, m_per_unit),
                     label=f"dataset: {task} ep {span} frame 0 (median of {n})", desc=f"{task} episode {span}")
        cv2.imwrite(str(out / f"reference_{CAM}_{task}.png"), ref)

    def save_snapshot(live, shown, aligned, tag=""):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"live_{CAM}_{stamp}{tag}"
        cv2.imwrite(str(out / f"{name}.png"), live)
        record = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in shown.items() if k != "inlier_pts"}
        record.update(time=stamp, aligned=aligned, reference=state["desc"], rois=rects, K=K.tolist(),
                      criteria=dict(shift_px=OK_SHIFT_PX, rot_deg=OK_ROT_DEG, scale=OK_SCALE,
                                    min_inliers=OK_MIN_INLIERS, frames=OK_FRAMES))
        (out / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return name

    activate(0)
    print(f"  目标：画面中心偏移 < {OK_SHIFT_PX:g} px、转角 < {OK_ROT_DEG:g}°、缩放误差 < {OK_SCALE * 100:g}%，"
          f"内点 ≥ {OK_MIN_INLIERS}，连续 {OK_FRAMES} 帧。\n"
          "  左边默认半透明（实时 + 数据集各一半）：调到回零的夹爪完全没有重影。"
          "对准后窗口底部出现绿色 ALIGNED，终端提示并自动存快照。", flush=True)
    history, streak, mode, last_print = deque(maxlen=5), 0, OVERLAYS.index(args.overlay), 0.0
    was_aligned = False
    win = "cam_high align (t: dataset task, o: overlay, s: snapshot, q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 2 * W, H + 92)
    try:
        while True:
            frame = cams.peek().get(CAM)
            depth = cams.latest_depth(CAM)
            if frame is None or depth is None:
                time.sleep(0.05)
                continue
            live = cv2.cvtColor(np.ascontiguousarray(frame[..., :3]), cv2.COLOR_RGB2BGR)   # 驱动给 RGB
            raw = state["est"](live, depth)
            if "shift" in raw:
                history.append(raw)
            shown = dict(raw)
            for key in ("shift", "rot", "scale", "move_mm", "rot_deg", "reproj_px"):
                value = median_of(history, key)
                if value is not None:
                    shown[key] = value
            ok = ("shift" in raw and raw.get("inliers2d", 0) >= OK_MIN_INLIERS and np.hypot(*raw["shift"]) < OK_SHIFT_PX
                  and abs(raw["rot"]) < OK_ROT_DEG and abs(raw["scale"] - 1) < OK_SCALE)
            streak = streak + 1 if ok else 0
            aligned = streak >= OK_FRAMES
            if aligned and not was_aligned:                         # 刚对准：响一声、提示、自动存快照
                name = save_snapshot(live, shown, True, "_aligned")
                print(f"\a\n  ✓✓ 相机已对准：中心偏移 {np.hypot(*raw['shift']):.2f} px、转 {raw['rot']:+.3f}°、"
                      f"缩放 {raw['scale']:.4f}（{raw['inliers2d']} 个内点，连续 {OK_FRAMES} 帧）。"
                      f"已存快照 {name}.png / .json —— 可以按 q 退出，固定好相机支架\n", flush=True)
            elif was_aligned and not aligned:
                print("  ✗ 又偏出 1 px 了（支架松了？碰到了？）", flush=True)
            was_aligned = aligned
            cv2.imshow(win, compose(live, state["ref"], state["edges"], OVERLAYS[mode], shown, aligned, rects,
                                    state["label"]))
            now = time.monotonic()
            if now - last_print > 1.0 and "shift" in shown:
                last_print = now
                text = (f"  画面要挪 ({shown['shift'][0]:+5.1f}, {shown['shift'][1]:+5.1f}) px  转 {shown['rot']:+5.2f}°  "
                        f"缩放 {shown['scale']:.4f}")
                if "move_mm" in shown:
                    text += " | 相机 " + directions(shown["move_mm"], shown["rot_deg"])
                print(text + ("  ✓ 对准" if aligned else ""), flush=True)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                mode = (mode + 1) % len(OVERLAYS)
            if key == ord("t") and len(refs) > 1:
                activate((state["i"] + 1) % len(refs))
                history.clear()
                streak = 0
                print(f"  换数据集画面：{state['desc']}", flush=True)
            if key == ord("s"):
                print(f"  存了快照 {save_snapshot(live, shown, aligned)}.png / .json", flush=True)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        cams.disconnect()

if __name__ == "__main__":
    main()

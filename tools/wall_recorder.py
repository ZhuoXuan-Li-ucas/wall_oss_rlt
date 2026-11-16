"""执行时录两路相机视频，格式同采集（每路一个 mp4、mp4v、30 fps、原分辨率，驱动给的 RGB 转 BGR 再写），
外加每帧的序号和拍摄时刻，和 ticks.npz 的 t 是同一个时钟（perf_counter），视频能逐帧对上控制日志。

独立线程，碰不到发臂指令的回路：
- 真相机：`cameras.frames_since(名字, 上一个序号)` 一帧不落地取。驱动缓存约 8 帧（30 fps 约 0.27 s），
  线程每 50 ms 取一次；序号断档就是丢帧，记进统计。它和 peek() 一样不推进 read() 的帧序号、不计 stale_reads。
- 没有 frames_since 的（模拟相机）：退回 peek()，按 fps 取，时刻取 perf_counter。
写坏了只停录像、打一行警告，执行照常。

产物（在记录目录下）：
  cam_high.mp4 / cam_front.mp4   俯视 / 腕部
  video_frames.npz               {cam}_t：每帧拍摄时刻；{cam}_seq：驱动帧序号
  stop() 的返回值                每路帧数、丢帧数、首末时刻、实际帧率（执行器写进 run.json 的 "video"）
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np


class Recorder:
    POLL_S = 0.05

    def __init__(self, cameras, out_dir, names, fps: float = 30.0, codec: str = "mp4v"):
        self.cameras, self.out_dir, self.names = cameras, Path(out_dir), tuple(names)
        self.fps, self.codec = float(fps), codec
        self._since = getattr(cameras, "frames_since", None)
        self.poll_s = self.POLL_S if self._since is not None else 1.0 / self.fps
        self.writers = {}
        self.stamps = {n: [] for n in self.names}
        self.seqs = {n: [] for n in self.names}
        self.last = {n: None for n in self.names}
        self.gaps = {n: 0 for n in self.names}
        self.error = None
        self.t_start = None
        self._halt = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="wall-recorder", daemon=True)

    def start(self):
        try:
            if self._since is not None:             # 开录之前缓存里的旧帧不要
                for n in self.names:
                    cands = self._since(n, None)
                    if cands:
                        self.last[n] = cands[-1][0]
            self.t_start = time.perf_counter()
            self.thread.start()
        except Exception as exc:  # noqa: BLE001  录不了就不录，执行照常
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"\n  ⚠ 录像没开起来：{self.error}（执行照常）", flush=True)
        return self

    def _grab(self):
        out = []
        if self._since is not None:
            for n in self.names:
                for seq, stamp, frame in self._since(n, self.last[n]):
                    if self.last[n] is not None and seq > self.last[n] + 1:
                        self.gaps[n] += seq - self.last[n] - 1
                    self.last[n] = seq
                    out.append((n, seq, stamp, frame))
        else:
            frames, now = self.cameras.peek(), time.perf_counter()
            for n in self.names:
                if n in frames:
                    self.last[n] = (self.last[n] or 0) + 1
                    out.append((n, self.last[n], now, frames[n]))
        return out

    def _write(self, name, seq, stamp, frame):
        import cv2
        img = np.ascontiguousarray(frame[..., :3])
        writer = self.writers.get(name)
        if writer is None:
            h, w = img.shape[:2]
            self.out_dir.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(self.out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*self.codec),
                                     self.fps, (w, h))
            if not writer.isOpened():
                raise IOError(f"{name}.mp4 开不了写（编码 {self.codec}）")
            self.writers[name] = writer
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))     # 驱动给 RGB，写视频要 BGR（同 collect）
        self.stamps[name].append(float(stamp))
        self.seqs[name].append(int(seq))

    def _loop(self):
        try:
            while not self._halt.is_set():
                started = time.perf_counter()
                for item in self._grab():
                    self._write(*item)
                self._halt.wait(max(0.0, self.poll_s - (time.perf_counter() - started)))
            for item in self._grab():              # 收尾：停之前已经到的帧
                self._write(*item)
        except Exception as exc:  # noqa: BLE001  录像坏了不影响执行
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"\n  ⚠ 录像停了：{self.error}（执行照常）", flush=True)

    def stop(self) -> dict:
        self._halt.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        arrays = {}
        for n in self.names:
            arrays[f"{n}_t"] = np.asarray(self.stamps[n], np.float64)
            arrays[f"{n}_seq"] = np.asarray(self.seqs[n], np.int64)
        try:
            for writer in self.writers.values():
                writer.release()
            self.out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self.out_dir / "video_frames.npz", **arrays)
        except Exception as exc:  # noqa: BLE001  收尾失败也要让执行器把 run.json 写完
            self.error = (self.error + "；" if self.error else "") + f"收尾 {type(exc).__name__}: {exc}"
            print(f"\n  ⚠ 录像收尾出错：{exc}", flush=True)
        stats ={"codec": self.codec, "fps_nominal": self.fps, "t_start": self.t_start, "error": self.error,
                 "source": "frames_since" if self._since is not None else "peek", "cams": {}}
        for n in self.names:
            t = arrays[f"{n}_t"]
            span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
            stats["cams"][n] = {"file": f"{n}.mp4" if n in self.writers else None, "frames": int(len(t)),
                                "dropped": int(self.gaps[n]),
                                "t_first": float(t[0]) if len(t) else None, "t_last": float(t[-1]) if len(t) else None,
                                "duration_s": round(span, 3),
                                "fps_actual": round((len(t) - 1) / span, 2) if span > 0 else None}
        return stats

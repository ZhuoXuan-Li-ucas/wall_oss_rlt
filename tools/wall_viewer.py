"""wall_execute.py 的实时画面窗口：左边俯视（cam_high），右边腕部（cam_front），底下一行状态。

**独立进程**，从共享内存读 wall_execute.py 写进来的最新画面，不碰硬件、不连推理服务。
照 DexGraspVLA- 的 deploy_viewer.py：窗口卡了、被关了都碰不到执行器发臂指令的回路
（AR5 0.25 s 收不到目标就锁停）。由 wall_execute.py 自动启动，不用单独跑；关掉窗口只是不看了。
"""
import signal
import sys
from multiprocessing import resource_tracker, shared_memory

import cv2
import numpy as np

H, W = 480, 640
# 共享内存布局，和 wall_execute.LiveView 一致
N_HDR = 9   # seq, alive, running, elapsed_s, chunk, step, latency_s, grip_pred, grip_cmd
HDR_BYTES = N_HDR * 8
IMG = H * W * 3
SIZE = HDR_BYTES + 2 * IMG


def views(buf):
    hdr = np.ndarray((N_HDR,), dtype=np.float64, buffer=buf, offset=0)
    top = np.ndarray((H, W, 3), dtype=np.uint8, buffer=buf, offset=HDR_BYTES)
    wrist = np.ndarray((H, W, 3), dtype=np.uint8, buffer=buf, offset=HDR_BYTES + IMG)
    return hdr, top, wrist


def compose(top, wrist, hdr):
    canvas = cv2.cvtColor(np.hstack([top, wrist]), cv2.COLOR_RGB2BGR)     # atom_robot 的相机给 RGB
    for x, name in ((0, "cam_high (top)"), (W, "cam_front (wrist)")):
        cv2.rectangle(canvas, (x, 0), (x + W, 22), (0, 0, 0), -1)
        cv2.putText(canvas, name, (x + 8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    running, elapsed, chunk, step, latency, gpred, gcmd = hdr[2:9]
    bar = np.zeros((34, canvas.shape[1], 3), np.uint8)
    cv2.circle(bar, (18, 17), 8, (60, 60, 255) if running else (180, 180, 180), -1)
    # 全是 ASCII：cv2.putText 没有中文字形
    if running:
        text = (f"RUNNING  {elapsed:5.1f} s   chunk {int(chunk)}  step {step:4.1f}   "
                f"latency {latency * 1000:4.0f} ms   gripper pred {gpred:.2f} -> {'OPEN' if gcmd >= 0.5 else 'CLOSE'}")
    else:
        text = "IDLE  (waiting for Enter, or stopped)"
    cv2.putText(bar, text, (36, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([canvas, bar])


def main(name):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C 交给执行器，它收工时会让这里退出
    shm = shared_memory.SharedMemory(name=name)
    # 只是借用：别让这个进程退出时把共享内存删掉（Python 3.11 的 resource_tracker 会这么干）
    resource_tracker.unregister(shm._name, "shared_memory")
    hdr, top, wrist = views(shm.buf)
    win = "WALL execute live (top | wrist)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 514)
    last_seq = -1.0
    try:
        while hdr[1] > 0:                                     # alive
            seq = hdr[0]
            if seq != last_seq and int(seq) % 2 == 0:           # 偶数 = 写完了
                frame = compose(top, wrist, hdr)
                if hdr[0] == seq:                               # 拷的过程中没被改写
                    last_seq = seq
                    cv2.imshow(win, frame)
            cv2.waitKey(30)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break                                           # 窗口被关了：不看了，执行照常
    finally:
        cv2.destroyAllWindows()
        del hdr, top, wrist                                     # 先放掉对共享内存的引用，close 才不报 BufferError
        shm.close()


if __name__ == "__main__":
    main(sys.argv[1])

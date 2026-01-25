#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import subprocess
import sys
from typing import List

FS_HZ = 700

def sec_to_window_samples(sec: float, fs_hz: int = FS_HZ) -> int:
    ws = int(round(sec * fs_hz))
    # train_baseline.py 要求 window_size 为偶数，保证 50% overlap
    if ws % 2 != 0:
        ws += 1
    return ws

def frange(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step 必须大于 0")
    vals = []
    x = start
    eps = 1e-9
    while x <= end + eps:
        vals.append(round(x, 10))
        x += step
    return vals

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--range", nargs=3, type=float, required=True,
                   metavar=("START_S", "END_S", "STEP_S"),
                   help="窗口秒数范围，例如 0.5 10 0.5")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--kfold", type=int, default=2, choices=[0, 2, 3])
    p.add_argument("--python", type=str, default="python3",
                   help="python 可执行文件，例如 python3")
    p.add_argument("--script", type=str, default="train_baseline.py",
                   help="训练脚本路径，默认同目录 train_baseline.py")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印命令不执行")
    args = p.parse_args()

    start_s, end_s, step_s = args.range
    secs = frange(start_s, end_s, step_s)

    print(f"[Runner] fs_hz={FS_HZ}  secs={secs[0]}..{secs[-1]} step={step_s}  count={len(secs)}")

    for sec in secs:
        ws = sec_to_window_samples(sec, FS_HZ)
        cmd = [
            args.python, args.script,
            "--data-root", args.data_root,
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--window-size", str(ws),
            "--kfold", str(args.kfold),
        ]
        print("\n[Runner] sec=", sec, " window_size=", ws)
        print("[Runner] cmd:", " ".join(cmd))

        if not args.dry_run:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"[Runner] 失败: sec={sec} window_size={ws} returncode={r.returncode}")
                sys.exit(r.returncode)

    print("\n[Runner] 全部完成")

if __name__ == "__main__":
    main()

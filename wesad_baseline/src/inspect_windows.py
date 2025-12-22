#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_windows.py

A 阶段验证脚本
目标：检查 WESAD 滑窗是否跨越不同标签，是否包含 ignore 标签 0 或 4
输出：
  1) subject_summary.csv  每个 subject 的总窗数 保留窗数 丢弃原因 分布等
  2) window_details.csv   可选 每个 window 的起止 index 多标签比例 丢弃原因等

用法示例
  python3 inspect_windows.py \
    --data-root /workspace/data/WESAD \
    --window-size 700 \
    --step-size 350 \
    --subjects 2,3,4,5,6,7,8,9,10,11,13,14,15,16,17 \
    --tau 1.0 \
    --out-dir /workspace/logs

说明
  1 WESAD 的 label 一般使用 0 1 2 3 4
    0 和 4 常用于 ignore 或非目标区间
    1 2 3 是三类状态
  2 tau 表示 window 内标签一致性的阈值
    tau=1.0 表示 window 内必须全一致
    tau=0.95 表示 majority label 占比至少 95 percent 才保留
"""

import argparse
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# 你可以按你项目的 label 定义改这里
DEFAULT_VALID_LABELS = [1, 2, 3]
DEFAULT_IGNORE_LABELS = [0, 4]


def parse_subjects(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def try_load_labels_from_file(path: str) -> Optional[np.ndarray]:
    """
    支持：
      npy: 直接 load
      npz: 自动找第一个 1D 数组
      csv txt: 读第一列或名为 label 的列
    返回 1D int array
    """
    p = str(path)
    ext = os.path.splitext(p)[1].lower()

    try:
        if ext == ".npy":
            arr = np.load(p, allow_pickle=False)
            arr = np.asarray(arr).squeeze()
            if arr.ndim != 1:
                return None
            return arr.astype(int)

        if ext == ".npz":
            z = np.load(p, allow_pickle=False)
            # 找到第一个一维数组
            for k in z.files:
                a = np.asarray(z[k]).squeeze()
                if a.ndim == 1:
                    return a.astype(int)
            return None

        if ext in [".csv", ".txt"]:
            # 轻量 csv 读取，优先找列名 label，否则取第一列可解析为 int 的
            with open(p, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if not rows:
                return None

            header = rows[0]
            data_rows = rows[1:] if any(not x.strip().isdigit() for x in header) else rows

            if data_rows is rows:
                # 无 header，当作单列 label
                vals = []
                for r in data_rows:
                    if not r:
                        continue
                    vals.append(int(float(r[0])))
                arr = np.asarray(vals, dtype=int)
                if arr.ndim != 1:
                    return None
                return arr

            # 有 header
            header_l = [h.strip().lower() for h in header]
            if "label" in header_l:
                idx = header_l.index("label")
            else:
                idx = 0

            vals = []
            for r in data_rows:
                if not r:
                    continue
                if idx >= len(r):
                    continue
                cell = r[idx].strip()
                if cell == "":
                    continue
                vals.append(int(float(cell)))
            arr = np.asarray(vals, dtype=int)
            if arr.ndim != 1:
                return None
            return arr

    except Exception:
        return None

    return None


def find_label_file_for_subject(data_root: str, sid: int) -> Optional[str]:
    """
    尝试自动定位某个 subject 的 label 文件
    策略：
      1 优先找包含 S{sid} 或 subject{sid} 的路径
      2 文件名包含 label 或 y 或 annot
      3 扩展名优先 npy npz csv txt
    """
    root = Path(data_root)
    if not root.exists():
        return None

    sid_tokens = [f"S{sid}", f"s{sid}", f"subject{sid}", f"subj{sid}", f"Subj{sid}"]

    cand = []
    for dirpath, _, filenames in os.walk(root):
        dp = str(dirpath)
        if not any(tok in dp for tok in sid_tokens):
            continue
        for fn in filenames:
            fn_l = fn.lower()
            if not (fn_l.endswith(".npy") or fn_l.endswith(".npz") or fn_l.endswith(".csv") or fn_l.endswith(".txt")):
                continue
            if ("label" in fn_l) or ("annot" in fn_l) or (fn_l.startswith("y")):
                cand.append(str(Path(dirpath) / fn))

    # 如果没找到明显 label 文件，再退一步：只要在 subject 目录下找任何可能的一维数组文件
    if not cand:
        for dirpath, _, filenames in os.walk(root):
            dp = str(dirpath)
            if not any(tok in dp for tok in sid_tokens):
                continue
            for fn in filenames:
                fn_l = fn.lower()
                if fn_l.endswith(".npy") or fn_l.endswith(".npz") or fn_l.endswith(".csv") or fn_l.endswith(".txt"):
                    cand.append(str(Path(dirpath) / fn))

    if not cand:
        return None

    # 简单打分排序
    def score(p: str) -> int:
        s = 0
        pl = os.path.basename(p).lower()
        if "label" in pl:
            s += 10
        if "annot" in pl:
            s += 8
        if pl.startswith("y"):
            s += 5
        if pl.endswith(".npy"):
            s += 4
        if pl.endswith(".npz"):
            s += 3
        if pl.endswith(".csv"):
            s += 2
        if pl.endswith(".txt"):
            s += 1
        return -s

    cand.sort(key=score)

    # 选出第一个能成功 load 为 1D 的
    for p in cand[:30]:
        arr = try_load_labels_from_file(p)
        if arr is not None and arr.ndim == 1 and len(arr) > 0:
            return p

    return None


def majority_vote(labels: np.ndarray) -> Tuple[int, float]:
    """
    返回 (major_label, ratio)
    """
    vals, counts = np.unique(labels, return_counts=True)
    idx = int(np.argmax(counts))
    major = int(vals[idx])
    ratio = float(counts[idx] / counts.sum())
    return major, ratio


def inspect_subject_windows(
    sid: int,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    valid_labels: List[int],
    ignore_labels: List[int],
    tau: float,
    emit_details: bool,
) -> Tuple[Dict, List[Dict]]:
    """
    输出：
      subject_summary: dict
      details: list[dict]
    """
    n = int(labels.shape[0])
    ws = int(window_size)
    ss = int(step_size)

    total = 0
    kept = 0

    drop_ignore = 0
    drop_inconsistent = 0
    keep_by_class = {c: 0 for c in valid_labels}

    details = []

    for start in range(0, n - ws + 1, ss):
        end = start + ws
        wlab = labels[start:end]
        total += 1

        uniq = set(int(x) for x in np.unique(wlab))

        reason = "keep"
        major_label = None
        major_ratio = None

        if any(x in ignore_labels for x in uniq):
            reason = "drop_contains_ignore_label"
            drop_ignore += 1
        else:
            # 只允许 valid_labels
            if not all(x in valid_labels for x in uniq):
                reason = "drop_contains_unknown_label"
                drop_inconsistent += 1
            else:
                if len(uniq) == 1:
                    major_label = int(next(iter(uniq)))
                    major_ratio = 1.0
                    kept += 1
                    keep_by_class[major_label] += 1
                else:
                    major_label, major_ratio = majority_vote(wlab)
                    if major_ratio >= tau:
                        reason = "keep_majority"
                        kept += 1
                        keep_by_class[major_label] += 1
                    else:
                        reason = "drop_label_inconsistent"
                        drop_inconsistent += 1

        if emit_details:
            details.append(
                {
                    "subject": f"S{sid}",
                    "start": start,
                    "end": end,
                    "window_size": ws,
                    "step_size": ss,
                    "unique_labels": "|".join(str(x) for x in sorted(list(uniq))),
                    "major_label": "" if major_label is None else int(major_label),
                    "major_ratio": "" if major_ratio is None else float(major_ratio),
                    "reason": reason,
                }
            )

    summary = {
        "subject": f"S{sid}",
        "label_len": n,
        "window_size": ws,
        "step_size": ss,
        "tau": float(tau),
        "total_windows": total,
        "kept_windows": kept,
        "kept_ratio": 0.0 if total == 0 else kept / total,
        "dropped_ignore": drop_ignore,
        "dropped_inconsistent": drop_inconsistent,
    }
    for c in valid_labels:
        summary[f"kept_class_{c}"] = int(keep_by_class[c])

    return summary, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="WESAD 数据根目录")
    ap.add_argument("--subjects", default="", help="subject 列表，例如 2,3,4")
    ap.add_argument("--window-size", type=int, required=True)
    ap.add_argument("--step-size", type=int, required=True)
    ap.add_argument("--tau", type=float, default=1.0, help="一致性阈值，建议 1.0 或 0.95")
    ap.add_argument("--valid-labels", default="1,2,3")
    ap.add_argument("--ignore-labels", default="0,4")
    ap.add_argument("--out-dir", default="./logs", help="输出目录")
    ap.add_argument("--emit-details", action="store_true", help="是否输出 window 级明细 csv")
    args = ap.parse_args()

    data_root = args.data_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects(args.subjects)
    if not subjects:
        raise RuntimeError("你必须通过 --subjects 指定 subject 列表，例如 2,3,4")

    valid_labels = parse_subjects(args.valid_labels)
    ignore_labels = parse_subjects(args.ignore_labels)
    tau = float(args.tau)

    all_summaries = []
    all_details = []

    print("[inspect_windows] data_root:", data_root)
    print("[inspect_windows] subjects :", subjects)
    print("[inspect_windows] window_size, step_size:", args.window_size, args.step_size)
    print("[inspect_windows] valid_labels:", valid_labels, "ignore_labels:", ignore_labels, "tau:", tau)
    print("")

    for sid in subjects:
        label_path = find_label_file_for_subject(data_root, sid)
        if label_path is None:
            print(f"[WARN] S{sid}: 没找到 label 文件，请检查 data_root 目录结构")
            continue

        labels = try_load_labels_from_file(label_path)
        if labels is None:
            print(f"[WARN] S{sid}: 找到了 label 文件但读取失败: {label_path}")
            continue

        summary, details = inspect_subject_windows(
            sid=sid,
            labels=labels,
            window_size=args.window_size,
            step_size=args.step_size,
            valid_labels=valid_labels,
            ignore_labels=ignore_labels,
            tau=tau,
            emit_details=bool(args.emit_details),
        )

        summary["label_file"] = label_path
        all_summaries.append(summary)
        all_details.extend(details)

        print(
            f"[OK] S{sid} label={len(labels)}  total_windows={summary['total_windows']}  kept={summary['kept_windows']}  kept_ratio={summary['kept_ratio']:.3f}  drop_ignore={summary['dropped_ignore']}  drop_inconsistent={summary['dropped_inconsistent']}"
        )

    # 写 summary
    if all_summaries:
        summary_path = out_dir / "subject_summary.csv"
        keys = list(all_summaries[0].keys())
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_summaries:
                w.writerow(r)
        print("\n[Saved] subject_summary:", summary_path)

    # 写 details
    if args.emit_details and all_details:
        details_path = out_dir / "window_details.csv"
        keys = list(all_details[0].keys())
        with open(details_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_details:
                w.writerow(r)
        print("[Saved] window_details:", details_path)

    print("\n[Done] inspect_windows finished")


if __name__ == "__main__":
    main()

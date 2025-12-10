#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_labels.py (FINAL CSV-ONLY VERSION)

输出:
  /workspace/logs/wesad_full_label_summary.csv

CSV 内容包含:
  subject
  total_windows
  count_cls0 count_cls1 count_cls2
  ratio_cls0 ratio_cls1 ratio_cls2
  unique_labels 
  raw_counts     
"""

import os
import argparse
import numpy as np
from pathlib import Path
import csv

from data.wesad_dataset import WESADDataset


def get_subject_ids(data_root):
    sids = []
    for name in os.listdir(data_root):
        if name.startswith("S"):
            try:
                sid = int(name[1:])
                sids.append(sid)
            except:
                pass
    return sorted(sids)


def inspect_subject(data_root, sid, window_size, step_size, num_classes=3):
    dataset = WESADDataset(
        root=data_root,
        subject_ids=[sid],
        window_size=window_size,
        step_size=step_size,
        num_classes=num_classes,
        normalize=True,
    )

    n_samples = len(dataset)
    if n_samples == 0:
        return {
            "subject": f"S{sid}",
            "total_windows": 0,
            "unique_labels": "[]",
            "raw_counts": "[]",
            "count_cls0": 0,
            "count_cls1": 0,
            "count_cls2": 0,
            "ratio_cls0": 0.0,
            "ratio_cls1": 0.0,
            "ratio_cls2": 0.0,
        }

    labels = []
    for i in range(n_samples):
        _, y = dataset[i]
        labels.append(int(y))
    labels = np.array(labels)

    unique, counts = np.unique(labels, return_counts=True)

    # 填满 0/1/2
    full_counts = np.zeros(num_classes, dtype=int)
    for u, c in zip(unique, counts):
        if 0 <= u < num_classes:
            full_counts[u] = c

    ratios = (
        full_counts.astype(float) / n_samples
        if n_samples > 0
        else np.zeros(num_classes)
    )

    return {
        "subject": f"S{sid}",
        "total_windows": int(n_samples),
        "unique_labels": str(list(unique)),
        "raw_counts": str(list(counts)),
        "count_cls0": int(full_counts[0]),
        "count_cls1": int(full_counts[1]),
        "count_cls2": int(full_counts[2]),
        "ratio_cls0": float(ratios[0]),
        "ratio_cls1": float(ratios[1]),
        "ratio_cls2": float(ratios[2]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="/workspace/data/WESAD",
    )
    parser.add_argument("--window-size", type=int, default=700)
    parser.add_argument("--step-size", type=int, default=350)
    parser.add_argument("--num-classes", type=int, default=3)

    args = parser.parse_args()

    data_root = args.data_root
    sids = get_subject_ids(data_root)

    # 收集所有 subject 结果
    rows = []

    global_total = 0
    global_counts = np.zeros(args.num_classes, dtype=int)

    for sid in sids:
        info = inspect_subject(
            data_root,
            sid,
            args.window_size,
            args.step_size,
            args.num_classes,
        )
        rows.append(info)

        global_total += info["total_windows"]
        global_counts[0] += info["count_cls0"]
        global_counts[1] += info["count_cls1"]
        global_counts[2] += info["count_cls2"]

    # 全局比例
    if global_total > 0:
        global_ratios = global_counts.astype(float) / global_total
    else:
        global_ratios = np.zeros(args.num_classes)

    # 加 ALL 行
    rows.append(
        {
            "subject": "ALL",
            "total_windows": int(global_total),
            "unique_labels": "N/A",
            "raw_counts": "N/A",
            "count_cls0": int(global_counts[0]),
            "count_cls1": int(global_counts[1]),
            "count_cls2": int(global_counts[2]),
            "ratio_cls0": float(global_ratios[0]),
            "ratio_cls1": float(global_ratios[1]),
            "ratio_cls2": float(global_ratios[2]),
        }
    )

    # 写 CSV
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    csv_path = log_dir / "wesad_full_label_summary.csv"

    fieldnames = rows[0].keys()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("")
    print("CSV summary saved to:")
    print("  ", csv_path)


if __name__ == "__main__":
    main()
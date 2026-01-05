#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

from data.wesad_dataset import WESADDataset
from data.stream_builder import build_subject_blocks  


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def summarize_labels(dataset: WESADDataset, indices: List[int], num_classes: int) -> Dict:
    cnt = [0] * num_classes
    for i in indices:
        y = int(dataset.get_label(i))
        if 0 <= y < num_classes:
            cnt[y] += 1
    n = sum(cnt)
    ratios = [safe_div(c, n) for c in cnt]
    return {"n": n, "counts": cnt, "ratios": ratios}


def is_strictly_increasing(xs: List[int]) -> bool:
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            return False
    return True


def contiguous_runs(indices: List[int]) -> int:
    """
    统计 index 序列中连续段的数量
    段越多，说明越碎，越不符合 streaming 的直觉
    """
    if not indices:
        return 0
    xs = sorted(indices)
    runs = 1
    for i in range(1, len(xs)):
        if xs[i] != xs[i - 1] + 1:
            runs += 1
    return runs


def build_blocks_from_indices(
    indices: List[int],
    block_size: int,
) -> List[Tuple[int, int]]:
    """
    根据 index 推回 block 边界，仅用于验证
    返回 block 的 [start, end) 区间
    """
    if not indices:
        return []
    xs = sorted(indices)
    blocks = []
    cur_start = xs[0]
    cur_end = xs[0] + 1
    for k in range(1, len(xs)):
        if xs[k] == cur_end:
            cur_end += 1
        else:
            blocks.append((cur_start, cur_end))
            cur_start = xs[k]
            cur_end = xs[k] + 1
    blocks.append((cur_start, cur_end))
    return blocks


def block_purity_stats(
    dataset: WESADDataset,
    start: int,
    end: int,
    num_classes: int,
) -> Dict:
    cnt = [0] * num_classes
    for i in range(start, end):
        y = int(dataset.get_label(i))
        if 0 <= y < num_classes:
            cnt[y] += 1
    total = sum(cnt)
    major = int(max(range(num_classes), key=lambda c: cnt[c])) if total > 0 else -1
    purity = safe_div(max(cnt), total) if total > 0 else 0.0
    return {
        "start": start,
        "end": end,
        "len": end - start,
        "counts": cnt,
        "major": major,
        "purity": purity,
    }


def verify_subject_A(
    data_root: str,
    subject_id: int,
    window_size: int,
    step_size: int,
    num_classes: int = 3,
    normalize: bool = True,
    block_size: int = 50,
    split_ratio: Tuple[float, float, float] = (0.2, 0.4, 0.4),
    purity_warn_th: float = 0.8,
) -> Dict:
    dataset, idx_pre, idx_adapt, idx_eval = build_subject_blocks(
        root=data_root,
        subject_id=subject_id,
        window_size=window_size,
        step_size=step_size,
        num_classes=num_classes,
        normalize=normalize,
        block_size=block_size,
        split_ratio=split_ratio,
    )

    n = len(dataset)

    set_pre = set(idx_pre)
    set_adapt = set(idx_adapt)
    set_eval = set(idx_eval)

    overlap_pre_adapt = len(set_pre & set_adapt)
    overlap_pre_eval = len(set_pre & set_eval)
    overlap_adapt_eval = len(set_adapt & set_eval)

    union_all = set_pre | set_adapt | set_eval
    covered = len(union_all)
    missing = n - covered

    stats_pre = summarize_labels(dataset, list(set_pre), num_classes)
    stats_adapt = summarize_labels(dataset, list(set_adapt), num_classes)
    stats_eval = summarize_labels(dataset, list(set_eval), num_classes)

    # 纯度检查
    # 注意：build_subject_blocks 是按固定 block_size 切 block，再按 major_label 分配
    # 这里重新按原始 block 划分去计算纯度，以验证 major_label 的合理性
    num_blocks = (n + block_size - 1) // block_size
    purities = []
    low_purity_blocks = 0
    for b in range(num_blocks):
        s = b * block_size
        e = min((b + 1) * block_size, n)
        ps = block_purity_stats(dataset, s, e, num_classes)
        purities.append(ps["purity"])
        if ps["purity"] < purity_warn_th:
            low_purity_blocks += 1

    mean_purity = float(sum(purities) / len(purities)) if purities else 0.0
    min_purity = float(min(purities)) if purities else 0.0

    # streaming 结构感检查
    runs_pre = contiguous_runs(list(set_pre))
    runs_adapt = contiguous_runs(list(set_adapt))
    runs_eval = contiguous_runs(list(set_eval))

    inc_pre = is_strictly_increasing(sorted(set_pre))
    inc_adapt = is_strictly_increasing(sorted(set_adapt))
    inc_eval = is_strictly_increasing(sorted(set_eval))

    return {
        "subject": f"S{subject_id}",
        "n_total": n,
        "n_pre": stats_pre["n"],
        "n_adapt": stats_adapt["n"],
        "n_eval": stats_eval["n"],
        "missing": missing,
        "overlap_pre_adapt": overlap_pre_adapt,
        "overlap_pre_eval": overlap_pre_eval,
        "overlap_adapt_eval": overlap_adapt_eval,
        "pre_counts": stats_pre["counts"],
        "adapt_counts": stats_adapt["counts"],
        "eval_counts": stats_eval["counts"],
        "pre_ratios": stats_pre["ratios"],
        "adapt_ratios": stats_adapt["ratios"],
        "eval_ratios": stats_eval["ratios"],
        "block_size": block_size,
        "num_blocks": num_blocks,
        "mean_block_purity": mean_purity,
        "min_block_purity": min_purity,
        "low_purity_blocks": low_purity_blocks,
        "purity_warn_th": purity_warn_th,
        "runs_pre": runs_pre,
        "runs_adapt": runs_adapt,
        "runs_eval": runs_eval,
        "strict_increasing_pre": inc_pre,
        "strict_increasing_adapt": inc_adapt,
        "strict_increasing_eval": inc_eval,
        "window_size": window_size,
        "step_size": step_size,
        "split_ratio": split_ratio,
    }


def print_subject_report(r: Dict):
    print("")
    print("================================================")
    print(f"[Verify A] {r['subject']}  total={r['n_total']}")
    print("------------------------------------------------")
    print(f"Split sizes: pre={r['n_pre']}  adapt={r['n_adapt']}  eval={r['n_eval']}  missing={r['missing']}")
    print(f"Overlaps  : pre∩adapt={r['overlap_pre_adapt']}  pre∩eval={r['overlap_pre_eval']}  adapt∩eval={r['overlap_adapt_eval']}")

    print("")
    print("Class counts [c0,c1,c2]")
    print(f"  pre   : {r['pre_counts']}   ratios={ [round(x,4) for x in r['pre_ratios']] }")
    print(f"  adapt : {r['adapt_counts']} ratios={ [round(x,4) for x in r['adapt_ratios']] }")
    print(f"  eval  : {r['eval_counts']}  ratios={ [round(x,4) for x in r['eval_ratios']] }")

    print("")
    print("Block purity check")
    print(f"  block_size        : {r['block_size']}")
    print(f"  num_blocks        : {r['num_blocks']}")
    print(f"  mean block purity : {r['mean_block_purity']:.4f}")
    print(f"  min  block purity : {r['min_block_purity']:.4f}")
    print(f"  low purity blocks : {r['low_purity_blocks']}  (purity<th={r['purity_warn_th']})")

    print("")
    print("Streaming structure check")
    print(f"  contiguous runs pre   : {r['runs_pre']}")
    print(f"  contiguous runs adapt : {r['runs_adapt']}")
    print(f"  contiguous runs eval  : {r['runs_eval']}")
    print(f"  increasing pre        : {r['strict_increasing_pre']}")
    print(f"  increasing adapt      : {r['strict_increasing_adapt']}")
    print(f"  increasing eval       : {r['strict_increasing_eval']}")
    print("================================================")


def save_csv(reports: List[Dict], out_csv: str):
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)

    # 展平字段，counts 和 ratios 拉平为 3 列
    rows = []
    for r in reports:
        row = {
            "subject": r["subject"],
            "n_total": r["n_total"],
            "n_pre": r["n_pre"],
            "n_adapt": r["n_adapt"],
            "n_eval": r["n_eval"],
            "missing": r["missing"],
            "overlap_pre_adapt": r["overlap_pre_adapt"],
            "overlap_pre_eval": r["overlap_pre_eval"],
            "overlap_adapt_eval": r["overlap_adapt_eval"],
            "pre_c0": r["pre_counts"][0], "pre_c1": r["pre_counts"][1], "pre_c2": r["pre_counts"][2],
            "adapt_c0": r["adapt_counts"][0], "adapt_c1": r["adapt_counts"][1], "adapt_c2": r["adapt_counts"][2],
            "eval_c0": r["eval_counts"][0], "eval_c1": r["eval_counts"][1], "eval_c2": r["eval_counts"][2],
            "pre_r0": r["pre_ratios"][0], "pre_r1": r["pre_ratios"][1], "pre_r2": r["pre_ratios"][2],
            "adapt_r0": r["adapt_ratios"][0], "adapt_r1": r["adapt_ratios"][1], "adapt_r2": r["adapt_ratios"][2],
            "eval_r0": r["eval_ratios"][0], "eval_r1": r["eval_ratios"][1], "eval_r2": r["eval_ratios"][2],
            "block_size": r["block_size"],
            "num_blocks": r["num_blocks"],
            "mean_block_purity": r["mean_block_purity"],
            "min_block_purity": r["min_block_purity"],
            "low_purity_blocks": r["low_purity_blocks"],
            "runs_pre": r["runs_pre"],
            "runs_adapt": r["runs_adapt"],
            "runs_eval": r["runs_eval"],
            "inc_pre": r["strict_increasing_pre"],
            "inc_adapt": r["strict_increasing_adapt"],
            "inc_eval": r["strict_increasing_eval"],
            "window_size": r["window_size"],
            "step_size": r["step_size"],
        }
        rows.append(row)

    cols = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print("")
    print(f"[OK] CSV saved to: {out_csv}")


def main():
    data_root = os.environ.get("WESAD_ROOT", "/workspace/data/WESAD")

    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    window_size = 700
    step_size = 350
    block_size = 50
    split_ratio = (0.2, 0.4, 0.4)

    reports = []
    for sid in subjects:
        r = verify_subject_A(
            data_root=data_root,
            subject_id=sid,
            window_size=window_size,
            step_size=step_size,
            num_classes=3,
            normalize=True,
            block_size=block_size,
            split_ratio=split_ratio,
            purity_warn_th=0.8,
        )
        print_subject_report(r)
        reports.append(r)

        # 关键硬性条件，任何一个触发都说明 A 有 bug 或不合理
        if r["overlap_pre_adapt"] or r["overlap_pre_eval"] or r["overlap_adapt_eval"]:
            print(f"[FATAL] {r['subject']} split overlap detected. A is invalid.")
            break

    out_csv = "/workspace/logs/verify_A_blocksplit_summary.csv"
    save_csv(reports, out_csv)


if __name__ == "__main__":
    main()

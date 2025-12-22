#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.cnn_baseline import CNNBaseline
from data.wesad_dataset import WESADDataset
from t3a_wrapper_v2 import T3AWrapper


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1).clamp_min(1e-12)
    return -(p * torch.log(p)).sum(dim=1)


@torch.no_grad()
def run_verify(
    data_root: str,
    ckpt_path: str,
    subject_id: int,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    M: int,
    ent_th: float,
    warmup: int,
    out_json: str,
):
    state = torch.load(ckpt_path, map_location="cpu")
    config = state.get("config", {})
    window_size = int(config.get("window_size", 700))
    step_size = int(config.get("step_size", 350))

    model = CNNBaseline(in_channels=8, num_classes=3).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    ds = WESADDataset(
        root=data_root,
        subject_ids=[subject_id],
        window_size=window_size,
        step_size=step_size,
        num_classes=3,
        normalize=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # t3a = T3AWrapper(
    #     head_linear=model.head,
    #     num_classes=3,
    #     M=M,
    #     ent_threshold=ent_th,
    #     warmup_steps=warmup,
    #     device=device,
    # ).to(device)
    t3a = T3AWrapper(
        head_linear=model.head,
        num_classes=3,
        M=t3a_params["M"],
        warmup_steps=t3a_params.get("warmup_steps", 0),
        device=device,
        ent_quantile=t3a_params.get("ent_quantile", 0.2),
        ent_threshold=t3a_params.get("ent_threshold", None),
        tau=t3a_params.get("tau", 1.0),
        use_head_bias=t3a_params.get("use_head_bias", True),
        keep_weight_anchor=t3a_params.get("keep_weight_anchor", True),
    ).to(device)
 

    rows = []
    changed_count = 0
    total_seen = 0

    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        z = model.forward_features(x)

        logits_base = model.head(z)
        pred_base = torch.argmax(logits_base, dim=1)

        ent_base = entropy_from_logits(logits_base)

        logits_t3a = t3a(z, update=True)
        pred_t3a = torch.argmax(logits_t3a, dim=1)

        diff = (logits_t3a - logits_base).abs().mean().item()
        delta_pred = (pred_t3a != pred_base).float().mean().item()

        base_acc = (pred_base == y).float().mean().item()
        t3a_acc = (pred_t3a == y).float().mean().item()

        sizes = t3a.support_sizes()

        rows.append(
            {
                "batch": int(b),
                "samples": int(y.numel()),
                "base_acc": float(base_acc),
                "t3a_acc": float(t3a_acc),
                "mean_abs_logit_diff": float(diff),
                "pred_changed_ratio": float(delta_pred),
                "entropy_mean": float(ent_base.mean().item()),
                "entropy_p50": float(ent_base.median().item()),
                "entropy_p10": float(torch.quantile(ent_base, 0.10).item()),
                "entropy_p90": float(torch.quantile(ent_base, 0.90).item()),
                "support_sizes": [int(s) for s in sizes],
            }
        )

        changed_count += int((pred_t3a != pred_base).sum().item())
        total_seen += int(y.numel())

        if b % 10 == 0:
            print(
                "batch",
                b,
                "base_acc",
                round(base_acc, 4),
                "t3a_acc",
                round(t3a_acc, 4),
                "diff",
                round(diff, 6),
                "changed",
                round(delta_pred, 4),
                "sizes",
                sizes,
                flush=True,
            )

    summary = {
        "ckpt_path": ckpt_path,
        "data_root": data_root,
        "subject_id": int(subject_id),
        "device": str(device),
        "batch_size": int(batch_size),
        "max_batches": int(max_batches),
        "window_size": int(window_size),
        "step_size": int(step_size),
        "t3a_params": {
            "M": int(M),
            "ent_threshold": float(ent_th),
            "warmup_steps": int(warmup),
        },
        "total_samples_seen": int(total_seen),
        "overall_pred_changed_ratio": float(changed_count / max(1, total_seen)),
        "first_batch": rows[0] if rows else None,
        "last_batch": rows[-1] if rows else None,
        "per_batch": rows,
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("")
    print("saved json to", out_json, flush=True)

    verdict = []
    if not rows:
        verdict.append("没有跑到任何 batch")
    else:
        first_sizes = rows[0]["support_sizes"]
        last_sizes = rows[-1]["support_sizes"]
        if last_sizes == first_sizes:
            verdict.append("support_sizes 没有变化 说明没有更新或被完全过滤")
        if rows[-1]["mean_abs_logit_diff"] < 1e-6:
            verdict.append("logits 几乎没变化 说明 wrapper 没生效")
        if summary["overall_pred_changed_ratio"] == 0.0:
            verdict.append("预测从不改变 说明 wrapper 没改变决策或 support 没增长")
        if not verdict:
            verdict.append("wrapper 有工作迹象 support 在增长 且 logits 与预测发生变化")

    print("verdict")
    for v in verdict:
        print(v, flush=True)


def main():
    p = argparse.ArgumentParser("Verify T3AWrapper without modifying any existing files")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--M", type=int, default=30)
    p.add_argument("--ent-th", type=float, default=0.6)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--out-json", type=str, default="logs/t3a_verify.json")
    args = p.parse_args()

    device = torch.device("cuda:0") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    run_verify(
        data_root=args.data_root,
        ckpt_path=args.ckpt,
        subject_id=args.subject,
        device=device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        M=args.M,
        ent_th=args.ent_th,
        warmup=args.warmup,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()

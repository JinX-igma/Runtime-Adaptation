#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_online_adapt_grid.py

独立验证脚本：不改原代码，只调用现有 dataset + split builder，
对同一 subject 的 streaming 数据执行：

1) frozen (不更新)
2) lr0 (走更新路径但 lr=0，验证不会“偷偷改参数”)
3) online head-only (伪标签 + 置信门控 + 小buffer)

并支持参数 grid，输出完整 csv 日志与汇总表。

依赖：torch, numpy
你需要确保能 import：
- models.cnn_baseline.CNNBaseline
- data.wesad_dataset.WESADDataset
- 你的 split builder：build_subject_blocks
"""

import os
import time
import json
import math
import argparse
import platform
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


from models.cnn_baseline import CNNBaseline
from data.wesad_dataset import WESADDataset

from data.stream_builder import build_subject_blocks  

NUM_CLASSES = 3


# =========================
# Utils
# =========================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_rel_log_dir(default_name="verifyA_grid") -> str:
    """
    默认把日志放在脚本同级的 ../logs/<default_name> 里
    这样 docker 外挂载 /workspace/logs 时也更容易看到
    """
    base = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(base, "..", "logs", default_name))
    ensure_dir(cand)
    return cand


def load_checkpoint_any(ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    兼容不同保存格式：
    - 纯 state_dict
    - {"state_dict": ...}
    - {"model_state": ...}
    """
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state", "model"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    if isinstance(obj, dict):
        # 也可能直接就是 state_dict
        # 简单判定：key 里是否含 conv/bn/fc
        keys = list(obj.keys())
        if keys and any(("conv" in kk or "bn" in kk or "fc" in kk) for kk in keys):
            return obj
    raise RuntimeError(f"无法识别 checkpoint 格式: {ckpt_path}")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_update(conf: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < conf.shape[0] and 0 <= p < conf.shape[1]:
            conf[t, p] += 1


def per_class_recall(conf: np.ndarray) -> List[float]:
    rec = []
    for c in range(conf.shape[0]):
        denom = conf[c, :].sum()
        rec.append(float(conf[c, c] / denom) if denom > 0 else 0.0)
    return rec


@torch.no_grad()
def eval_subset(
    model: nn.Module,
    dataset: WESADDataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[float, float, int, np.ndarray]:
    """
    返回：loss, acc, n_samples, conf_mat
    """
    if len(indices) == 0:
        return 0.0, 0.0, 0, np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum").item()
        pred = torch.argmax(logits, dim=1)

        total_loss += loss
        total_correct += int((pred == yb).sum().item())
        total_n += int(yb.numel())

        confusion_update(conf, yb.detach().cpu().numpy(), pred.detach().cpu().numpy())

    mean_loss = float(total_loss / max(total_n, 1))
    acc = float(total_correct / max(total_n, 1))
    return mean_loss, acc, total_n, conf


# =========================
# Online Head-only Learner
# =========================

@dataclass
class OnlineCfg:
    mode: str  # "frozen" | "lr0" | "online_head"
    lr: float
    conf_th: float
    buffer_max: int
    update_every: int
    batch_size: int
    seed: int
    use_teacher: bool
    ema: float


class TinyBuffer:
    def __init__(self, max_n: int):
        self.max_n = int(max_n)
        self.x = []
        self.y = []

    def __len__(self):
        return len(self.x)

    def add(self, xb: torch.Tensor, yb: torch.Tensor):
        # 存 CPU，节省显存
        xb_cpu = xb.detach().cpu()
        yb_cpu = yb.detach().cpu()
        for i in range(xb_cpu.shape[0]):
            self.x.append(xb_cpu[i])
            self.y.append(yb_cpu[i])
        # 截断
        if len(self.x) > self.max_n:
            extra = len(self.x) - self.max_n
            self.x = self.x[extra:]
            self.y = self.y[extra:]

    def sample_all(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.x) == 0:
            return None, None
        xb = torch.stack(self.x, dim=0).to(device)
        yb = torch.stack(self.y, dim=0).to(device)
        return xb, yb


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_head_only(model: nn.Module) -> List[nn.Parameter]:
    """
    只更新最后的 fc（你 baseline 里叫 self.fc）
    """
    freeze_all(model)
    params = []
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        for p in model.fc.parameters():
            p.requires_grad_(True)
            params.append(p)
    else:
        raise RuntimeError("模型中未找到 fc 层（CNNBaseline.fc）")
    return params


@torch.no_grad()
def update_teacher(teacher: nn.Module, student: nn.Module, ema: float) -> None:
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(ema).add_(sp.data, alpha=(1.0 - ema))


def run_one_subject(
    subject_id: int,
    ckpt_path: str,
    data_root: str,
    window_size: int,
    step_size: int,
    device: torch.device,
    out_dir: str,
    cfg: OnlineCfg,
    block_size: int = 50,
    split_ratio: Tuple[float, float, float] = (0.2, 0.4, 0.4),
) -> Dict:
    """
    跑单个 subject 单个配置
    输出：
      - per_step.csv
      - final_eval.csv
      - meta.json
    返回 summary dict（写入 grid_summary.csv）
    """
    set_seed(cfg.seed)

    exp_key = f"S{subject_id}_{cfg.mode}_lr{cfg.lr}_th{cfg.conf_th}_buf{cfg.buffer_max}_ue{cfg.update_every}_bs{cfg.batch_size}_tea{int(cfg.use_teacher)}"
    run_dir = os.path.join(out_dir, exp_key + "_" + now_ts())
    ensure_dir(run_dir)

    # 1) build dataset + streaming split
    dataset, idx_pre, idx_adapt, idx_eval = build_subject_blocks(
        root=data_root,
        subject_id=subject_id,
        window_size=window_size,
        step_size=step_size,
        num_classes=NUM_CLASSES,
        normalize=True,
        block_size=block_size,
        split_ratio=split_ratio,
    )

    # 2) load model
    model = CNNBaseline(in_channels=8, num_classes=NUM_CLASSES).to(device)
    sd = load_checkpoint_any(ckpt_path, device)
    model.load_state_dict(sd, strict=True)

    teacher = None
    if cfg.use_teacher:
        teacher = CNNBaseline(in_channels=8, num_classes=NUM_CLASSES).to(device)
        teacher.load_state_dict(sd, strict=True)
        teacher.eval()
        freeze_all(teacher)

    # 3) prepare optimizer if needed
    if cfg.mode in ["online_head", "lr0"]:
        params = unfreeze_head_only(model)
        opt = torch.optim.Adam(params, lr=float(cfg.lr))
        if cfg.mode == "lr0":
            for g in opt.param_groups:
                g["lr"] = 0.0
    else:
        freeze_all(model)
        opt = None

    # 4) loaders for streaming
    # pre: 只统计/前向，不更新（这里先不做额外动作，保留你后续扩展）
    # adapt: 顺序流式
    # eval: 最终评估
    pre_loader = DataLoader(Subset(dataset, idx_pre), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    adapt_loader = DataLoader(Subset(dataset, idx_adapt), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # 5) logs
    per_step_csv = os.path.join(run_dir, "per_step.csv")
    final_eval_csv = os.path.join(run_dir, "final_eval.csv")
    meta_json = os.path.join(run_dir, "meta.json")

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_key": exp_key,
                "subject_id": subject_id,
                "ckpt_path": ckpt_path,
                "data_root": data_root,
                "window_size": window_size,
                "step_size": step_size,
                "device": str(device),
                "cfg": asdict(cfg),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "time": time.time(),
                "split_sizes": {"pre": len(idx_pre), "adapt": len(idx_adapt), "eval": len(idx_eval)},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 6) Stage: pre (no update)
    model.eval()
    pre_seen = 0
    for xb, yb in pre_loader:
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
        pre_seen += xb.shape[0]

    # 7) Stage: adapt (maybe update)
    buf = TinyBuffer(cfg.buffer_max)
    t0 = time.time()
    step_rows = []
    model.train() if cfg.mode in ["online_head", "lr0"] else model.eval()

    adapt_seen = 0
    accepted = 0
    updates = 0

    # 记录：每个 batch 的即时表现（在 adapt 段上），以及是否发生更新
    for bi, (xb, yb) in enumerate(adapt_loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # forward
        logits = model(xb)
        prob = F.softmax(logits, dim=1)
        conf, pred = torch.max(prob, dim=1)

        # 当前 batch 的“即时”acc（仅用于观察流式变化）
        batch_acc = float((pred == yb).float().mean().item())
        batch_loss = float(F.cross_entropy(logits, yb).item())

        # pseudo-label from teacher or self
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(xb)
                t_prob = F.softmax(t_logits, dim=1)
                t_conf, t_pred = torch.max(t_prob, dim=1)
            gate_conf = t_conf
            pseudo_y = t_pred
        else:
            gate_conf = conf
            pseudo_y = pred

        # gate
        mask = gate_conf >= float(cfg.conf_th)
        n_accept = int(mask.sum().item())

        if cfg.mode in ["online_head", "lr0"] and n_accept > 0:
            accepted += n_accept
            buf.add(xb[mask], pseudo_y[mask])

        did_update = 0
        if cfg.mode in ["online_head", "lr0"] and (bi + 1) % int(cfg.update_every) == 0:
            xb_u, yb_u = buf.sample_all(device)
            if xb_u is not None and xb_u.shape[0] > 0:
                model.train()
                opt.zero_grad(set_to_none=True)
                out_u = model(xb_u)
                loss_u = F.cross_entropy(out_u, yb_u)
                loss_u.backward()
                opt.step()
                updates += 1
                did_update = 1

                # update teacher
                if teacher is not None:
                    update_teacher(teacher, model, ema=float(cfg.ema))

        adapt_seen += xb.shape[0]
        step_rows.append(
            {
                "batch_idx": bi,
                "adapt_seen": adapt_seen,
                "batch_size": int(xb.shape[0]),
                "batch_loss_true": batch_loss,
                "batch_acc_true": batch_acc,
                "accept_n": n_accept,
                "accept_ratio": float(n_accept / max(int(xb.shape[0]), 1)),
                "buffer_size": int(len(buf)),
                "did_update": did_update,
                "updates_total": updates,
                "accepted_total": accepted,
                "unix_time": time.time(),
            }
        )

    t_adapt = time.time() - t0

    # 8) Final eval (no update)
    freeze_all(model)
    model.eval()
    ev_loss, ev_acc, ev_n, ev_conf = eval_subset(model, dataset, idx_eval, device, batch_size=64)
    ev_rec = per_class_recall(ev_conf)

    # 9) write csvs
    # per_step.csv
    with open(per_step_csv, "w", encoding="utf-8") as f:
        cols = list(step_rows[0].keys()) if step_rows else []
        f.write(",".join(cols) + "\n")
        for r in step_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # final_eval.csv
    with open(final_eval_csv, "w", encoding="utf-8") as f:
        f.write("subject,mode,lr,conf_th,buffer_max,update_every,batch_size,use_teacher,ema,eval_n,eval_loss,eval_acc,recall_c0,recall_c1,recall_c2,adapt_wall_time_sec,accepted_total,updates_total\n")
        f.write(
            "{},{},{},{},{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.3f},{},{}\n".format(
                subject_id,
                cfg.mode,
                cfg.lr,
                cfg.conf_th,
                cfg.buffer_max,
                cfg.update_every,
                cfg.batch_size,
                int(cfg.use_teacher),
                cfg.ema,
                ev_n,
                ev_loss,
                ev_acc,
                ev_rec[0],
                ev_rec[1],
                ev_rec[2],
                t_adapt,
                accepted,
                updates,
            )
        )

    summary = {
        "exp_key": exp_key,
        "run_dir": run_dir,
        "subject": subject_id,
        "mode": cfg.mode,
        "lr": cfg.lr,
        "conf_th": cfg.conf_th,
        "buffer_max": cfg.buffer_max,
        "update_every": cfg.update_every,
        "batch_size": cfg.batch_size,
        "use_teacher": int(cfg.use_teacher),
        "ema": cfg.ema,
        "split_pre": len(idx_pre),
        "split_adapt": len(idx_adapt),
        "split_eval": len(idx_eval),
        "accepted_total": accepted,
        "updates_total": updates,
        "adapt_wall_time_sec": float(t_adapt),
        "eval_n": int(ev_n),
        "eval_loss": float(ev_loss),
        "eval_acc": float(ev_acc),
        "recall_c0": float(ev_rec[0]),
        "recall_c1": float(ev_rec[1]),
        "recall_c2": float(ev_rec[2]),
        "unix_time_min": min([r["unix_time"] for r in step_rows], default=time.time()),
        "unix_time_max": max([r["unix_time"] for r in step_rows], default=time.time()),
    }
    return summary


# =========================
# Grid runner
# =========================

def parse_list(s: str, typ=float) -> List:
    s = s.strip()
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if typ is float:
            out.append(float(x))
        elif typ is int:
            out.append(int(x))
        else:
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="WESAD 根目录，例如 /workspace/data/WESAD")
    ap.add_argument("--ckpt", required=True, help="baseline checkpoint path (.pt)")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--subject-ids", default="2,9,17", help="例如 2,9,17")
    ap.add_argument("--window-size", type=int, default=700)
    ap.add_argument("--step-size", type=int, default=350)
    ap.add_argument("--out-dir", default="", help="日志输出目录，默认 ../logs/verifyA_grid/")

    # grid params
    ap.add_argument("--lrs", default="1e-4,5e-4,1e-3", help="逗号分隔")
    ap.add_argument("--conf-ths", default="0.8,0.9,0.95", help="逗号分隔")
    ap.add_argument("--buffer-maxs", default="128,256,512", help="逗号分隔")
    ap.add_argument("--update-everys", default="5,10,20", help="逗号分隔（每几个 adapt batch 更新一次）")
    ap.add_argument("--batch-sizes", default="32", help="逗号分隔")
    ap.add_argument("--use-teacher", default="1", choices=["0", "1"])
    ap.add_argument("--ema", type=float, default=0.99)

    # fixed
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--block-size", type=int, default=50)
    ap.add_argument("--split-ratio", default="0.2,0.4,0.4")

    args = ap.parse_args()

    out_dir = args.out_dir.strip() or safe_rel_log_dir("verifyA_grid")
    ensure_dir(out_dir)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    subject_ids = parse_list(args.subject_ids, int)
    lrs = parse_list(args.lrs, float)
    conf_ths = parse_list(args.conf_ths, float)
    buffer_maxs = parse_list(args.buffer_maxs, int)
    update_everys = parse_list(args.update_everys, int)
    batch_sizes = parse_list(args.batch_sizes, int)
    use_teacher = bool(int(args.use_teacher))
    split_ratio = tuple(parse_list(args.split_ratio, float))
    assert len(split_ratio) == 3 and abs(sum(split_ratio) - 1.0) < 1e-6

    # 写汇总
    grid_summary_csv = os.path.join(out_dir, f"grid_summary_{now_ts()}.csv")

    summaries = []

    # 必做的三条验证：frozen + lr0 + online
    # frozen / lr0 不走 grid，只跑一组固定配置（保证 sanity check）
    sanity_cfgs = [
        OnlineCfg(mode="frozen", lr=0.0, conf_th=1.0, buffer_max=0, update_every=999999, batch_size=64, seed=args.seed, use_teacher=False, ema=args.ema),
        OnlineCfg(mode="lr0", lr=1e-3, conf_th=0.0, buffer_max=512, update_every=5, batch_size=64, seed=args.seed, use_teacher=use_teacher, ema=args.ema),
    ]

    for sid in subject_ids:
        for cfg in sanity_cfgs:
            summaries.append(
                run_one_subject(
                    subject_id=sid,
                    ckpt_path=args.ckpt,
                    data_root=args.data_root,
                    window_size=args.window_size,
                    step_size=args.step_size,
                    device=device,
                    out_dir=out_dir,
                    cfg=cfg,
                    block_size=args.block_size,
                    split_ratio=split_ratio,
                )
            )

    # online grid
    for sid in subject_ids:
        for lr in lrs:
            for th in conf_ths:
                for bm in buffer_maxs:
                    for ue in update_everys:
                        for bs in batch_sizes:
                            cfg = OnlineCfg(
                                mode="online_head",
                                lr=float(lr),
                                conf_th=float(th),
                                buffer_max=int(bm),
                                update_every=int(ue),
                                batch_size=int(bs),
                                seed=args.seed,
                                use_teacher=use_teacher,
                                ema=float(args.ema),
                            )
                            summaries.append(
                                run_one_subject(
                                    subject_id=sid,
                                    ckpt_path=args.ckpt,
                                    data_root=args.data_root,
                                    window_size=args.window_size,
                                    step_size=args.step_size,
                                    device=device,
                                    out_dir=out_dir,
                                    cfg=cfg,
                                    block_size=args.block_size,
                                    split_ratio=split_ratio,
                                )
                            )

    # 写 grid_summary.csv
    cols = list(summaries[0].keys()) if summaries else []
    with open(grid_summary_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in summaries:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print("\n[Done] All runs finished.")
    print("  out_dir         :", out_dir)
    print("  grid_summary_csv:", grid_summary_csv)
    print("  Each run has per_step.csv + final_eval.csv + meta.json in its run_dir.")


if __name__ == "__main__":
    main()

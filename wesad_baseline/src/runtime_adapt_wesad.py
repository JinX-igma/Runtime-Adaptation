#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import json
import platform
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# 你已有的 WESAD dataset
# 请按你项目真实路径修改
from data.wesad_dataset import WESADDataset

# 你已有的模型
# 请按你项目真实路径修改
from models.cnn_baseline import CNNBaseline


NUM_CLASSES = 3


# =========================================================
# 工具函数：指标，日志
# =========================================================

def now_unix() -> float:
    return time.time()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p: [B, C]
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=1)  # [B]


def conf_from_probs(p: torch.Tensor) -> torch.Tensor:
    return torch.max(p, dim=1)[0]  # [B]


def confusion_matrix(pred: np.ndarray, y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pi, yi in zip(pred, y):
        if 0 <= yi < num_classes and 0 <= pi < num_classes:
            cm[int(yi), int(pi)] += 1
    return cm


def per_class_recall(cm: np.ndarray) -> List[float]:
    # recall_i = TP_i / (TP_i + FN_i) = cm[i,i] / sum_row_i
    recalls = []
    for i in range(cm.shape[0]):
        denom = cm[i, :].sum()
        if denom <= 0:
            recalls.append(0.0)
        else:
            recalls.append(float(cm[i, i] / denom))
    return recalls


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 3
) -> Tuple[float, float, int, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_pred = []
    all_y = []

    with torch.no_grad():
        for batch in loader:
            # 兼容你 dataset 返回 (x, y) 或 dict
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x, y = batch["x"], batch["y"]
            else:
                raise RuntimeError("Unknown batch format from dataset")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")

            total_loss += float(loss.item())
            total += int(y.numel())

            pred = torch.argmax(logits, dim=1)
            all_pred.append(pred.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())

    if total == 0:
        return 0.0, 0.0, 0, np.zeros((num_classes, num_classes), dtype=np.int64)

    all_pred = np.concatenate(all_pred, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    cm = confusion_matrix(all_pred, all_y, num_classes=num_classes)
    acc = float((all_pred == all_y).mean())
    mean_loss = total_loss / float(total)
    return mean_loss, acc, total, cm


def write_csv_header(path: str, cols: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")


def append_csv_row(path: str, row: Dict[str, object], cols: List[str]):
    with open(path, "a", encoding="utf-8") as f:
        vals = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        f.write(",".join(vals) + "\n")


# =========================================================
# 参数过滤与冻结
# =========================================================

def get_bn_modules(model: nn.Module) -> List[nn.BatchNorm1d]:
    bns = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            bns.append(m)
    return bns


def freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_bn_affine_only(model: nn.Module):
    # 只解冻 BN 的 weight 与 bias
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True


def unfreeze_head_and_bn(model: nn.Module, head_name: str = "fc"):
    # 解冻分类头与 BN affine
    for name, p in model.named_parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    # 假设你的 head 是 model.fc
    head = getattr(model, head_name, None)
    if head is None:
        raise RuntimeError(f"model has no head named {head_name}")
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_last_conv_head_bn(model: nn.Module, last_conv_name: str = "conv3", head_name: str = "fc"):
    # 解冻最后一层卷积 + head + BN affine
    for name, p in model.named_parameters():
        p.requires_grad = False

    # BN affine
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    # last conv
    last_conv = getattr(model, last_conv_name, None)
    if last_conv is None:
        raise RuntimeError(f"model has no conv named {last_conv_name}")
    for p in last_conv.parameters():
        p.requires_grad = True

    # head
    head = getattr(model, head_name, None)
    if head is None:
        raise RuntimeError(f"model has no head named {head_name}")
    for p in head.parameters():
        p.requires_grad = True


def collect_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# =========================================================
# 你提出的分段索引构建
# 这里直接复用你已经写好的 build_subject_blocks
# 你如果要替换为 build_eval_update_indices 也行
# =========================================================

from typing import Dict as _Dict, List as _List, Tuple as _Tuple
from collections import namedtuple, defaultdict

Block = namedtuple("Block", ["start", "end", "major_label"])


def build_subject_blocks(
    root: str,
    subject_id: int,
    window_size: int = 700,
    step_size: int = 350,
    num_classes: int = 3,
    normalize: bool = True,
    block_size: int = 50,
    split_ratio: _Tuple[float, float, float] = (0.2, 0.4, 0.4),
):
    dataset = WESADDataset(
        root=root,
        subject_ids=[subject_id],
        window_size=window_size,
        step_size=step_size,
        num_classes=num_classes,
        normalize=normalize,
    )

    n = len(dataset)
    r_pre, r_adapt, r_eval = split_ratio
    assert abs(r_pre + r_adapt + r_eval - 1.0) < 1e-6

    blocks: _List[Block] = []
    num_blocks = (n + block_size - 1) // block_size

    labels = [dataset.get_label(i) for i in range(n)]

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, n)
        cnt = [0] * num_classes
        for i in range(start, end):
            y = labels[i]
            if 0 <= y < num_classes:
                cnt[y] += 1
        major_label = int(max(range(num_classes), key=lambda c: cnt[c]))
        blocks.append(Block(start=start, end=end, major_label=major_label))

    label_to_blocks: _Dict[int, _List[Block]] = defaultdict(list)
    for blk in blocks:
        label_to_blocks[blk.major_label].append(blk)

    pre_blocks: _List[Block] = []
    adapt_blocks: _List[Block] = []
    eval_blocks: _List[Block] = []

    for c in range(num_classes):
        blks = label_to_blocks[c]
        if not blks:
            continue
        k = len(blks)
        k_pre = int(k * r_pre)
        k_adapt = int(k * r_adapt)
        pre_blocks.extend(blks[:k_pre])
        adapt_blocks.extend(blks[k_pre:k_pre + k_adapt])
        eval_blocks.extend(blks[k_pre + k_adapt:])

    def expand_blocks(blks: _List[Block]) -> _List[int]:
        idx = []
        for blk in blks:
            idx.extend(list(range(blk.start, blk.end)))
        return idx

    idx_pre = expand_blocks(pre_blocks)
    idx_adapt = expand_blocks(adapt_blocks)
    idx_eval = expand_blocks(eval_blocks)

    return dataset, idx_pre, idx_adapt, idx_eval


# =========================================================
# 算法 1：AdaBN 预热 + TENT 在线微调
# =========================================================

@dataclass
class AdaBNTENTConfig:
    # AdaBN
    warmup_batches: int = 50
    # TENT
    lr: float = 2e-4
    steps_per_batch: int = 1
    entropy_th: float = 0.8
    batch_size: int = 32
    # 日志
    eval_every_updates: int = 50


def adabn_warmup(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_batches: int = 50
):
    # 只更新 BN running_mean 与 running_var
    # 不能更新任何可学习参数
    model.train()
    freeze_all_params(model)

    # 确保 BN tracking 开启
    for bn in get_bn_modules(model):
        bn.track_running_stats = True

    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch["x"]
            else:
                raise RuntimeError("Unknown batch format")

            x = x.to(device, non_blocking=True)
            _ = model(x)
            n_batches += 1
            if n_batches >= warmup_batches:
                break


def tent_online(
    model: nn.Module,
    adapt_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    cfg: AdaBNTENTConfig,
    log_csv_path: str,
    exp_meta: Dict[str, object]
):
    # 只更新 BN gamma beta
    model.train()
    freeze_all_params(model)
    unfreeze_bn_affine_only(model)

    params = collect_trainable_params(model)
    opt = torch.optim.Adam(params, lr=cfg.lr)

    cols = [
        "unix_time",
        "update_step",
        "batch_entropy_mean",
        "used_ratio",
        "eval_loss",
        "eval_acc",
        "eval_recall_c0",
        "eval_recall_c1",
        "eval_recall_c2",
        "meta_json"
    ]
    write_csv_header(log_csv_path, cols)

    update_step = 0
    meta_json = json.dumps(exp_meta, ensure_ascii=False)

    for batch in adapt_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch["x"]
        else:
            raise RuntimeError("Unknown batch format")

        x = x.to(device, non_blocking=True)

        # forward
        logits = model(x)
        p = softmax_probs(logits)
        ent = entropy_from_probs(p)  # [B]
        ent_mean = float(ent.mean().item())

        # 置信或低熵筛选
        use_mask = (ent < cfg.entropy_th)
        used = int(use_mask.sum().item())
        total = int(use_mask.numel())
        used_ratio = used / float(max(total, 1))

        # 如果一个都没选中，则跳过更新
        if used > 0:
            for _ in range(cfg.steps_per_batch):
                opt.zero_grad(set_to_none=True)
                logits2 = model(x)
                p2 = softmax_probs(logits2)
                ent2 = entropy_from_probs(p2)
                ent_used = ent2[use_mask]
                loss = ent_used.mean()
                loss.backward()
                opt.step()

        update_step += 1

        # 周期性评估
        if (update_step % cfg.eval_every_updates) == 0:
            eval_loss, eval_acc, n_s, cm = evaluate_model(model, eval_loader, device, num_classes=NUM_CLASSES)
            rec = per_class_recall(cm)

            row = {
                "unix_time": now_unix(),
                "update_step": update_step,
                "batch_entropy_mean": ent_mean,
                "used_ratio": used_ratio,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "eval_recall_c0": rec[0],
                "eval_recall_c1": rec[1],
                "eval_recall_c2": rec[2],
                "meta_json": meta_json
            }
            append_csv_row(log_csv_path, row, cols)

    # 结束再评估一次
    eval_loss, eval_acc, n_s, cm = evaluate_model(model, eval_loader, device, num_classes=NUM_CLASSES)
    rec = per_class_recall(cm)
    row = {
        "unix_time": now_unix(),
        "update_step": update_step,
        "batch_entropy_mean": -1.0,
        "used_ratio": -1.0,
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_recall_c0": rec[0],
        "eval_recall_c1": rec[1],
        "eval_recall_c2": rec[2],
        "meta_json": meta_json
    }
    append_csv_row(log_csv_path, row, cols)


def run_adabn_tent_for_subject(
    root: str,
    ckpt_path: str,
    subject_id: int,
    window_size: int,
    step_size: int,
    out_dir: str,
    device: str = "cuda",
    cfg: Optional[AdaBNTENTConfig] = None
):
    if cfg is None:
        cfg = AdaBNTENTConfig()

    dev = torch.device(device)

    # dataset + split indices
    dataset, idx_pre, idx_adapt, idx_eval = build_subject_blocks(
        root=root,
        subject_id=subject_id,
        window_size=window_size,
        step_size=step_size,
        num_classes=NUM_CLASSES,
        normalize=True,
        block_size=50,
        split_ratio=(0.2, 0.4, 0.4)
    )

    ds_pre = Subset(dataset, idx_pre)
    ds_adapt = Subset(dataset, idx_adapt)
    ds_eval = Subset(dataset, idx_eval)

    # loaders
    pre_loader = DataLoader(ds_pre, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    adapt_loader = DataLoader(ds_adapt, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(ds_eval, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # model
    model = CNNBaseline(in_channels=8, num_classes=NUM_CLASSES)
    state = torch.load(ckpt_path, map_location="cpu")
    # 兼容你保存的是纯 state_dict 或完整 checkpoint
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(dev)

    ensure_dir(out_dir)
    exp_key = f"AdaBN_TENT_S{subject_id}_ws{window_size}_ss{step_size}_th{int(cfg.entropy_th*100)}_lr{cfg.lr}"
    log_csv = os.path.join(out_dir, f"exp_{time.strftime('%Y%m%d_%H%M%S')}_{exp_key}.csv")

    exp_meta = {
        "algo": "AdaBN+TENT",
        "subject": f"S{subject_id}",
        "window_size": window_size,
        "step_size": step_size,
        "ckpt": os.path.basename(ckpt_path),
        "device": str(dev),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cfg": cfg.__dict__
    }

    # 1 AdaBN warmup
    adabn_warmup(model, pre_loader, dev, warmup_batches=cfg.warmup_batches)

    # 2 TENT online with periodic eval on eval subset
    tent_online(model, adapt_loader, eval_loader, dev, cfg, log_csv, exp_meta)

    print("Done:", log_csv)


# =========================================================
# 算法 2：Temporal Consistency Adaptation
# 支持无监督，支持半监督少量标签
# 支持 EMA teacher 与置信门控
# =========================================================

@dataclass
class TemporalConsistencyConfig:
    lr: float = 1e-4
    batch_size: int = 32
    update_variant: str = "head_bn"  # head_bn 或 lastconv_head_bn
    steps_per_batch: int = 1

    # consistency
    lam_smooth: float = 0.05
    use_ema_teacher: bool = True
    ema_decay: float = 0.99

    # gating
    conf_th: float = 0.7

    # semi supervised
    use_labels: bool = False
    alpha_ce: float = 0.1
    labeled_ratio: float = 0.02

    # eval schedule
    eval_every_updates: int = 50


def clone_model(model: nn.Module) -> nn.Module:
    import copy
    m2 = copy.deepcopy(model)
    for p in m2.parameters():
        p.requires_grad = False
    return m2


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float):
    td = teacher.state_dict()
    sd = student.state_dict()
    for k in td.keys():
        if k in sd:
            td[k].mul_(decay).add_(sd[k], alpha=(1.0 - decay))
    teacher.load_state_dict(td)


def temporal_consistency_loss(
    student_logits_t: torch.Tensor,
    teacher_logits_tp1: torch.Tensor,
    smooth_on_logits: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    # symmetric KL in prob space
    ps = softmax_probs(student_logits_t)
    pt = softmax_probs(teacher_logits_tp1)

    kl1 = F.kl_div(torch.log(torch.clamp(ps, 1e-8, 1.0)), pt, reduction="batchmean")
    kl2 = F.kl_div(torch.log(torch.clamp(pt, 1e-8, 1.0)), ps, reduction="batchmean")
    l_cons = kl1 + kl2

    if smooth_on_logits:
        l_smooth = torch.mean(torch.abs(teacher_logits_tp1 - student_logits_t))
    else:
        l_smooth = torch.tensor(0.0, device=student_logits_t.device)

    return l_cons, l_smooth


def sample_labeled_mask(batch_size: int, labeled_ratio: float, device: torch.device) -> torch.Tensor:
    # 随机挑选少量样本作为 labeled
    k = max(1, int(batch_size * labeled_ratio))
    idx = torch.randperm(batch_size, device=device)[:k]
    mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    mask[idx] = True
    return mask


def run_temporal_consistency_for_subject(
    root: str,
    ckpt_path: str,
    subject_id: int,
    window_size: int,
    step_size: int,
    out_dir: str,
    device: str = "cuda",
    cfg: Optional[TemporalConsistencyConfig] = None
):
    if cfg is None:
        cfg = TemporalConsistencyConfig()

    dev = torch.device(device)

    dataset, idx_pre, idx_adapt, idx_eval = build_subject_blocks(
        root=root,
        subject_id=subject_id,
        window_size=window_size,
        step_size=step_size,
        num_classes=NUM_CLASSES,
        normalize=True,
        block_size=50,
        split_ratio=(0.0, 0.6, 0.4)  # 你不想浪费 pre 就设 0.0
    )

    ds_adapt = Subset(dataset, idx_adapt)
    ds_eval = Subset(dataset, idx_eval)

    adapt_loader = DataLoader(ds_adapt, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(ds_eval, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # student model
    model = CNNBaseline(in_channels=8, num_classes=NUM_CLASSES)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(dev)

    # set trainable params
    model.train()
    if cfg.update_variant == "head_bn":
        unfreeze_head_and_bn(model, head_name="fc")
    elif cfg.update_variant == "lastconv_head_bn":
        unfreeze_last_conv_head_bn(model, last_conv_name="conv3", head_name="fc")
    else:
        raise RuntimeError("unknown update_variant")

    opt = torch.optim.Adam(collect_trainable_params(model), lr=cfg.lr)

    # teacher
    teacher = None
    if cfg.use_ema_teacher:
        teacher = clone_model(model).to(dev)
        teacher.eval()

    ensure_dir(out_dir)
    exp_key = f"TempCons_S{subject_id}_ws{window_size}_ss{step_size}_{cfg.update_variant}_lr{cfg.lr}"
    log_csv = os.path.join(out_dir, f"exp_{time.strftime('%Y%m%d_%H%M%S')}_{exp_key}.csv")

    cols = [
        "unix_time",
        "update_step",
        "loss_cons",
        "loss_smooth",
        "loss_ce",
        "used_ratio",
        "eval_loss",
        "eval_acc",
        "eval_recall_c0",
        "eval_recall_c1",
        "eval_recall_c2",
        "meta_json"
    ]
    write_csv_header(log_csv, cols)

    exp_meta = {
        "algo": "TemporalConsistency",
        "subject": f"S{subject_id}",
        "window_size": window_size,
        "step_size": step_size,
        "ckpt": os.path.basename(ckpt_path),
        "device": str(dev),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cfg": cfg.__dict__
    }
    meta_json = json.dumps(exp_meta, ensure_ascii=False)

    update_step = 0

    # streaming 训练核心
    # 对每个 batch，构造相邻对 x_t 与 x_{t+1}
    for batch in adapt_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            raise RuntimeError("Unknown batch format")

        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True).long()

        B = x.shape[0]
        if B < 2:
            continue

        # 形成相邻对
        x_t = x[:-1]
        x_tp1 = x[1:]
        y_t = y[:-1]

        # teacher 输出用于稳定
        if teacher is not None:
            with torch.no_grad():
                teacher_logits_tp1 = teacher(x_tp1)
                pt = softmax_probs(teacher_logits_tp1)
                conf = conf_from_probs(pt)
                gate = conf > cfg.conf_th  # [B-1]
        else:
            with torch.no_grad():
                tmp_logits = model(x_tp1)
                pt = softmax_probs(tmp_logits)
                conf = conf_from_probs(pt)
                gate = conf > cfg.conf_th
            teacher_logits_tp1 = tmp_logits.detach()

        used = int(gate.sum().item())
        total = int(gate.numel())
        used_ratio = used / float(max(total, 1))
        if used == 0:
            update_step += 1
            continue

        # student 输出
        student_logits_t = model(x_t)

        # 只对通过 gate 的样本计算一致性
        sl = student_logits_t[gate]
        tl = teacher_logits_tp1[gate]

        l_cons, l_smooth = temporal_consistency_loss(sl, tl, smooth_on_logits=True)
        loss = l_cons + cfg.lam_smooth * l_smooth

        l_ce_val = torch.tensor(0.0, device=dev)
        if cfg.use_labels:
            # 半监督，随机抽少量标签
            mask_lab = sample_labeled_mask(sl.shape[0], cfg.labeled_ratio, dev)
            if int(mask_lab.sum().item()) > 0:
                # 这里用 y_t 对应 gate 后再采样
                y_used = y_t[gate]
                y_lab = y_used[mask_lab]
                p_lab = sl[mask_lab]
                l_ce_val = F.cross_entropy(p_lab, y_lab)
                loss = loss + cfg.alpha_ce * l_ce_val

        # 更新
        for _ in range(cfg.steps_per_batch):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # 更新 teacher
        if teacher is not None:
            ema_update(teacher, model, decay=cfg.ema_decay)

        update_step += 1

        # 周期性评估
        if (update_step % cfg.eval_every_updates) == 0:
            eval_loss, eval_acc, n_s, cm = evaluate_model(model, eval_loader, dev, num_classes=NUM_CLASSES)
            rec = per_class_recall(cm)

            row = {
                "unix_time": now_unix(),
                "update_step": update_step,
                "loss_cons": float(l_cons.item()),
                "loss_smooth": float(l_smooth.item()),
                "loss_ce": float(l_ce_val.item()) if cfg.use_labels else 0.0,
                "used_ratio": used_ratio,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "eval_recall_c0": rec[0],
                "eval_recall_c1": rec[1],
                "eval_recall_c2": rec[2],
                "meta_json": meta_json
            }
            append_csv_row(log_csv, row, cols)

    # 结束再评估一次
    eval_loss, eval_acc, n_s, cm = evaluate_model(model, eval_loader, dev, num_classes=NUM_CLASSES)
    rec = per_class_recall(cm)
    row = {
        "unix_time": now_unix(),
        "update_step": update_step,
        "loss_cons": -1.0,
        "loss_smooth": -1.0,
        "loss_ce": -1.0,
        "used_ratio": -1.0,
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_recall_c0": rec[0],
        "eval_recall_c1": rec[1],
        "eval_recall_c2": rec[2],
        "meta_json": meta_json
    }
    append_csv_row(log_csv, row, cols)

    print("Done:", log_csv)


# =========================================================
# 主入口示例
# =========================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, required=True, choices=["adabn_tent", "temp_cons"])
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--ckpt-path", type=str, required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--window-size", type=int, default=700)
    ap.add_argument("--step-size", type=int, default=350)
    ap.add_argument("--out-dir", type=str, default="logs")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.algo == "adabn_tent":
        cfg = AdaBNTENTConfig(
            warmup_batches=50,
            lr=2e-4,
            steps_per_batch=1,
            entropy_th=0.8,
            batch_size=32,
            eval_every_updates=50
        )
        run_adabn_tent_for_subject(
            root=args.data_root,
            ckpt_path=args.ckpt_path,
            subject_id=args.subject,
            window_size=args.window_size,
            step_size=args.step_size,
            out_dir=args.out_dir,
            device=args.device,
            cfg=cfg
        )
    else:
        cfg = TemporalConsistencyConfig(
            lr=1e-4,
            batch_size=32,
            update_variant="head_bn",
            steps_per_batch=1,
            lam_smooth=0.05,
            use_ema_teacher=True,
            ema_decay=0.99,
            conf_th=0.7,
            use_labels=False,
            alpha_ce=0.1,
            labeled_ratio=0.02,
            eval_every_updates=50
        )
        run_temporal_consistency_for_subject(
            root=args.data_root,
            ckpt_path=args.ckpt_path,
            subject_id=args.subject,
            window_size=args.window_size,
            step_size=args.step_size,
            out_dir=args.out_dir,
            device=args.device,
            cfg=cfg
        )


if __name__ == "__main__":
    main()

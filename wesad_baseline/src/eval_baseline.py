#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_baseline_checkpoint.py

使用訓練好的 CNNBaseline checkpoint
在每個 subject 上做完整評估，並記錄每類別的表現。

用法示例:
  python3 eval_baseline_checkpoint.py \
    --data-root /workspace/data/WESAD \
    --ckpt-path /workspace/src/checkpoints/exp_20251210_baseline_cnn.pt \
    --batch-size 32 \
    --device cuda
"""

import argparse
import os
from datetime import datetime
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.wesad_dataset import WESADDataset
from models.cnn_baseline import CNNBaseline


# 所有 WESAD subjects
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
NUM_CLASSES = 3


def create_experiment_id(prefix: str = "baseline_eval") -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"exp_{now}_{prefix}"


def create_logger(exp_id: str, log_dir: str = None):
    """
    log_dir 默認為項目根目錄下的 logs
    例如 /workspace/logs
    """
    if log_dir is None:
        project_root = os.path.dirname(os.path.dirname(__file__))  # src 的上一級
        log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_id}.log")
    f = open(log_path, "w", encoding="utf-8")
    return f, log_path


def log(f, msg: str):
    print(msg)
    if f is not None:
        f.write(msg + "\n")
        f.flush()


def build_dataset_for_subjects(
    data_root: str,
    subject_ids,
    window_size: int,
    step_size: int,
    num_classes: int = NUM_CLASSES,
    normalize: bool = True,
):
    dataset = WESADDataset(
        root=data_root,
        subject_ids=subject_ids,
        window_size=window_size,
        step_size=step_size,
        num_classes=num_classes,
        normalize=normalize,
    )
    return dataset


def evaluate_model(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

            preds_np = preds.cpu().numpy().reshape(-1)
            y_np = y.cpu().numpy().reshape(-1)
            for t, p in zip(y_np, preds_np):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    conf_mat[t, p] += 1

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc, total_samples, conf_mat


def per_class_recall_from_conf(conf_mat: np.ndarray):
    num_classes = conf_mat.shape[0]
    recalls = []
    for c in range(num_classes):
        true_c = conf_mat[c, c]
        total_c = conf_mat[c, :].sum()
        if total_c > 0:
            recalls.append(true_c / float(total_c))
        else:
            recalls.append(0.0)
    return recalls


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CNNBaseline checkpoint on each WESAD subject"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="WESAD 根目錄，如 /workspace/data/WESAD",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="訓練好的 checkpoint 路徑 (包含 model_state 和 config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="評估的 batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="評估設備 cpu 或 cuda",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="滑窗大小。如不指定則從 ckpt.config 中讀取",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="滑窗步長。如不指定則從 ckpt.config 中讀取",
    )
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 加載 checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    config = ckpt.get("config", {})

    # 決定 window_size 和 step_size
    window_size = args.window_size or config.get("window_size", 700)
    step_size = args.step_size or config.get("step_size", 350)

    exp_id = create_experiment_id("baseline_eval")
    f, log_path = create_logger(exp_id)

    log(f, "========== Baseline Checkpoint Evaluation ==========")
    log(f, f"Experiment id    : {exp_id}")
    log(f, f"Log path         : {log_path}")
    log(f, f"Checkpoint path  : {args.ckpt_path}")
    log(f, f"Device           : {device}")
    log(f, f"Python version   : {platform.python_version()}")
    log(f, f"PyTorch version  : {torch.__version__}")
    log(f, "")
    log(f, "Config from checkpoint (if available):")
    log(f, f"  data_root      : {config.get('data_root', 'N/A')}")
    log(f, f"  batch_size     : {config.get('batch_size', 'N/A')}")
    log(f, f"  epochs         : {config.get('epochs', 'N/A')}")
    log(f, f"  lr             : {config.get('lr', 'N/A')}")
    log(f, f"  weight_decay   : {config.get('weight_decay', 'N/A')}")
    log(f, f"  window_size    : {window_size}")
    log(f, f"  step_size      : {step_size}")
    log(f, "")

    # 構建模型並加載權重
    in_channels = 8
    num_classes = NUM_CLASSES
    model = CNNBaseline(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f, "Model information")
    log(f, f"  Total params   : {total_params}")
    log(f, "")

    data_root = args.data_root
    batch_size = args.batch_size

    start_time = time.time()

    # 用於整體統計的混淆矩陣
    conf_all = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    per_subject_results = []  # (sid, loss, acc, n, per_class_recall)

    log(f, "Start subject wise evaluation...")
    log(f, f"Subjects to evaluate: {ALL_SUBJECTS}")
    log(f, "")

    for sid in ALL_SUBJECTS:
        subj_str = f"S{sid}"
        log(f, f"[Eval] Subject {subj_str}")

        dataset = build_dataset_for_subjects(
            data_root=data_root,
            subject_ids=[sid],
            window_size=window_size,
            step_size=step_size,
            num_classes=NUM_CLASSES,
            normalize=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        loss, acc, n_samples, conf_mat = evaluate_model(
            model, loader, device, num_classes=NUM_CLASSES
        )
        conf_all += conf_mat
        per_cls_rec = per_class_recall_from_conf(conf_mat)

        log(
            f,
            "  Subject {}  loss {:.4f}  acc {:.4f}  samples {}  per-class recall [c0,c1,c2]={}".format(
                subj_str,
                loss,
                acc,
                n_samples,
                [round(r, 4) for r in per_cls_rec],
            ),
        )

        per_subject_results.append((sid, loss, acc, n_samples, per_cls_rec))

    # 整體統計
    losses = np.array([r[1] for r in per_subject_results], dtype=float)
    accs = np.array([r[2] for r in per_subject_results], dtype=float)
    ns = np.array([r[3] for r in per_subject_results], dtype=int)

    mean_loss = losses.mean()
    std_loss = losses.std()
    mean_acc = accs.mean()
    std_acc = accs.std()
    weighted_acc = (accs * ns / ns.sum()).sum()

    per_cls_rec_all = per_class_recall_from_conf(conf_all)

    log(f, "")
    log(f, "========== Summary over all subjects ==========")
    log(f, f"  subjects                    : {[r[0] for r in per_subject_results]}")
    log(f, f"  mean loss over subjects     : {mean_loss:.4f} (std {std_loss:.4f})")
    log(f, f"  mean acc over subjects      : {mean_acc:.4f} (std {std_acc:.4f})")
    log(f, f"  weighted acc by #samples    : {weighted_acc:.4f}")
    log(
        f,
        f"  per-class recall [class0,class1,class2]: {[round(r, 4) for r in per_cls_rec_all]}",
    )

    elapsed = time.time() - start_time
    log(f, "")
    log(f, f"Total evaluation time: {elapsed:.1f} sec")
    log(f, "All done.")
    f.close()


if __name__ == "__main__":
    main()

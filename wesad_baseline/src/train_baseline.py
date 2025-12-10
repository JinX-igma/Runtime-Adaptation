#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fresh baseline training + evaluation for WESAD with fixed CNNBaseline.

特性:
- 使用既定 CNNBaseline 结构 (in_channels=8, num_classes=3)
- 全新训练 (随机初始化, 不加载旧权重)
- Subject 划分清晰:
  TRAIN_SUBJECTS / VAL_SUBJECTS / AE_SUBJECTS / HELDOUT_TEST_SUBJECTS
- 训练完后, 在 seen vs unseen subject 上分别评估并写入 log.

用法 (在 Docker 内):
  python3 train_baseline_fresh.py \
    --data-root /workspace/data/WESAD \
    --batch-size 32 \
    --epochs 40 \
    --device cpu

用法 (在 TX2 上也可以, 如果只是想重训):
  python3 train_baseline_fresh.py \
    --data-root /media/tx2/Base/WESAD \
    --device cuda
"""

import argparse
import os
import time
import platform
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler 


from data.wesad_dataset import WESADDataset
from models.cnn_baseline import CNNBaseline


# ============================================================
# 全局 subject 划分
# ============================================================
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# 1) 训练集 (参与梯度更新)
TRAIN_SUBJECTS = [3, 4, 5, 6, 7, 8, 10, 11]

# 2) 验证集 (early stopping / epoch 选择)
VAL_SUBJECTS = [13, 14]

# 3) 自适应实验专用 (不用于 baseline 训练)
AE_SUBJECTS = [2, 9, 17]

# 4) 最终外部评估专用 (不用于训练 / 自适应)
HELDOUT_TEST_SUBJECTS = [15, 16]

def compute_label_stats(dataset, num_classes=3):
    """
    统计数据集中每个标签的数量和比例
    依赖 dataset.labels 为一维标签数组
    返回 (counts, ratios)
    """
    labels = np.array(dataset.labels)
    counts = np.array([np.sum(labels == c) for c in range(num_classes)], dtype=int)
    total = counts.sum() if counts.sum() > 0 else 1
    ratios = counts.astype(float) / float(total)
    return counts, ratios

# ============================================================
# 工具函数: 随机种子, 日志, exp_id
# ============================================================
def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_experiment_id(prefix: str = "baseline_fresh") -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"exp_{now}_{prefix}"


def create_logger(exp_id: str, log_dir: str = None):
    import os
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    if log_dir is None:
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


# ============================================================
# EarlyStopping: 监控 val_acc
# ============================================================
class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a patience.
    mode = 'max' → 监控 val_acc
    mode = 'min' → 监控 val_loss
    """

    def __init__(self, patience=5, mode="max", min_delta=0.0):
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        返回 True 表示应该 early stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improvement = self.best_score - score
        else:
            improvement = score - self.best_score

        if improvement > self.min_delta:
            # 有提升
            self.best_score = score
            self.counter = 0
        else:
            # 无提升
            self.counter += 1
            print(f"  [EarlyStop] counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


# ============================================================
# 数据构建 & 评估函数
# ============================================================
def build_dataset_for_subjects(
    data_root: str,
    subject_ids,
    window_size: int,
    step_size: int,
    num_classes: int = 3,
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


def evaluate_model(model, loader, device, num_classes=3):
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
            # 更新混淆矩阵
            preds_np = preds.cpu().numpy().reshape(-1)
            y_np = y.cpu().numpy().reshape(-1)
            for t, p in zip(y_np, preds_np):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    conf_mat[t, p] += 1
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc, total_samples, conf_mat


def evaluate_group(model, data_root, subject_ids, window_size, step_size, device, batch_size, num_classes=3):
    """
    对一组 subject 逐个评估, 返回列表:
    [(sid, loss, acc, n_samples, conf_mat), ...]
    """
    results = []
    for sid in subject_ids:
        subj_str = f"S{sid}"
        print(f"  [Eval] Subject {subj_str} ...")
        dataset = build_dataset_for_subjects(
            data_root=data_root,
            subject_ids=[sid],
            window_size=window_size,
            step_size=step_size,
            num_classes=num_classes,
            normalize=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        loss, acc, n_samples, conf_mat = evaluate_model(model, loader, device, num_classes=num_classes)
        print(
            f"    loss {loss:.4f}  acc {acc:.4f}  samples {n_samples}"
        )
        results.append((sid, loss, acc, n_samples, conf_mat))
    return results


def summarize_group(name: str, results, f=None):
    if not results:
        log(f, f"[WARN] No results for group {name}")
        return
    sids = [r[0] for r in results]
    losses = np.array([r[1] for r in results], dtype=float)
    accs = np.array([r[2] for r in results], dtype=float)
    ns = np.array([r[3] for r in results], dtype=int)

    # 聚合所有 subject 的混淆矩阵以统计每個 class 的識別情況
    num_classes = 3
    conf_sum = np.zeros((num_classes, num_classes), dtype=int)
    for r in results:
        conf_sum += r[4]

    log(f, "--------------------------------------------------")
    log(f, f"Summary for group: {name}")
    log(f, f"  subjects: {sids}")
    log(f, f"  mean loss: {losses.mean():.4f} (std {losses.std():.4f})")
    log(f, f"  mean acc : {accs.mean():.4f} (std {accs.std():.4f})")
    acc_weighted = (accs * ns / ns.sum()).sum()
    log(f, f"  weighted acc by #samples: {acc_weighted:.4f}")
    # 每個 class 的召回率 (正確識別該類別的比例)
    per_class_recall = []
    for c in range(num_classes):
        true_c = conf_sum[c, c]
        total_c = conf_sum[c, :].sum()
        recall_c = float(true_c) / float(total_c) if total_c > 0 else 0.0
        per_class_recall.append(round(recall_c, 4))
    log(f, f"  per-class recall [class0,class1,class2]: {per_class_recall}")


# ============================================================
# 主训练流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fresh baseline training (CNNBaseline) with clear seen/unseen subject split"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="WESAD 根目录, 如 /workspace/data/WESAD 或 /media/tx2/Base/WESAD",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="训练与评估的 batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="最大训练轮数 (early stopping 可能提前结束)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="权重衰减",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=700,
        help="滑窗大小 (与之前保持一致)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=350,
        help="滑窗步长 (与之前保持一致)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="训练设备 (cpu/cuda)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early stopping patience (监控 val_acc)",
    )
    args = parser.parse_args()

    set_seed(7)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    data_root = args.data_root
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    window_size = args.window_size
    step_size = args.step_size

    # 生成实验 id & logger
    exp_id = create_experiment_id("baseline_fresh")
    f, log_path = create_logger(exp_id)

    # checkpoint 目录
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{exp_id}_baseline_cnn.pt")

    log(f, "========== Fresh Baseline Training ==========")
    log(f, f"Experiment id : {exp_id}")
    log(f, f"Log path      : {log_path}")
    log(f, f"Checkpoint    : {ckpt_path}")
    log(f, "")
    log(f, "Subject split:")
    log(f, f"  TRAIN_SUBJECTS          : {TRAIN_SUBJECTS}")
    log(f, f"  VAL_SUBJECTS            : {VAL_SUBJECTS}")
    log(f, f"  AE_SUBJECTS (for adapt) : {AE_SUBJECTS}")
    log(f, f"  HELDOUT_TEST_SUBJECTS   : {HELDOUT_TEST_SUBJECTS}")
    log(f, "")

    # 构建原始訓練集與驗證集
    train_dataset = build_dataset_for_subjects(
        data_root=data_root,
        subject_ids=TRAIN_SUBJECTS,
        window_size=window_size,
        step_size=step_size,
        num_classes=3,
        normalize=True,
    )
    val_dataset = build_dataset_for_subjects(
        data_root=data_root,
        subject_ids=VAL_SUBJECTS,
        window_size=window_size,
        step_size=step_size,
        num_classes=3,
        normalize=True,
    )

    # 先統計原始標籤分佈
    train_labels = np.array(train_dataset.labels, dtype=int)
    val_labels = np.array(val_dataset.labels, dtype=int)

    num_classes = 3
    train_counts = np.array(
        [(train_labels == c).sum() for c in range(num_classes)], dtype=int
    )
    train_ratios = train_counts / train_counts.sum()

    val_counts = np.array(
        [(val_labels == c).sum() for c in range(num_classes)], dtype=int
    )
    val_ratios = val_counts / val_counts.sum()

    log(f, "Dataset information")
    log(f, f"  Data root   : {data_root}")
    log(f, f"  Window size : {window_size}")
    log(f, f"  Step size   : {step_size}")
    log(f, f"  Train samples: {len(train_dataset)}")
    log(f, f"  Val samples  : {len(val_dataset)}")
    log(f, "")
    log(f, "Label distribution in TRAIN dataset (before sampling)")
    log(f, f"  counts per class [0,1,2]: {train_counts.tolist()}")
    log(f, f"  ratios per class [0,1,2]: {[round(r, 4) for r in train_ratios]}")
    log(f, "")
    log(f, "Label distribution in VAL dataset")
    log(f, f"  counts per class [0,1,2]: {val_counts.tolist()}")
    log(f, f"  ratios per class [0,1,2]: {[round(r, 4) for r in val_ratios]}")
    log(f, "")

    # 構建類別均衡的 WeightedRandomSampler
    # 每個類別的權重與其出現次數的倒數成正比
    class_weights = 1.0 / train_counts.astype(np.float64)
    sample_weights = class_weights[train_labels]  # 每一個樣本的權重

    log(f, "Class weights used for balanced sampling")
    log(f, f"  class_weights [0,1,2]: {[float(w) for w in class_weights]}")
    log(f, "")

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,   # 使用類別均衡采樣
        shuffle=False,     # 有 sampler 時不能再 shuffle
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 額外做一個 epoch 的采樣分佈檢查 用於 log 中明確證明近似 1 比 1 比 1
    inspect_counts = np.zeros(num_classes, dtype=int)
    for _, y_batch in train_loader:
        y_np = y_batch.numpy()
        for c in range(num_classes):
            inspect_counts[c] += (y_np == c).sum()
    inspect_ratios = inspect_counts / inspect_counts.sum()

    log(f, "Label distribution in one TRAIN epoch (after balanced sampling)")
    log(f, f"  counts per class [0,1,2]: {inspect_counts.tolist()}")
    log(f, f"  ratios per class [0,1,2]: {[round(r, 4) for r in inspect_ratios]}")
    log(f, "")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,      # 使用均衡采样
        shuffle=False,        # sampler 已经负责随机抽样 这里必须为 False
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,        # 验证集保持顺序评估
        num_workers=0,
    )

    # 模型 & 优化器
    in_channels = 8
    num_classes = 3

    model = CNNBaseline(in_channels=in_channels, num_classes=num_classes)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f, "Model information")
    log(f, f"  Python version : {platform.python_version()}")
    log(f, f"  PyTorch version: {torch.__version__}")
    log(f, f"  Device         : {device}")
    log(f, f"  Total params   : {total_params}")
    log(f, "")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    log(f, "Training configuration")
    log(f, f"  Batch size   : {batch_size}")
    log(f, f"  Epochs (max) : {num_epochs}")
    log(f, f"  LR           : {lr}")
    log(f, f"  Weight decay : {weight_decay}")
    log(f, f"  EarlyStop patience: {args.patience}")
    log(f, "")

    config = {
        "exp_id": exp_id,
        "data_root": data_root,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "window_size": window_size,
        "step_size": step_size,
        "device": str(device),
        "patience": args.patience,
        "train_subjects": TRAIN_SUBJECTS,
        "val_subjects": VAL_SUBJECTS,
        "ae_subjects": AE_SUBJECTS,
        "heldout_test_subjects": HELDOUT_TEST_SUBJECTS,
    }

    # Early stopping based on val_acc
    early_stopper = EarlyStopping(patience=args.patience, mode="max", min_delta=0.0)
    best_val_acc = 0.0
    best_epoch = 0

    start_time = time.time()
    log(f, "Start training...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        val_loss, val_acc, val_samples, _ = evaluate_model(model, val_loader, device, num_classes=num_classes)

        log(
            f,
            "Epoch {:03d}  Train loss (train subjects) {:.4f}  Train acc (train subjects) {:.4f}  Val loss (unseen subjects) {:.4f}  Val acc (unseen subjects) {:.4f}".format(
                epoch, train_loss, train_acc, val_loss, val_acc
            ),
        )

        # 保存 best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            ckpt = {
                "exp_id": exp_id,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
            }
            torch.save(ckpt, ckpt_path)
            log(
                f,
                "  New best model saved at epoch {:03d}  val_acc {:.4f}".format(
                    epoch, val_acc
                ),
            )

        # early stopping 检查
        if early_stopper(val_acc):
            log(f, f"  Early stopping triggered at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    log(f, "")
    log(f, "========== Training Finished ==========")
    log(f, f"  Best epoch   : {best_epoch}")
    log(f, f"  Best val_acc : {best_val_acc:.4f}")
    log(f, f"  Total time   : {elapsed:.1f} sec")
    log(f, f"  Final ckpt   : {ckpt_path}")
    log(f, "")

    # ========================================================
    # 使用 best checkpoint 在 seen / unseen 上做完整评估
    # ========================================================
    log(f, "========== Final Evaluation on seen vs unseen sets ==========")
    # 重新加载 best 权重
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    # 如有需要, 可以在這裡讀取 state["config"] 進行一致性檢查
    model.to(device)

    # 1) TRAIN_SUBJECTS
    log(f, "Evaluate TRAIN_SUBJECTS (seen-train)")
    results_train = evaluate_group(
        model=model,
        data_root=data_root,
        subject_ids=TRAIN_SUBJECTS,
        window_size=window_size,
        step_size=step_size,
        device=device,
        batch_size=batch_size,
    )
    summarize_group("Train-Subjects (unseen windows)", results_train, f=f)

    # 2) VAL_SUBJECTS
    log(f, "Evaluate VAL_SUBJECTS (seen-val)")
    results_val = evaluate_group(
        model=model,
        data_root=data_root,
        subject_ids=VAL_SUBJECTS,
        window_size=window_size,
        step_size=step_size,
        device=device,
        batch_size=batch_size,
    )
    summarize_group("Val-Subjects (subject-wise validation)", results_val, f=f)

    # 3) AE_SUBJECTS
    log(f, "Evaluate AE_SUBJECTS (unseen-for-adapt)")
    results_ae = evaluate_group(
        model=model,
        data_root=data_root,
        subject_ids=AE_SUBJECTS,
        window_size=window_size,
        step_size=step_size,
        device=device,
        batch_size=batch_size,
    )
    summarize_group("Adapt-Subjects (unseen for training)", results_ae, f=f)

    # 4) HELDOUT_TEST_SUBJECTS
    log(f, "Evaluate HELDOUT_TEST_SUBJECTS (final-unseen-test)")
    results_test = evaluate_group(
        model=model,
        data_root=data_root,
        subject_ids=HELDOUT_TEST_SUBJECTS,
        window_size=window_size,
        step_size=step_size,
        device=device,
        batch_size=batch_size,
    )
    summarize_group("Held-out Test Subjects (final evaluation)", results_test, f=f)

    log(f, "")
    log(f, "All done.")
    f.close()


if __name__ == "__main__":
    main()

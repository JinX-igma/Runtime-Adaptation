#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
WESAD 基线训练与评估（固定五人最终测试集，训练池内可选小折数交叉验证）。

目标
1 留出 5 个受试者作为最终部署测试集，从头到尾不参与训练和任何选择。
2 剩余受试者作为训练池，用于训练群体 baseline。
3 在训练池内可选 2 折或 3 折交叉验证，仅用于验证训练流程稳定性，不用于调参。
4 每个窗口大小只训练 1 个 baseline checkpoint，然后在最终测试集上评估。

说明
step_size 固定为 window_size 的一半，对应 50% overlap，不允许手动指定。
"""

import argparse
import os
import time
import platform
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader  # sampler referenced via torch.utils.data


from data.wesad_dataset import WESADDataset
from models.cnn_baseline import CNNBaseline



# ============================================================
# 受试者列表
# ============================================================
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# 默认最终测试集，5 人
# 你可以通过命令行参数覆盖
DEFAULT_FINAL_TEST_SUBJECTS = [13, 14, 15, 16, 17]

# ============================================================
# 窗口元信息（用于日志与论文复现）
# ============================================================
FS_HZ = 700  # WESAD 胸部信号采样率

def window_meta(window_size: int, step_size: int, fs_hz: int = FS_HZ):
    """返回 window_sec step_sec overlap_ratio 便于日志记录。"""
    w_sec = float(window_size) / float(fs_hz)
    s_sec = float(step_size) / float(fs_hz)
    overlap = 1.0 - (float(step_size) / float(window_size))
    return w_sec, s_sec, overlap

def choose_val_subjects(train_pool, preferred=(13, 14), k: int = 2):
    """从当前训练池中选择验证受试者。

    优先选择 preferred 中的 ID，如不足则补充最小的 ID。
    保证 LOSO 完全受试者不重叠。
    """
    pool = [int(s) for s in train_pool]
    val = [s for s in preferred if s in pool]
    if len(val) < k:
        remain = [s for s in sorted(pool) if s not in val]
        val += remain[: max(0, k - len(val))]
    return val[:k]

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


def create_experiment_id(prefix: str = "baseline", window_size: int = None, kfold: int = None) -> str:
    """生成实验编号。

    目标是让文件名一眼可读，至少包含时间戳与关键配置。
    window_size 以样本数记录，例如 W700。
    kfold 记录为 K0 K2 K3。
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = ["exp", now]
    if window_size is not None:
        parts.append(f"W{int(window_size)}")
    if kfold is not None:
        parts.append(f"K{int(kfold)}")
    parts.append(prefix)
    return "_".join(parts)


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
    # 归一化说明（复现关键点）
    # 如果 WESADDataset 使用传入的 subject_ids 计算 mean std
    # 那么在评估阶段对每个测试受试者单独 normalize=True 等价于使用目标受试者统计量
    # 这不是标签泄漏，但会改变跨受试者设定
    # 更严格的做法是仅用训练受试者统计量，并在所有划分中复用（需要数据集支持传入预先计算的统计量）
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
    对一组受试者逐个评估，返回列表:
    [(sid, loss, acc, n_samples, conf_mat), ...]
    """
    results = []
    for sid in subject_ids:
        subj_str = f"S{sid}"
        print(f"  [评估] 受试者 {subj_str} ...")
        # 当前评估阶段使用 normalize=True，归一化说明见 build_dataset_for_subjects。
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
        log(f, f"[警告] 分组 {name} 没有结果")
        return
    sids = [r[0] for r in results]
    losses = np.array([r[1] for r in results], dtype=float)
    accs = np.array([r[2] for r in results], dtype=float)
    ns = np.array([r[3] for r in results], dtype=int)

    num_classes = 3
    conf_sum = np.zeros((num_classes, num_classes), dtype=int)
    for r in results:
        conf_sum += r[4]

    log(f, "--------------------------------------------------")
    log(f, f"分组总结: {name}")
    log(f, f"  受试者: {sids}")
    log(f, f"  平均损失: {losses.mean():.4f} (std {losses.std():.4f})")
    log(f, f"  平均准确率: {accs.mean():.4f} (std {accs.std():.4f})")
    acc_weighted = (accs * ns / ns.sum()).sum()
    log(f, f"  按样本加权准确率: {acc_weighted:.4f}")
    # 每类召回率，两种聚合方式
    # 方式一 按样本加权 先累加混淆矩阵再计算
    per_class_recall_weighted = []
    for c in range(num_classes):
        true_c = conf_sum[c, c]
        total_c = conf_sum[c, :].sum()
        recall_c = float(true_c) / float(total_c) if total_c > 0 else 0.0
        per_class_recall_weighted.append(recall_c)

    # 方式二 按受试者统计 先算每个受试者再做汇总
    subj_recalls = []
    for (sid, _loss, _acc, _n, cm) in results:
        r_list = []
        for c in range(num_classes):
            denom = cm[c, :].sum()
            r_list.append(float(cm[c, c]) / float(denom) if denom > 0 else 0.0)
        subj_recalls.append(r_list)

    subj_recalls = np.array(subj_recalls, dtype=float) if len(subj_recalls) > 0 else np.zeros((0, num_classes))

    log(f, f"  每类召回率（样本加权）: {[round(x, 4) for x in per_class_recall_weighted]}")
    if subj_recalls.shape[0] > 0:
        mean_subj = subj_recalls.mean(axis=0)
        med_subj = np.median(subj_recalls, axis=0)
        p10_subj = np.percentile(subj_recalls, 10, axis=0)
        log(f, f"  每类召回率（受试者均值）  : {[round(x, 4) for x in mean_subj]}")
        log(f, f"  每类召回率（受试者中位数）: {[round(x, 4) for x in med_subj]}")
        log(f, f"  每类召回率（受试者10分位）: {[round(x, 4) for x in p10_subj]}")


# ============================================================
# 主训练流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="WESAD 基线训练与评估"
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
        "--final-test-subjects",
        type=str,
        default=",".join([str(x) for x in DEFAULT_FINAL_TEST_SUBJECTS]),
        help="最终测试集受试者 ID，逗号分隔，例如 13,14,15,16,17。该集合从头到尾不参与训练。",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=2,
        choices=[0, 2, 3],
        help="训练池内交叉验证折数。0 表示不做。2 或 3 仅用于稳定性检查，不用于调参。",
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

    def parse_subject_list(s: str):
        items = []
        for part in s.split(","):
            part = part.strip()
            if part:
                items.append(int(part))
        return items

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

    # step_size 和 overlap 由 window_size 决定，保持 50% 重叠
    # 这样可以避免不同实验之间由于推理频率不同造成不可比
    if window_size % 2 != 0:
        raise ValueError(f"window_size 必须为偶数以保证 50% 重叠, 当前 window_size={window_size}")
    step_size = window_size // 2

    w_sec, s_sec, ov = window_meta(window_size, step_size)

    # exp_id 写入窗口大小与 kfold 配置，便于定位日志与 checkpoint
    exp_id = create_experiment_id("baseline", window_size=window_size, kfold=args.kfold)
    f, log_path = create_logger(exp_id)

    # checkpoint 目录
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log(f, "========== 基线训练与评估 ==========")
    log(f, f"实验编号         : {exp_id}")
    log(f, f"日志路径         : {log_path}")
    log(f, "命名规则         : exp_时间_W窗口样本_K折数_baseline")
    log(f, f"设备             : {device}")
    log(f, f"数据根目录       : {data_root}")
    log(f, f"采样率           : {FS_HZ} Hz")
    log(f, f"窗口大小         : {window_size}")
    log(f, f"步长             : {step_size} (由 window_size 计算)")
    log(f, f"窗口秒数         : {w_sec:.3f}")
    log(f, f"步长秒数         : {s_sec:.3f} (由 window_size 计算)")
    log(f, f"重叠率           : {ov:.3f} ({ov*100:.1f}%)")
    log(f, f"最大训练轮数     : {num_epochs}")
    log(f, f"Batch size       : {batch_size}")
    log(f, f"学习率           : {lr}")
    log(f, f"权重衰减         : {weight_decay}")
    log(f, f"EarlyStop patience: {args.patience}")
    log(f, "")

    log(f, "开始执行固定五人最终测试集协议 ...")

    # 最终测试集与训练池
    final_test_subjects = parse_subject_list(args.final_test_subjects)
    final_test_subjects = [s for s in final_test_subjects if s in ALL_SUBJECTS]
    final_test_subjects = sorted(list(dict.fromkeys(final_test_subjects)))

    if len(final_test_subjects) != 5:
        raise ValueError(f"final_test_subjects 必须包含 5 个受试者，当前为 {final_test_subjects}")

    train_pool_subjects = [s for s in ALL_SUBJECTS if s not in final_test_subjects]
    if len(train_pool_subjects) == 0:
        raise ValueError("训练池为空，请检查 final_test_subjects")

    log(f, "训练与评估协议")
    log(f, f"  最终测试集(5人) : {final_test_subjects}")
    log(f, f"  训练池(其余人)  : {train_pool_subjects}")
    log(f, f"  训练池 kfold    : {args.kfold}")
    log(f, "")

    def train_one_model(train_subjects, val_subjects, ckpt_path, use_early_stop: bool):
        """训练一个模型。

        use_early_stop=True 时，每个 epoch 计算 val_acc 并基于 val_acc 保存最佳 checkpoint。
        use_early_stop=False 时，不进行验证选择，训练满 epochs 后保存最终 checkpoint。
        """
        train_dataset = build_dataset_for_subjects(
            data_root=data_root,
            subject_ids=train_subjects,
            window_size=window_size,
            step_size=step_size,
            num_classes=3,
            normalize=True,
        )

        if val_subjects is not None and len(val_subjects) > 0:
            val_dataset = build_dataset_for_subjects(
                data_root=data_root,
                subject_ids=val_subjects,
                window_size=window_size,
                step_size=step_size,
                num_classes=3,
                normalize=True,
            )
        else:
            val_dataset = None

        train_labels = np.array(train_dataset.labels, dtype=int)
        train_counts = np.array([(train_labels == c).sum() for c in range(3)], dtype=int)
        log(f, f"  训练样本数: {len(train_dataset)}  各类计数 {train_counts.tolist()}")

        if val_dataset is not None:
            val_labels = np.array(val_dataset.labels, dtype=int)
            val_counts = np.array([(val_labels == c).sum() for c in range(3)], dtype=int)
            log(f, f"  验证样本数: {len(val_dataset)}  各类计数 {val_counts.tolist()}")

        # 类别均衡采样权重，防止为0
        eps = 1e-12
        safe_counts = np.maximum(train_counts.astype(np.float64), eps)
        class_weights = 1.0 / safe_counts
        sample_weights = class_weights[train_labels]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights.astype(np.float32)),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
        )

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            val_loader = None

        model = CNNBaseline(in_channels=8, num_classes=3)
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_acc = -1.0
        best_epoch = 0
        early_stopper = EarlyStopping(patience=args.patience, mode="max", min_delta=0.0) if use_early_stop else None

        log(f, "  开始训练 ...")
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

            if val_loader is not None:
                val_loss, val_acc, _val_samples, _ = evaluate_model(model, val_loader, device, num_classes=3)
                log(f, f"  Epoch {epoch:03d}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    ckpt = {
                        "exp_id": exp_id,
                        "epoch": epoch,
                        "best_val_acc": float(best_val_acc),
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "window_size": window_size,
                        "step_size": step_size,
                        "fs_hz": FS_HZ,
                        "window_sec": w_sec,
                        "step_sec": s_sec,
                        "overlap": ov,
                        "train_subjects": list(train_subjects),
                        "val_subjects": list(val_subjects) if val_subjects is not None else [],
                        "final_test_subjects": list(final_test_subjects),
                        "seed": 7,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "epochs": num_epochs,
                        "patience": args.patience,
                    }
                    torch.save(ckpt, ckpt_path)
                    log(f, f"  新最佳 checkpoint 已保存 epoch {epoch:03d}")

                if early_stopper is not None and early_stopper(val_acc):
                    log(f, f"  Early stopping 触发于 epoch {epoch}")
                    break
            else:
                log(f, f"  Epoch {epoch:03d}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}")

        # 无验证选择时，保存最终 checkpoint
        if val_loader is None:
            ckpt = {
                "exp_id": exp_id,
                "epoch": num_epochs,
                "best_val_acc": None,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "window_size": window_size,
                "step_size": step_size,
                "fs_hz": FS_HZ,
                "window_sec": w_sec,
                "step_sec": s_sec,
                "overlap": ov,
                "train_subjects": list(train_subjects),
                "val_subjects": [],
                "final_test_subjects": list(final_test_subjects),
                "seed": 7,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "patience": args.patience,
            }
            torch.save(ckpt, ckpt_path)
            log(f, f"  最终 checkpoint 已保存 epoch {num_epochs:03d}")
        else:
            log(f, f"  本次训练最佳 epoch {best_epoch}  best val_acc {best_val_acc:.4f}")

        return ckpt_path

    # 训练池内 kfold 仅用于流程稳定性检查，不用于调参
    if args.kfold > 0:
        k = args.kfold
        log(f, "训练池内交叉验证（稳定性检查）")
        log(f, f"  折数 k = {k}")
        pool = sorted(list(train_pool_subjects))
        folds = [[] for _ in range(k)]
        for idx, sid in enumerate(pool):
            folds[idx % k].append(sid)

        for i in range(k):
            val_subjects = folds[i]
            train_subjects = [s for j in range(k) if j != i for s in folds[j]]
            log(f, "")
            log(f, f"  kfold 第 {i+1} 折")
            log(f, f"    训练受试者: {train_subjects}")
            log(f, f"    验证受试者: {val_subjects}")

            # kfold 临时 checkpoint 文件名包含折序号，方便排查
            ckpt_tmp = os.path.join(ckpt_dir, f"{exp_id}_kfold_fold{i+1}_tmp.pt")
            train_one_model(train_subjects, val_subjects, ckpt_tmp, use_early_stop=True)

        log(f, "")
        log(f, "训练池内交叉验证完成，仅用于稳定性确认")
        log(f, "")

    # 最终训练，每个窗口大小只训练一个 baseline checkpoint
    log(f, "开始最终训练（使用训练池全部受试者）")
    final_ckpt_path = os.path.join(ckpt_dir, f"{exp_id}_final_baseline_cnn.pt")
    train_one_model(train_pool_subjects, val_subjects=None, ckpt_path=final_ckpt_path, use_early_stop=False)

    # 在最终测试集上评估
    log(f, "")
    log(f, "开始在最终测试集上评估")
    state = torch.load(final_ckpt_path, map_location="cpu")
    model = CNNBaseline(in_channels=8, num_classes=3)
    model.load_state_dict(state["model_state"])
    model.to(device)

    results_final_test = evaluate_group(
        model=model,
        data_root=data_root,
        subject_ids=final_test_subjects,
        window_size=window_size,
        step_size=step_size,
        device=device,
        batch_size=batch_size,
        num_classes=3,
    )
    summarize_group("最终测试集评估", results_final_test, f=f)

    log(f, "")
    log(f, "全部完成。")
    f.close()


if __name__ == "__main__":
    main()

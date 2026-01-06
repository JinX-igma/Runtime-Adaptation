import copy
from contextlib import contextmanager
from typing import Iterable, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_bn(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


def _set_dropout_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.eval()


def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def enable_bn_affine_grads_only(model: nn.Module) -> List[nn.Parameter]:
    """
    仅解冻 BN 的 gamma(beta) 即 weight(bias)，返回可训练参数列表
    """
    trainable = []
    for m in model.modules():
        if _is_bn(m):
            if m.weight is not None:
                m.weight.requires_grad = True
                trainable.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad = True
                trainable.append(m.bias)
    return trainable


@contextmanager
def bn_momentum_cma(model: nn.Module):
    """
    AdaBN 校准时，把 BN momentum 临时设为 None
    使 running 统计采用 cumulative moving average，减少小 batch 抖动
    """
    old = {}
    for name, m in model.named_modules():
        if _is_bn(m):
            old[name] = m.momentum
            m.momentum = None
    try:
        yield
    finally:
        for name, m in model.named_modules():
            if _is_bn(m) and name in old:
                m.momentum = old[name]


@torch.no_grad()
def adabn_calibrate(
    model: nn.Module,
    adapt_loader: Iterable,
    device: torch.device,
    num_passes: int = 1,
    max_batches: Optional[int] = None,
) -> None:
    """
    AdaBN：只用目标 subject 的无标签数据，更新 BN 的 running_mean 和 running_var
    不做反向传播，不更新任何可学习参数
    """
    model.to(device)
    model.train()
    _set_dropout_eval(model)

    freeze_all_params(model)

    with bn_momentum_cma(model):
        for _ in range(int(num_passes)):
            for bi, batch in enumerate(adapt_loader):
                if max_batches is not None and bi >= max_batches:
                    break

                # 兼容 batch 为 (x,) 或 x
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device, non_blocking=True)

                _ = model(x)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    返回每个样本的熵，shape: (B,)
    """
    p = F.softmax(logits, dim=1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    return ent


def tent_adapt(
    model: nn.Module,
    adapt_loader: Iterable,
    device: torch.device,
    lr: float = 1e-4,
    steps_per_batch: int = 1,
    ent_threshold: Optional[float] = 0.8,
    max_prob_threshold: Optional[float] = None,
    max_batches: Optional[int] = None,
) -> None:
    """
    TENT：最小化预测熵，仅更新 BN affine 参数 gamma beta
    """
    model.to(device)
    model.train()
    _set_dropout_eval(model)

    freeze_all_params(model)
    trainable = enable_bn_affine_grads_only(model)

    if len(trainable) == 0:
        raise RuntimeError("模型中未找到 BatchNorm 层或 BN 没有 affine 参数")

    opt = torch.optim.Adam(trainable, lr=lr)

    for bi, batch in enumerate(adapt_loader):
        if max_batches is not None and bi >= max_batches:
            break

        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)

        for _ in range(int(steps_per_batch)):
            opt.zero_grad(set_to_none=True)

            logits = model(x)
            ent = entropy_from_logits(logits)

            # 置信门控，优先用熵阈值，其次可用 max prob 阈值
            mask = torch.ones_like(ent, dtype=torch.bool)
            if ent_threshold is not None:
                mask = mask & (ent < float(ent_threshold))

            if max_prob_threshold is not None:
                p = F.softmax(logits, dim=1)
                conf = p.max(dim=1).values
                mask = mask & (conf > float(max_prob_threshold))

            if mask.sum().item() == 0:
                continue

            loss = ent[mask].mean()
            loss.backward()
            opt.step()

import torch
import torch.nn as nn
import torch.nn.functional as F


class T3AWrapper(nn.Module):
    """
    Refined T3A-style online prototype adjustment for (encoder + linear head).

    Changes vs v1:
      1) Logits alignment: use centroid-as-weights + optional head bias + temperature tau
      2) Robust filtering: support entropy quantile filtering (preferred) or fixed threshold
      3) Explicit weight anchor: keep_weight_anchor option, anchor excluded from topM pruning
    """

    def __init__(
        self,
        head_linear: nn.Linear,
        num_classes: int,
        M: int = 30,
        ent_threshold: float = 0.6,      # kept for backward compatibility (optional)
        ent_quantile: float = None,      # new: e.g., 0.2 keeps lowest-entropy 20% in each batch
        warmup_steps: int = 0,
        tau: float = 1.0,               # new: temperature for aligned logits
        use_head_bias: bool = True,      # new: keep original head bias if exists
        keep_weight_anchor: bool = True, # new: anchor is kept permanently, not pruned
        device=None,
    ):
        super().__init__()
        assert isinstance(head_linear, nn.Linear)
        self.head = head_linear
        self.K = int(num_classes)
        self.M = int(M)

        # Filtering
        self.ent_threshold = float(ent_threshold) if ent_threshold is not None else None
        self.ent_quantile = float(ent_quantile) if ent_quantile is not None else None
        if self.ent_quantile is not None:
            if not (0.0 < self.ent_quantile <= 1.0):
                raise ValueError("ent_quantile must be in (0, 1].")

        # Runtime control
        self.warmup_steps = int(warmup_steps)
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError("tau must be > 0.")

        self.use_head_bias = bool(use_head_bias)
        self.keep_weight_anchor = bool(keep_weight_anchor)

        self.device = device or next(head_linear.parameters()).device

        # Internal buffers
        self.step = 0
        self.support_z = [None for _ in range(self.K)]
        self.support_ent = [None for _ in range(self.K)]
        self.anchor_z = None  # (K, D) if keep_weight_anchor
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.step = 0
        self.support_z = [None for _ in range(self.K)]
        self.support_ent = [None for _ in range(self.K)]

        # Weight anchor from classifier weights
        W = self.head.weight.data  # (K, D)
        Wn = F.normalize(W, p=2, dim=1).to(self.device)

        if self.keep_weight_anchor:
            # Store anchors separately (not pruned)
            self.anchor_z = Wn.clone()  # (K, D)
            for k in range(self.K):
                # Start with empty support (no implicit "entropy=0 trick")
                self.support_z[k] = torch.empty((0, Wn.size(1)), device=self.device)
                self.support_ent[k] = torch.empty((0,), device=self.device)
        else:
            # Treat weight anchor as normal support sample
            self.anchor_z = None
            for k in range(self.K):
                self.support_z[k] = Wn[k:k + 1].clone()
                # Give it a reasonable non-extreme entropy so it can be pruned if needed
                self.support_ent[k] = torch.ones(1, device=self.device)

    @torch.no_grad()
    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=1).clamp_min(1e-12)
        return -(p * p.log()).sum(dim=1)

    @torch.no_grad()
    def _select_keep_mask(self, ent: torch.Tensor) -> torch.Tensor:
        """
        ent: (B,)
        returns keep: (B,) bool
        Priority:
          - if ent_quantile is set: keep lowest ent_quantile fraction
          - else if ent_threshold is set: keep ent <= ent_threshold
          - else: keep all
        """
        B = ent.numel()
        if B == 0:
            return torch.zeros_like(ent, dtype=torch.bool)

        if self.ent_quantile is not None:
            # keep the lowest q fraction
            q = self.ent_quantile
            k_keep = max(1, int(round(B * q)))
            # topk smallest entropy
            idx = torch.topk(ent, k=k_keep, largest=False).indices
            keep = torch.zeros(B, device=ent.device, dtype=torch.bool)
            keep[idx] = True
            return keep

        if self.ent_threshold is not None:
            return ent <= self.ent_threshold

        return torch.ones(B, device=ent.device, dtype=torch.bool)

    @torch.no_grad()
    def _keep_topM_low_entropy(self):
        """
        Prune per-class supports to at most M elements (anchors handled separately).
        """
        for k in range(self.K):
            ent = self.support_ent[k]
            z = self.support_z[k]
            if ent.numel() <= self.M:
                continue
            idx = torch.argsort(ent)[: self.M]
            self.support_ent[k] = ent[idx]
            self.support_z[k] = z[idx]

    @torch.no_grad()
    def update(self, z: torch.Tensor):
        """
        z: (B, D) features from encoder (NOT necessarily normalized)
        """
        self.step += 1
        if self.step <= self.warmup_steps:
            return

        # Pseudo-label source is original head on raw z (as in v1)
        logits_base = self.head(z)               # (B, K)
        y_hat = torch.argmax(logits_base, dim=1) # (B,)
        ent = self._entropy(logits_base)         # (B,)

        keep = self._select_keep_mask(ent)       # (B,) bool
        if not bool(keep.any().item()):
            return

        # Normalize z for prototype geometry
        z = F.normalize(z, p=2, dim=1)

        # Vectorized append: group by class
        z_keep = z[keep]          # (B_keep, D)
        y_keep = y_hat[keep]      # (B_keep,)
        ent_keep = ent[keep]      # (B_keep,)

        for k in range(self.K):
            mk = (y_keep == k)
            if not bool(mk.any().item()):
                continue
            self.support_z[k] = torch.cat([self.support_z[k], z_keep[mk]], dim=0)
            self.support_ent[k] = torch.cat([self.support_ent[k], ent_keep[mk]], dim=0)

        self._keep_topM_low_entropy()

    @torch.no_grad()
    def _centroids(self) -> torch.Tensor:
        """
        Build (K, D) centroids from (anchor + supports).
        """
        C_list = []
        for k in range(self.K):
            parts = []
            if self.keep_weight_anchor and self.anchor_z is not None:
                parts.append(self.anchor_z[k:k + 1])  # (1, D)
            if self.support_z[k] is not None and self.support_z[k].numel() > 0:
                parts.append(self.support_z[k])
            if len(parts) == 0:
                # fallback: use current head weight direction
                wk = self.head.weight.data[k:k + 1].to(self.device)
                parts = [F.normalize(wk, p=2, dim=1)]
            ck = torch.cat(parts, dim=0).mean(dim=0, keepdim=True)  # (1, D)
            C_list.append(ck)

        C = torch.cat(C_list, dim=0)  # (K, D)
        C = F.normalize(C, p=2, dim=1)
        return C

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return aligned logits (B, K):
          logits = (normalize(z) @ C.T) / tau + bias(optional)
        """
        C = self._centroids()                    # (K, D)
        z = F.normalize(z, p=2, dim=1)           # (B, D)
        logits = (z @ C.t()) / self.tau          # (B, K)

        if self.use_head_bias and (self.head.bias is not None):
            logits = logits + self.head.bias.view(1, -1)

        return logits

    @torch.no_grad()
    def forward(self, z: torch.Tensor, update: bool = True) -> torch.Tensor:
        if update:
            self.update(z)
        return self.predict(z)

    @torch.no_grad()
    def support_sizes(self):
        # report supports only (exclude anchors) for transparency
        return [int(self.support_z[k].shape[0]) for k in range(self.K)]

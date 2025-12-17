import torch
import torch.nn as nn
import torch.nn.functional as F


class T3AWrapper(nn.Module):
    """
    T3A style online prototype adjustment for any (encoder + linear head) model.
    Does NOT modify model weights. Only keeps per-class buffers in memory.
    """

    def __init__(
        self,
        head_linear: nn.Linear,
        num_classes: int,
        M: int = 30,
        ent_threshold: float = 0.6,
        warmup_steps: int = 0,
        device=None,
    ):
        super().__init__()
        assert isinstance(head_linear, nn.Linear)
        self.head = head_linear
        self.K = int(num_classes)
        self.M = int(M)
        self.ent_threshold = float(ent_threshold) if ent_threshold is not None else None
        self.warmup_steps = int(warmup_steps)
        self.device = device or next(head_linear.parameters()).device

        self.step = 0
        self.support_z = [None for _ in range(self.K)]
        self.support_ent = [None for _ in range(self.K)]
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.step = 0
        self.support_z = [None for _ in range(self.K)]
        self.support_ent = [None for _ in range(self.K)]

        # Initialize each class support with normalized classifier weight vector
        W = self.head.weight.data  # (K, D)
        Wn = F.normalize(W, p=2, dim=1)
        for k in range(self.K):
            self.support_z[k] = Wn[k:k + 1].clone().to(self.device)     # (1, D)
            self.support_ent[k] = torch.zeros(1, device=self.device)     # (1,)

    @torch.no_grad()
    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=1).clamp_min(1e-12)
        return -(p * p.log()).sum(dim=1)

    @torch.no_grad()
    def _keep_topM_low_entropy(self):
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
        z: (B, D) features from encoder
        """
        self.step += 1
        if self.step <= self.warmup_steps:
            return

        logits_base = self.head(z)                 # pseudo label source
        y_hat = torch.argmax(logits_base, dim=1)   # (B,)
        ent = self._entropy(logits_base)           # (B,)

        if self.ent_threshold is not None:
            keep = ent <= self.ent_threshold
        else:
            keep = torch.ones_like(ent, dtype=torch.bool)

        z = F.normalize(z, p=2, dim=1)

        B = z.size(0)
        for i in range(B):
            if not bool(keep[i].item()):
                continue
            k = int(y_hat[i].item())
            self.support_z[k] = torch.cat([self.support_z[k], z[i:i + 1]], dim=0)
            self.support_ent[k] = torch.cat([self.support_ent[k], ent[i:i + 1]], dim=0)

        self._keep_topM_low_entropy()

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        centroid similarity logits: (B, K)
        """
        centroids = []
        for k in range(self.K):
            ck = self.support_z[k].mean(dim=0, keepdim=True)  # (1, D)
            centroids.append(ck)
        C = torch.cat(centroids, dim=0)                       # (K, D)
        C = F.normalize(C, p=2, dim=1)

        z = F.normalize(z, p=2, dim=1)
        return z @ C.t()

    @torch.no_grad()
    def forward(self, z: torch.Tensor, update: bool = True) -> torch.Tensor:
        if update:
            self.update(z)
        return self.predict(z)

    @torch.no_grad()
    def support_sizes(self):
        return [int(self.support_z[k].shape[0]) for k in range(self.K)]

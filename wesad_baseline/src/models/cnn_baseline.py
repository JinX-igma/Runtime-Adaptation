import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    只负责特征抽取的部分
    輸入: (batch, in_channels, seq_len)
    輸出: (batch, 128)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)          # (batch, 128, 1)
        x = x.squeeze(-1)         # (batch, 128)
        return x


class CNNBaseline(nn.Module):
    """
    完整分類模型
    encoder 用於 SSL 與個性化
    head 用於分類
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = CNNEncoder(in_channels)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.encoder(x)     # (batch, 128)
        logits = self.head(feat)   # (batch, num_classes)
        return logits

    def forward_features(self, x):
        """
        只取特征, 方便 SSL 或 linear probe 使用
        """
        return self.encoder(x)
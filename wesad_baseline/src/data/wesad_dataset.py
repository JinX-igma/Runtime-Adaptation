# src/data/wesad_dataset.py
# -*- coding: utf-8 -*-

import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def map_raw_label_to_3cls(raw_label: int) -> int:
    """
    WESAD 原始標籤含義:
      0: baseline
      1: stress
      2: amusement
      3: meditation
      4: 未標註段落
      5: 呼吸任務
      6,7: 其他無效或過渡狀態

    我們映射為三分類:
      neutral:   0  (原始標籤 1 對應 neutral 段前 baseline 等視為無效, 這裡只保留標籤 1 對應 neutral)
      stress:    1  (原始標籤 2)
      amusement: 2  (原始標籤 3)

    注意: 具體映射可以根據你之前 inspect_labels 的對應關係調整
    這裡采用你當前代碼中使用的設置:
      1 -> 0 (neutral)
      2 -> 1 (stress)
      3 -> 2 (amusement)
      其他 -> -1 無效
    """
    if raw_label == 1:
        return 0
    elif raw_label == 2:
        return 1
    elif raw_label == 3:
        return 2
    else:
        return -1


class WESADDataset(Dataset):
    """
    基於 WESAD chest 數據構建的滑窗數據集

    每個樣本形狀:
      x: (C, T)  其中 C=8 通道  T=window_size
      y: int 標籤 {0,1,2} 對應 neutral, stress, amusement
    """

    def __init__(
        self,
        root: str,
        subject_ids: List[int],
        window_size: int,
        step_size: int,
        num_classes: int = 3,
        normalize: bool = True,
    ):
        """
        root: WESAD 根目錄 例如 /workspace/data/WESAD
        subject_ids: 參與這個數據集的 subject 列表 例如 [2,3,4]
        window_size: 窗口長度 采樣點數
        step_size: 滑動步長 采樣點數
        num_classes: 類別數 默認 3 neutral stress amusement
        normalize: 是否做通道級 z score
        """
        self.root = root
        self.subject_ids = subject_ids
        self.window_size = window_size
        self.step_size = step_size
        self.num_classes = num_classes
        self.normalize = normalize

        # 讀取並構建全部 subject 的窗口
        xs_all: List[np.ndarray] = []
        ys_all: List[int] = []

        for sid in subject_ids:
            sx, sy = self._load_subject_windows(sid)
            if sx is None:
                continue
            xs_all.append(sx)
            ys_all.extend(sy)

        if not xs_all:
            raise RuntimeError("No valid samples loaded from WESAD")

        x = np.concatenate(xs_all, axis=0)  # (N, C, T)
        y = np.asarray(ys_all, dtype=np.int64)  # (N,)

        # 通道級 z score 正規化
        if normalize:
            # 計算每個 channel 在整個數據集上的 mean 和 std
            # x: (N, C, T)
            mean = x.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
            std = x.std(axis=(0, 2), keepdims=True)    # (1, C, 1)
            std[std == 0] = 1.0
            x = (x - mean) / std
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None

        self.x = x
        self.y = y
        # 緩存標籤列表給 get_label 使用
        self.labels = self.y.tolist()

    # -------------------------------------------------------
    # 內部輔助方法
    # -------------------------------------------------------

    def _load_subject_pkl(self, subject_id: int):
        """
        從 WESAD 加載單個 subject 的 chest 數據和 label
        期望路徑:
          root / "S{subject_id}" / "S{subject_id}.pkl"
        """
        sub_dir = os.path.join(self.root, f"S{subject_id}")
        pkl_path = os.path.join(sub_dir, f"S{subject_id}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[WESADDataset] Warning: pkl not found for S{subject_id}: {pkl_path}")
            return None, None, None

        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # data["signal"]["chest"] 是 dict
        chest = data["signal"]["chest"]  # keys: ACC, ECG, EMG, EDA, Temp, Resp
        labels = data["label"]           # shape: (N,)

        acc = chest["ACC"]   # (N, 3)
        ecg = chest["ECG"]   # (N, 1)
        emg = chest["EMG"]   # (N, 1)
        eda = chest["EDA"]   # (N, 1)
        temp = chest["Temp"] # (N, 1)
        resp = chest["Resp"] # (N, 1)

        # 將 chest 通道堆疊成 (N, C)
        # 通道順序: ACCx, ACCy, ACCz, ECG, EMG, EDA, Temp, Resp
        x_all = np.concatenate(
            [
                acc[:, 0:1],
                acc[:, 1:2],
                acc[:, 2:3],
                ecg,
                emg,
                eda,
                temp,
                resp,
            ],
            axis=1,
        )  # (N, 8)

        return x_all, labels, data

    def _build_windows_for_subject(self, x_all: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        基於單個 subject 的時序 x_all, labels 構建滑窗
        x_all: (N, C)
        labels: (N,)
        返回:
          xs: (M, C, T)
          ys: list[int] 長度 M
        """
        N, C = x_all.shape
        ws = self.window_size
        st = self.step_size

        xs: List[np.ndarray] = []
        ys: List[int] = []

        for start in range(0, N - ws + 1, st):
            end = start + ws
            seg_labels_raw = labels[start:end]
            # 映射到三類
            seg_labels_mapped = np.array(
                [map_raw_label_to_3cls(int(l)) for l in seg_labels_raw],
                dtype=np.int64,
            )
            # 如果存在無效標籤 則丟棄這個窗口
            if np.any(seg_labels_mapped < 0):
                continue
            # 要求整個窗口內標籤一致
            if not np.all(seg_labels_mapped == seg_labels_mapped[0]):
                continue

            y_mapped = int(seg_labels_mapped[0])
            # 如果你只保留 3 類 可以在這裡 assert
            if not (0 <= y_mapped < self.num_classes):
                continue

            # x_all: (N, C) -> window: (ws, C) -> (C, ws)
            x_win = x_all[start:end, :].T  # (C, T)
            xs.append(x_win)
            ys.append(y_mapped)

        if not xs:
            return None, []

        xs = np.stack(xs, axis=0)  # (M, C, T)
        return xs, ys

    def _load_subject_windows(self, subject_id: int) -> Tuple[np.ndarray, List[int]]:
        """
        整合 單個 subject 的 pkl 加載和滑窗構建
        """
        x_all, labels, _ = self._load_subject_pkl(subject_id)
        if x_all is None:
            return None, []

        xs, ys = self._build_windows_for_subject(x_all, labels)
        if xs is None:
            print(f"[WESADDataset] Warning: no valid windows for subject S{subject_id}")
            return None, []

        print(
            f"[WESADDataset] Subject S{subject_id}: {xs.shape[0]} windows, "
            f"shape per window: {xs.shape[1:]}"
        )
        return xs, ys

    # -------------------------------------------------------
    # Dataset 接口
    # -------------------------------------------------------

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        # x: (C, T) numpy -> tensor
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def get_label(self, idx: int) -> int:
        """
        提供 int 標籤 用於 stream_builder 做分段統計
        """
        return int(self.labels[idx])

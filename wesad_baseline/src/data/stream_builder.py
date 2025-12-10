from typing import Dict, List, Tuple
from collections import namedtuple, defaultdict
from data.wesad_dataset import WESADDataset

Block = namedtuple("Block", ["start", "end", "major_label"])


def build_subject_blocks(
    root: str,
    subject_id: int,
    window_size: int = 700,
    step_size: int = 350,
    num_classes: int = 3,
    normalize: bool = True,
    block_size: int = 50,
    split_ratio: Tuple[float, float, float] = (0.2, 0.4, 0.4),
):
    """
    為單個 subject 建立 block wise label stratified split.
    返回:
      dataset   WESADDataset
      idx_pre   List[int]
      idx_adapt List[int]
      idx_eval  List[int]
    """

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

    # 一 構建 block 列表
    blocks: List[Block] = []
    num_blocks = (n + block_size - 1) // block_size  # ceil

    # 假設 dataset 有一個屬性 labels 或 get_label(i)
    # 如果沒有 可以在 WESADDataset 中加一個 self.labels 緩存
    labels = [dataset.get_label(i) for i in range(n)]

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, n)
        # 統計這個 block 內的標籤分佈
        cnt = [0] * num_classes
        for i in range(start, end):
            y = labels[i]
            if 0 <= y < num_classes:
                cnt[y] += 1
        major_label = int(max(range(num_classes), key=lambda c: cnt[c]))
        blocks.append(Block(start=start, end=end, major_label=major_label))

    # 二 按主標籤聚類 blocks
    label_to_blocks: Dict[int, List[Block]] = defaultdict(list)
    for blk in blocks:
        label_to_blocks[blk.major_label].append(blk)

    # 三 對每個主標籤集合做 0.2/0.4/0.4 切分
    pre_blocks: List[Block] = []
    adapt_blocks: List[Block] = []
    eval_blocks: List[Block] = []

    for c in range(num_classes):
        blks = label_to_blocks[c]
        if not blks:
            continue
        # 這裡保持原順序，也可以隨機打亂再切
        k = len(blks)
        k_pre = int(k * r_pre)
        k_adapt = int(k * r_adapt)
        # 剩下都給 eval
        k_eval = k - k_pre - k_adapt

        pre_blocks.extend(blks[:k_pre])
        adapt_blocks.extend(blks[k_pre:k_pre + k_adapt])
        eval_blocks.extend(blks[k_pre + k_adapt:])

    # 四 展開 block 成窗口 index
    def expand_blocks(blks: List[Block]) -> List[int]:
        idx = []
        for blk in blks:
            idx.extend(list(range(blk.start, blk.end)))
        return idx

    idx_pre = expand_blocks(pre_blocks)
    idx_adapt = expand_blocks(adapt_blocks)
    idx_eval = expand_blocks(eval_blocks)

    return dataset, idx_pre, idx_adapt, idx_eval

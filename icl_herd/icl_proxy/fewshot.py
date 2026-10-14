from typing import List

import torch


def build_fewshot_draw(
    rows: List[List[dict]],
    row_token_lengths: List[int],
    token_budget: int,
    seed: int,
) -> List[dict]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=generator).tolist()

    packed = []
    used = 0
    for i in order:
        length = row_token_lengths[i]
        if used + length > token_budget:
            continue
        used += length
        packed.extend(rows[i])
    return packed

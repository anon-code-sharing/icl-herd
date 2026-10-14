from typing import List, Tuple

import torch
from tqdm import tqdm

from .kernels import vmf_kernel


class ShardedFeatureMatrix:
    def __init__(self, shards: List[torch.Tensor], devices: List[torch.device], offsets: List[int]):
        self.shards = shards
        self.devices = devices
        self.offsets = offsets

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, devices: List[torch.device], dtype: torch.dtype = torch.bfloat16) -> "ShardedFeatureMatrix":
        n = tensor.shape[0]
        n_devices = len(devices)
        base, remainder = divmod(n, n_devices)
        sizes = [base + 1 if i < remainder else base for i in range(n_devices)]

        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)

        shards = []
        for i, dev in enumerate(devices):
            start, end = offsets[i], offsets[i + 1]
            shard = tensor[start:end].to(dev, dtype=dtype)
            norm = shard.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            shard = shard.div_(norm)
            shards.append(shard)

        return cls(shards, devices, offsets)

    @property
    def n_rows(self) -> int:
        return self.offsets[-1]

    def row_count(self, shard_id: int) -> int:
        return self.offsets[shard_id + 1] - self.offsets[shard_id]

    def local_to_global(self, shard_id: int, local_idx: int) -> int:
        return self.offsets[shard_id] + local_idx


def _sharded_argmax(
    score_shards: List[torch.Tensor],
    available_shards: List[torch.Tensor],
    matrix: ShardedFeatureMatrix,
) -> Tuple[int, int, int]:
    candidates = []
    for shard_id, (scores, available) in enumerate(zip(score_shards, available_shards)):
        masked = scores.masked_fill(~available, float("-inf"))
        value, local_idx = torch.max(masked, dim=0)
        local_idx = int(local_idx.item())
        global_idx = matrix.local_to_global(shard_id, local_idx)
        candidates.append((float(value.item()), global_idx, shard_id, local_idx))
    _, global_idx, shard_id, local_idx = min(candidates, key=lambda c: (-c[0], c[1]))
    return global_idx, shard_id, local_idx


def _sharded_target(
    matrix: ShardedFeatureMatrix,
    target: torch.Tensor,
    kappa_inter: float,
    batch_size: int = 32,
) -> List[torch.Tensor]:
    targets = []
    for shard, dev in zip(matrix.shards, matrix.devices):
        local_target = target.to(dev, dtype=shard.dtype)
        n_target = local_target.shape[0]
        total = torch.zeros(shard.shape[0], dtype=torch.float64, device=dev)
        for start in range(0, n_target, batch_size):
            batch = local_target[start:start + batch_size]
            total += vmf_kernel(batch, shard, kappa_inter).sum(dim=0)
        targets.append(total / n_target)
    return targets


def herd_multigpu(
    matrix: ShardedFeatureMatrix,
    target_features: torch.Tensor,
    budget: int,
    kappa_inter: float,
    kappa_intra: float,
    alpha: float = 1.0,
) -> List[int]:
    n_shards = len(matrix.shards)
    targets = _sharded_target(matrix, target_features, kappa_inter)
    available = [torch.ones(matrix.row_count(i), dtype=torch.bool, device=matrix.devices[i]) for i in range(n_shards)]
    penalty = [torch.zeros(matrix.row_count(i), dtype=torch.float64, device=matrix.devices[i]) for i in range(n_shards)]

    selected = []
    for t in tqdm(range(budget), desc="kernel herding (multi-GPU)"):
        scores = targets if t == 0 else [targets[i] - alpha * (penalty[i] / t) for i in range(n_shards)]
        global_idx, shard_id, local_idx = _sharded_argmax(scores, available, matrix)
        selected.append(global_idx)
        available[shard_id][local_idx] = False
        winner = matrix.shards[shard_id][local_idx:local_idx + 1]
        for i in range(n_shards):
            local_winner = winner.to(matrix.devices[i])
            penalty[i] += vmf_kernel(local_winner, matrix.shards[i], kappa_intra)[0]
    return selected


def fused_herd_multigpu(
    matrix_a: ShardedFeatureMatrix,
    matrix_b: ShardedFeatureMatrix,
    target_features_a: torch.Tensor,
    target_features_b: torch.Tensor,
    budget: int,
    kappa_inter_a: float,
    kappa_inter_b: float,
    kappa_intra_a: float,
    kappa_intra_b: float,
    lam: float = 0.5,
    alpha: float = 1.0,
) -> List[int]:
    assert matrix_a.offsets == matrix_b.offsets
    n_shards = len(matrix_a.shards)
    targets_a = _sharded_target(matrix_a, target_features_a, kappa_inter_a)
    targets_b = _sharded_target(matrix_b, target_features_b, kappa_inter_b)
    targets = [lam * targets_a[i] + (1.0 - lam) * targets_b[i] for i in range(n_shards)]
    available = [torch.ones(matrix_a.row_count(i), dtype=torch.bool, device=matrix_a.devices[i]) for i in range(n_shards)]
    penalty = [torch.zeros(matrix_a.row_count(i), dtype=torch.float64, device=matrix_a.devices[i]) for i in range(n_shards)]

    selected = []
    for t in tqdm(range(budget), desc="fused kernel herding (multi-GPU)"):
        scores = targets if t == 0 else [targets[i] - alpha * (penalty[i] / t) for i in range(n_shards)]
        global_idx, shard_id, local_idx = _sharded_argmax(scores, available, matrix_a)
        selected.append(global_idx)
        available[shard_id][local_idx] = False
        winner_a = matrix_a.shards[shard_id][local_idx:local_idx + 1]
        winner_b = matrix_b.shards[shard_id][local_idx:local_idx + 1]
        for i in range(n_shards):
            k_a = vmf_kernel(winner_a.to(matrix_a.devices[i]), matrix_a.shards[i], kappa_intra_a)[0]
            k_b = vmf_kernel(winner_b.to(matrix_b.devices[i]), matrix_b.shards[i], kappa_intra_b)[0]
            penalty[i] += lam * k_a + (1.0 - lam) * k_b
    return selected

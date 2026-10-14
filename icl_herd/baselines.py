from typing import List

import torch
from tqdm import tqdm

from .kernels import normalize


def _cosine_scores(train: torch.Tensor, target: torch.Tensor, method: str, batch_size: int = 512) -> torch.Tensor:
    n_train = train.shape[0]
    n_target = target.shape[0]
    if method == "max":
        scores = torch.full((n_train,), float("-inf"), dtype=torch.float32, device=train.device)
    else:
        scores = torch.zeros(n_train, dtype=torch.float32, device=train.device)

    for start in range(0, n_target, batch_size):
        batch = target[start:start + batch_size]
        sim = torch.mm(batch, train.T).float()
        if method == "max":
            batch_max, _ = sim.max(dim=0)
            scores = torch.maximum(scores, batch_max)
        else:
            scores += sim.sum(dim=0)

    if method == "mean":
        scores /= n_target
    return scores


def cosine_select(
    train_features: torch.Tensor,
    target_features: torch.Tensor,
    budget: int,
    method: str = "max",
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
) -> List[int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train = normalize(train_features).to(device=device, dtype=dtype)
    target = normalize(target_features).to(device=device, dtype=dtype)
    scores = _cosine_scores(train, target, method)
    _, top = torch.topk(scores, k=budget)
    return top.cpu().tolist()


def cosine_round_robin(
    train_features: torch.Tensor,
    target_features: torch.Tensor,
    budget: int,
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
) -> List[int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train = normalize(train_features).to(device=device, dtype=dtype)
    target = normalize(target_features).to(device=device, dtype=dtype)
    n_train = train.shape[0]
    n_target = target.shape[0]
    available = torch.ones(n_train, dtype=torch.bool, device=device)

    selected = []
    for i in tqdm(range(budget), desc="cosine round-robin"):
        query = target[i % n_target]
        sim = torch.mv(train, query).float()
        sim = sim.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(sim).item())
        selected.append(idx)
        available[idx] = False
    return selected


def fused_cosine_select(
    train_features_a: torch.Tensor,
    train_features_b: torch.Tensor,
    target_features_a: torch.Tensor,
    target_features_b: torch.Tensor,
    budget: int,
    method: str = "max",
    weight: float = 0.5,
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
) -> List[int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    a = normalize(train_features_a).to(device=device, dtype=dtype)
    b = normalize(train_features_b).to(device=device, dtype=dtype)
    target_a = normalize(target_features_a).to(device=device, dtype=dtype)
    target_b = normalize(target_features_b).to(device=device, dtype=dtype)
    scores = weight * _cosine_scores(a, target_a, method) + (1.0 - weight) * _cosine_scores(b, target_b, method)
    _, top = torch.topk(scores, k=budget)
    return top.cpu().tolist()


def fused_cosine_round_robin(
    train_features_a: torch.Tensor,
    train_features_b: torch.Tensor,
    target_features_a: torch.Tensor,
    target_features_b: torch.Tensor,
    budget: int,
    weight: float = 0.5,
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
) -> List[int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    a = normalize(train_features_a).to(device=device, dtype=dtype)
    b = normalize(train_features_b).to(device=device, dtype=dtype)
    target_a = normalize(target_features_a).to(device=device, dtype=dtype)
    target_b = normalize(target_features_b).to(device=device, dtype=dtype)
    n_train = a.shape[0]
    n_target = target_a.shape[0]
    available = torch.ones(n_train, dtype=torch.bool, device=device)

    selected = []
    for i in tqdm(range(budget), desc="fused cosine round-robin"):
        idx_t = i % n_target
        sim = weight * torch.mv(a, target_a[idx_t]).float() + (1.0 - weight) * torch.mv(b, target_b[idx_t]).float()
        sim = sim.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(sim).item())
        selected.append(idx)
        available[idx] = False
    return selected

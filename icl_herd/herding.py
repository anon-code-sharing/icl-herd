from typing import List

import torch
from tqdm import tqdm

from .kernels import normalize, vmf_kernel


def _rkhs_target(train: torch.Tensor, target: torch.Tensor, kappa_inter: float, batch_size: int = 32) -> torch.Tensor:
    n_train = train.shape[0]
    n_target = target.shape[0]
    total = torch.zeros(n_train, dtype=torch.float64, device=train.device)
    for start in range(0, n_target, batch_size):
        batch = target[start:start + batch_size]
        total += vmf_kernel(batch, train, kappa_inter).sum(dim=0)
    return total / n_target


def herd(
    train_features: torch.Tensor,
    target_features: torch.Tensor,
    budget: int,
    kappa_inter: float,
    kappa_intra: float,
    alpha: float = 1.0,
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
) -> List[int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train = normalize(train_features).to(device=device, dtype=dtype)
    target = normalize(target_features).to(device=device, dtype=dtype)
    n_train = train.shape[0]

    target_score = _rkhs_target(train, target, kappa_inter)
    running_penalty = torch.zeros(n_train, dtype=torch.float64, device=device)
    available = torch.ones(n_train, dtype=torch.bool, device=device)

    selected = []
    for t in tqdm(range(budget), desc="kernel herding"):
        score = target_score if t == 0 else target_score - alpha * (running_penalty / t)
        score = score.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(score).item())
        selected.append(idx)
        available[idx] = False
        running_penalty += vmf_kernel(train[idx:idx + 1], train, kappa_intra)[0]
    return selected


def fused_herd(
    train_features_a: torch.Tensor,
    train_features_b: torch.Tensor,
    target_features_a: torch.Tensor,
    target_features_b: torch.Tensor,
    budget: int,
    kappa_inter_a: float,
    kappa_inter_b: float,
    kappa_intra_a: float,
    kappa_intra_b: float,
    lam: float = 0.5,
    alpha: float = 1.0,
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
) -> List[int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    a = normalize(train_features_a).to(device=device, dtype=dtype)
    b = normalize(train_features_b).to(device=device, dtype=dtype)
    target_a = normalize(target_features_a).to(device=device, dtype=dtype)
    target_b = normalize(target_features_b).to(device=device, dtype=dtype)
    assert a.shape[0] == b.shape[0]
    n_train = a.shape[0]

    target_score = lam * _rkhs_target(a, target_a, kappa_inter_a) + (1.0 - lam) * _rkhs_target(b, target_b, kappa_inter_b)
    running_penalty = torch.zeros(n_train, dtype=torch.float64, device=device)
    available = torch.ones(n_train, dtype=torch.bool, device=device)

    selected = []
    for t in tqdm(range(budget), desc="fused kernel herding"):
        score = target_score if t == 0 else target_score - alpha * (running_penalty / t)
        score = score.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(score).item())
        selected.append(idx)
        available[idx] = False
        k_a = vmf_kernel(a[idx:idx + 1], a, kappa_intra_a)[0]
        k_b = vmf_kernel(b[idx:idx + 1], b, kappa_intra_b)[0]
        running_penalty += lam * k_a + (1.0 - lam) * k_b
    return selected

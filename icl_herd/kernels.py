import torch


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def vmf_kernel(x: torch.Tensor, y: torch.Tensor, kappa: float) -> torch.Tensor:
    cosine = torch.mm(x, y.T).double()
    return torch.exp(kappa * (cosine - 1.0))


def median_cosine_distance(
    a: torch.Tensor,
    b: torch.Tensor = None,
    n_pairs: int = 10000,
    generator: torch.Generator = None,
) -> float:
    a = normalize(a)
    self_pairs = b is None
    b = a if self_pairs else normalize(b)
    n_a, n_b = a.shape[0], b.shape[0]

    idx_a = torch.randint(0, n_a, (n_pairs,), generator=generator)
    idx_b = torch.randint(0, n_b, (n_pairs,), generator=generator)
    if self_pairs:
        same = idx_a == idx_b
        while same.any():
            idx_b[same] = torch.randint(0, n_b, (int(same.sum()),), generator=generator)
            same = idx_a == idx_b

    cosine = (a[idx_a] * b[idx_b]).sum(dim=1)
    return (1.0 - cosine).median().item()


def kappa_from_median_heuristic(base_distance: float, scalar: float = 1.0) -> float:
    return scalar / (2.0 * base_distance)


def auto_kappa_inter(target_features: torch.Tensor, scalar: float = 1.0) -> float:
    base = median_cosine_distance(target_features)
    return kappa_from_median_heuristic(base, scalar)


def auto_kappa_intra(train_features: torch.Tensor, scalar: float = 1.0, sample_size: int = 10000) -> float:
    n = train_features.shape[0]
    if n > sample_size:
        idx = torch.randperm(n)[:sample_size]
        train_features = train_features[idx]
    base = median_cosine_distance(train_features)
    return kappa_from_median_heuristic(base, scalar)

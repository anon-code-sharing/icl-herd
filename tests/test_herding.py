import torch

from icl_herd.herding import fused_herd, herd
from icl_herd.kernels import auto_kappa_inter, auto_kappa_intra


def _make_clustered_pool(n_per_cluster=50, dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    center_a = torch.randn(dim, generator=generator)
    center_b = -center_a
    cluster_a = center_a + 0.1 * torch.randn(n_per_cluster, dim, generator=generator)
    cluster_b = center_b + 0.1 * torch.randn(n_per_cluster, dim, generator=generator)
    pool = torch.cat([cluster_a, cluster_b], dim=0)
    return pool, n_per_cluster


def test_herd_prefers_matching_cluster():
    pool, n_a = _make_clustered_pool()
    target = pool[:5]
    kappa_inter = auto_kappa_inter(target)
    kappa_intra = auto_kappa_intra(pool)
    selected = herd(pool, target, budget=10, kappa_inter=kappa_inter, kappa_intra=kappa_intra, device="cpu", dtype=torch.float32)
    assert sum(1 for i in selected if i < n_a) >= 8


def test_herd_selects_without_duplicates():
    pool, _ = _make_clustered_pool()
    target = pool[:5]
    kappa_inter = auto_kappa_inter(target)
    kappa_intra = auto_kappa_intra(pool)
    selected = herd(pool, target, budget=20, kappa_inter=kappa_inter, kappa_intra=kappa_intra, device="cpu", dtype=torch.float32)
    assert len(set(selected)) == len(selected)
    assert len(selected) == 20


def test_fused_herd_lambda_one_matches_pure_space_a():
    pool_a, n_a = _make_clustered_pool(seed=0)
    pool_b, _ = _make_clustered_pool(seed=1)
    target_a = pool_a[:5]
    target_b = pool_b[:5]
    kappa_inter_a = auto_kappa_inter(target_a)
    kappa_intra_a = auto_kappa_intra(pool_a)
    kappa_inter_b = auto_kappa_inter(target_b)
    kappa_intra_b = auto_kappa_intra(pool_b)

    pure = herd(pool_a, target_a, budget=10, kappa_inter=kappa_inter_a, kappa_intra=kappa_intra_a, device="cpu", dtype=torch.float32)
    fused = fused_herd(
        pool_a, pool_b, target_a, target_b, budget=10,
        kappa_inter_a=kappa_inter_a, kappa_inter_b=kappa_inter_b,
        kappa_intra_a=kappa_intra_a, kappa_intra_b=kappa_intra_b,
        lam=1.0, device="cpu", dtype=torch.float32,
    )
    assert pure == fused


def test_fused_herd_lambda_zero_matches_pure_space_b():
    pool_a, _ = _make_clustered_pool(seed=0)
    pool_b, n_b = _make_clustered_pool(seed=1)
    target_a = pool_a[:5]
    target_b = pool_b[:5]
    kappa_inter_a = auto_kappa_inter(target_a)
    kappa_intra_a = auto_kappa_intra(pool_a)
    kappa_inter_b = auto_kappa_inter(target_b)
    kappa_intra_b = auto_kappa_intra(pool_b)

    pure = herd(pool_b, target_b, budget=10, kappa_inter=kappa_inter_b, kappa_intra=kappa_intra_b, device="cpu", dtype=torch.float32)
    fused = fused_herd(
        pool_a, pool_b, target_a, target_b, budget=10,
        kappa_inter_a=kappa_inter_a, kappa_inter_b=kappa_inter_b,
        kappa_intra_a=kappa_intra_a, kappa_intra_b=kappa_intra_b,
        lam=0.0, device="cpu", dtype=torch.float32,
    )
    assert pure == fused

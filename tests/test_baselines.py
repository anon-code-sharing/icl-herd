import torch

from icl_herd.baselines import cosine_round_robin, cosine_select, fused_cosine_select


def _make_clustered_pool(n_per_cluster=30, dim=12, seed=1):
    generator = torch.Generator().manual_seed(seed)
    center_a = torch.randn(dim, generator=generator)
    center_b = -center_a
    cluster_a = center_a + 0.05 * torch.randn(n_per_cluster, dim, generator=generator)
    cluster_b = center_b + 0.05 * torch.randn(n_per_cluster, dim, generator=generator)
    pool = torch.cat([cluster_a, cluster_b], dim=0)
    return pool, n_per_cluster


def test_cosine_max_prefers_matching_cluster():
    pool, n_a = _make_clustered_pool()
    target = pool[:3]
    selected = cosine_select(pool, target, budget=10, method="max", device="cpu", dtype=torch.float32)
    assert all(i < n_a for i in selected)


def test_cosine_mean_prefers_matching_cluster():
    pool, n_a = _make_clustered_pool()
    target = pool[:3]
    selected = cosine_select(pool, target, budget=10, method="mean", device="cpu", dtype=torch.float32)
    assert all(i < n_a for i in selected)


def test_cosine_round_robin_selects_without_duplicates():
    pool, _ = _make_clustered_pool()
    target = pool[:3]
    selected = cosine_round_robin(pool, target, budget=15, device="cpu", dtype=torch.float32)
    assert len(set(selected)) == len(selected)
    assert len(selected) == 15


def test_fused_cosine_select_weight_one_matches_pure_space_a():
    pool_a, n_a = _make_clustered_pool(seed=1)
    pool_b, _ = _make_clustered_pool(seed=2)
    target_a = pool_a[:3]
    target_b = pool_b[:3]
    pure = cosine_select(pool_a, target_a, budget=10, method="max", device="cpu", dtype=torch.float32)
    fused = fused_cosine_select(
        pool_a, pool_b, target_a, target_b, budget=10, method="max", weight=1.0,
        device="cpu", dtype=torch.float32,
    )
    assert pure == fused

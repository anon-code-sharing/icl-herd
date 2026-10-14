import torch

from icl_herd.kernels import (
    auto_kappa_inter,
    auto_kappa_intra,
    kappa_from_median_heuristic,
    median_cosine_distance,
    normalize,
    vmf_kernel,
)


def test_normalize_unit_norm():
    x = torch.randn(10, 4)
    n = normalize(x)
    assert torch.allclose(n.norm(dim=1), torch.ones(10), atol=1e-5)


def test_vmf_kernel_self_similarity_is_one():
    x = normalize(torch.randn(5, 8))
    k = vmf_kernel(x, x, kappa=3.0)
    assert torch.allclose(k.diag(), torch.ones(5, dtype=k.dtype), atol=1e-5)


def test_vmf_kernel_decreases_with_distance():
    x = torch.tensor([[1.0, 0.0]])
    y_close = normalize(torch.tensor([[0.99, 0.1411]]))
    y_far = normalize(torch.tensor([[0.0, 1.0]]))
    k_close = vmf_kernel(x, y_close, kappa=5.0).item()
    k_far = vmf_kernel(x, y_far, kappa=5.0).item()
    assert k_close > k_far


def test_median_cosine_distance_self_population_is_positive():
    x = normalize(torch.randn(50, 16))
    d = median_cosine_distance(x, n_pairs=2000)
    assert d > 0.0


def test_median_cosine_distance_cross_population():
    a = normalize(torch.randn(30, 8))
    b = normalize(torch.randn(5, 8))
    d = median_cosine_distance(a, b, n_pairs=2000)
    assert 0.0 <= d <= 2.0


def test_kappa_from_median_heuristic():
    assert kappa_from_median_heuristic(base_distance=0.5, scalar=1.0) == 1.0
    assert kappa_from_median_heuristic(base_distance=0.5, scalar=2.0) == 2.0


def test_auto_kappa_inter_and_intra_are_positive():
    target = normalize(torch.randn(20, 8))
    train = normalize(torch.randn(200, 8))
    assert auto_kappa_inter(target) > 0
    assert auto_kappa_intra(train) > 0

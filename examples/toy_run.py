import torch

from icl_herd.baselines import cosine_select
from icl_herd.herding import fused_herd, herd
from icl_herd.kernels import auto_kappa_inter, auto_kappa_intra

GENERATOR = torch.Generator().manual_seed(0)
DIM = 32
N_PER_CLUSTER = 500
BUDGET = 20


def make_pool():
    center_a = torch.randn(DIM, generator=GENERATOR)
    center_b = -center_a
    off_topic = torch.randn(DIM, generator=GENERATOR)
    cluster_a = center_a + 0.2 * torch.randn(N_PER_CLUSTER, DIM, generator=GENERATOR)
    cluster_b = center_b + 0.2 * torch.randn(N_PER_CLUSTER, DIM, generator=GENERATOR)
    cluster_c = off_topic + 0.2 * torch.randn(N_PER_CLUSTER, DIM, generator=GENERATOR)
    return torch.cat([cluster_a, cluster_b, cluster_c], dim=0), center_a


def main():
    embed_pool, embed_center = make_pool()
    grad_pool, grad_center = make_pool()
    target_embed = embed_center.unsqueeze(0).repeat(10, 1) + 0.05 * torch.randn(10, DIM, generator=GENERATOR)
    target_grad = grad_center.unsqueeze(0).repeat(10, 1) + 0.05 * torch.randn(10, DIM, generator=GENERATOR)

    kappa_inter = auto_kappa_inter(target_embed)
    kappa_intra = auto_kappa_intra(embed_pool)
    herded = herd(embed_pool, target_embed, BUDGET, kappa_inter, kappa_intra, device="cpu")
    print("embed-only herding selected (first 10 of 3000-row pool):", sorted(herded)[:10])

    baseline = cosine_select(embed_pool, target_embed, BUDGET, method="max", device="cpu")
    print("cos-max baseline selected (first 10):", sorted(baseline)[:10])

    kappa_inter_a = auto_kappa_inter(target_embed)
    kappa_intra_a = auto_kappa_intra(embed_pool)
    kappa_inter_b = auto_kappa_inter(target_grad)
    kappa_intra_b = auto_kappa_intra(grad_pool)
    fused = fused_herd(
        embed_pool, grad_pool, target_embed, target_grad, BUDGET,
        kappa_inter_a, kappa_inter_b, kappa_intra_a, kappa_intra_b,
        lam=0.5, device="cpu",
    )
    print("embed+grad fused herding selected (first 10):", sorted(fused)[:10])


if __name__ == "__main__":
    main()

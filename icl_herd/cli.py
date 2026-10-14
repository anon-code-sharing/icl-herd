import argparse
import json
import sys
from pathlib import Path

import torch

from .baselines import cosine_round_robin, cosine_select
from .features import load_feature_matrix
from .herding import fused_herd, herd
from .kernels import auto_kappa_inter, auto_kappa_intra


def _write_selected(indices, train_data, output):
    needed = set(indices)
    rows = {}
    with open(train_data) as f:
        for i, line in enumerate(f):
            if i in needed:
                rows[i] = line.rstrip("\n")
    with open(output, "w") as f:
        for idx in indices:
            f.write(rows[idx] + "\n")


def _emit(selected, args):
    if args.train_data:
        _write_selected(selected, args.train_data, args.output)
    else:
        Path(args.output).write_text("\n".join(map(str, selected)))


def cmd_herd(args):
    train = load_feature_matrix(args.train_features)
    target = load_feature_matrix(args.target_features)
    kappa_inter = args.kappa_inter if args.kappa_inter is not None else auto_kappa_inter(target, args.kappa_inter_scalar)
    kappa_intra = args.kappa_intra if args.kappa_intra is not None else auto_kappa_intra(train, args.kappa_intra_scalar)
    print(f"kappa_inter={kappa_inter:.6f} kappa_intra={kappa_intra:.6f}", file=sys.stderr)
    selected = herd(train, target, args.budget, kappa_inter, kappa_intra, alpha=args.alpha)
    _emit(selected, args)


def cmd_fused_herd(args):
    train_a = load_feature_matrix(args.train_features_a)
    train_b = load_feature_matrix(args.train_features_b)
    target_a = load_feature_matrix(args.target_features_a)
    target_b = load_feature_matrix(args.target_features_b)
    kappa_inter_a = args.kappa_inter_a if args.kappa_inter_a is not None else auto_kappa_inter(target_a, args.kappa_inter_scalar)
    kappa_inter_b = args.kappa_inter_b if args.kappa_inter_b is not None else auto_kappa_inter(target_b, args.kappa_inter_scalar)
    kappa_intra_a = args.kappa_intra_a if args.kappa_intra_a is not None else auto_kappa_intra(train_a, args.kappa_intra_scalar)
    kappa_intra_b = args.kappa_intra_b if args.kappa_intra_b is not None else auto_kappa_intra(train_b, args.kappa_intra_scalar)
    print(
        f"kappa_inter_a={kappa_inter_a:.6f} kappa_inter_b={kappa_inter_b:.6f} "
        f"kappa_intra_a={kappa_intra_a:.6f} kappa_intra_b={kappa_intra_b:.6f}",
        file=sys.stderr,
    )
    selected = fused_herd(
        train_a, train_b, target_a, target_b, args.budget,
        kappa_inter_a, kappa_inter_b, kappa_intra_a, kappa_intra_b,
        lam=args.lam, alpha=args.alpha,
    )
    _emit(selected, args)


def cmd_baseline(args):
    train = load_feature_matrix(args.train_features)
    target = load_feature_matrix(args.target_features)
    if args.mode == "round-robin":
        selected = cosine_round_robin(train, target, args.budget)
    else:
        selected = cosine_select(train, target, args.budget, method=args.mode)
    _emit(selected, args)


def cmd_score(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .icl_proxy.score import score_config

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    herded_rows = [json.loads(line)["messages"] for line in open(args.herded) if line.strip()]
    eval_docs = [json.loads(line) for line in open(args.eval_file) if line.strip()]

    result = score_config(
        model, tokenizer, herded_rows, eval_docs,
        m_draws=args.m_draws, draw_seed_base=args.draw_seed_base,
        max_context=args.max_context, device=args.device,
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="icl-herd")
    sub = parser.add_subparsers(dest="command", required=True)

    p_herd = sub.add_parser("herd", help="Kernel herding in a single feature space.")
    p_herd.add_argument("--train-features", required=True)
    p_herd.add_argument("--target-features", required=True)
    p_herd.add_argument("--train-data", default=None, help="Training jsonl to materialize selected rows from. Without this, --output holds row indices.")
    p_herd.add_argument("--budget", type=int, required=True)
    p_herd.add_argument("--kappa-inter", type=float, default=None)
    p_herd.add_argument("--kappa-intra", type=float, default=None)
    p_herd.add_argument("--kappa-inter-scalar", type=float, default=1.0)
    p_herd.add_argument("--kappa-intra-scalar", type=float, default=1.0)
    p_herd.add_argument("--alpha", type=float, default=1.0)
    p_herd.add_argument("--output", required=True)
    p_herd.set_defaults(func=cmd_herd)

    p_fused = sub.add_parser("fused-herd", help="Kernel herding over an additive fusion of two feature spaces.")
    p_fused.add_argument("--train-features-a", required=True)
    p_fused.add_argument("--train-features-b", required=True)
    p_fused.add_argument("--target-features-a", required=True)
    p_fused.add_argument("--target-features-b", required=True)
    p_fused.add_argument("--train-data", default=None)
    p_fused.add_argument("--budget", type=int, required=True)
    p_fused.add_argument("--kappa-inter-a", type=float, default=None)
    p_fused.add_argument("--kappa-inter-b", type=float, default=None)
    p_fused.add_argument("--kappa-intra-a", type=float, default=None)
    p_fused.add_argument("--kappa-intra-b", type=float, default=None)
    p_fused.add_argument("--kappa-inter-scalar", type=float, default=1.0)
    p_fused.add_argument("--kappa-intra-scalar", type=float, default=1.0)
    p_fused.add_argument("--lam", type=float, default=0.5, help="1.0 = pure space A, 0.0 = pure space B.")
    p_fused.add_argument("--alpha", type=float, default=1.0)
    p_fused.add_argument("--output", required=True)
    p_fused.set_defaults(func=cmd_fused_herd)

    p_baseline = sub.add_parser("baseline", help="Plain cosine-similarity selection baselines.")
    p_baseline.add_argument("--mode", choices=["max", "mean", "round-robin"], required=True)
    p_baseline.add_argument("--train-features", required=True)
    p_baseline.add_argument("--target-features", required=True)
    p_baseline.add_argument("--train-data", default=None)
    p_baseline.add_argument("--budget", type=int, required=True)
    p_baseline.add_argument("--output", required=True)
    p_baseline.set_defaults(func=cmd_baseline)

    p_score = sub.add_parser("score", help="ICL-proxy score of a herded set (requires the icl-proxy extra).")
    p_score.add_argument("--herded", required=True)
    p_score.add_argument("--eval-file", required=True)
    p_score.add_argument("--model", required=True)
    p_score.add_argument("--m-draws", type=int, default=10)
    p_score.add_argument("--draw-seed-base", type=int, default=3000)
    p_score.add_argument("--max-context", type=int, default=32768)
    p_score.add_argument("--device", default="cuda")
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# icl-herd

Kernel herding for training-data selection, plus an ICL proxy for scoring selection
hyperparameters without finetuning. Method details are in the paper, not here.

## Install

```bash
pip install -e .
pip install -e ".[icl-proxy]"   # transformers, only needed for ICL-proxy scoring
```

## Input features

- Precomputed per-example feature vectors — not computed by this repo.
- One `(n_examples, dim)` tensor per pool (`torch.save`, a tensor or a dict with an
  `"embeddings"` key): training pool + target set, per feature space.
- Text-embedding features: [RDS+](https://github.com/hamishivi/automated-instruction-selection).
- Gradient features: [LESS](https://github.com/princeton-nlp/LESS).

## Usage

Single-space herding:

```bash
icl-herd herd --train-features train.pt --target-features target.pt --train-data train.jsonl --budget 10000 --output selected.jsonl
```

Embed+grad fusion (`--lam 1.0` = pure space A, `--lam 0.0` = pure space B):

```bash
icl-herd fused-herd --train-features-a embed.pt --train-features-b grad.pt --target-features-a target_embed.pt --target-features-b target_grad.pt --budget 10000 --output selected.jsonl
```

Cosine baselines (`--mode max|mean|round-robin`):

```bash
icl-herd baseline --mode max --train-features train.pt --target-features target.pt --budget 10000 --output selected.jsonl
```

ICL-proxy scoring:
- Input = chat-format jsonl, `"messages"` list of `{"role": ..., "content": ...}` turns.
- Eval-doc query/target = its last user/assistant turn pair.

```bash
icl-herd score --herded selected.jsonl --eval-file held_out_eval.jsonl --model /path/to/a/hf/causal/lm
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

- ICL-proxy's model-dependent test needs `ICL_HERD_TEST_MODEL` set to a local HF causal-LM path.

## License

MIT, see `LICENSE`.

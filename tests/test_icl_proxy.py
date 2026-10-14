import math
import os

import pytest
import torch

from icl_herd.icl_proxy.fewshot import build_fewshot_draw

MODEL_PATH = os.environ.get("ICL_HERD_TEST_MODEL")


def _toy_rows(n=20):
    rows = []
    row_lens = []
    for i in range(n):
        rows.append([
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        ])
        row_lens.append(10)
    return rows, row_lens


def test_build_fewshot_draw_respects_token_budget():
    rows, row_lens = _toy_rows(n=20)
    draw = build_fewshot_draw(rows, row_lens, token_budget=55, seed=0)
    n_packed_rows = len(draw) // 2
    assert n_packed_rows * 10 <= 55
    assert n_packed_rows >= 1


def test_build_fewshot_draw_is_deterministic_given_seed():
    rows, row_lens = _toy_rows(n=20)
    draw_a = build_fewshot_draw(rows, row_lens, token_budget=55, seed=42)
    draw_b = build_fewshot_draw(rows, row_lens, token_budget=55, seed=42)
    assert draw_a == draw_b


def test_build_fewshot_draw_varies_across_seeds():
    rows, row_lens = _toy_rows(n=20)
    draw_a = build_fewshot_draw(rows, row_lens, token_budget=55, seed=0)
    draw_b = build_fewshot_draw(rows, row_lens, token_budget=55, seed=1)
    assert draw_a != draw_b


@pytest.mark.skipif(MODEL_PATH is None, reason="set ICL_HERD_TEST_MODEL to a local HF causal-LM path to run this")
def test_score_config_smoke():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from icl_herd.icl_proxy.score import score_config

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to(device).eval()

    herded_rows, _ = _toy_rows(n=8)
    eval_docs = [
        {"messages": [{"role": "user", "content": "what is 1 + 1?"}, {"role": "assistant", "content": "2"}]},
        {"messages": [{"role": "user", "content": "what color is the sky?"}, {"role": "assistant", "content": "blue"}]},
    ]

    result = score_config(
        model, tokenizer, herded_rows, eval_docs,
        m_draws=1, draw_seed_base=0, max_context=4096, few_shot_margin=256, device=device,
    )
    assert "worst_of_m" in result
    assert result["worst_of_m"] == result["draw_scores"][0]
    assert not math.isnan(result["worst_of_m"])

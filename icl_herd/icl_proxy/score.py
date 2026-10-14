from typing import List, Optional

import torch

from .fewshot import build_fewshot_draw


def eval_doc_query_target(doc: dict):
    messages = doc["messages"]
    assert messages[-1]["role"] == "assistant"
    assert messages[-2]["role"] == "user"
    return messages[-2]["content"], messages[-1]["content"].strip()


def encode(tokenizer, messages: List[dict], add_generation_prompt: bool) -> List[int]:
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)


def score_doc(
    model,
    tokenizer,
    fewshot_messages: List[dict],
    doc: dict,
    max_context: int,
    device: str = "cuda",
) -> Optional[float]:
    user_content, target = eval_doc_query_target(doc)
    query_messages = fewshot_messages + [{"role": "user", "content": user_content}]
    prefix_ids = encode(tokenizer, query_messages, add_generation_prompt=True)
    full_ids = encode(tokenizer, query_messages + [{"role": "assistant", "content": target}], add_generation_prompt=False)
    target_ids = full_ids[len(prefix_ids):]
    if not target_ids or len(full_ids) > max_context:
        return None

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]

    start = len(full_ids) - len(target_ids)
    log_probs = torch.log_softmax(logits[start - 1:start - 1 + len(target_ids)].float(), dim=-1)
    target_tensor = torch.tensor(target_ids, device=device)
    token_logps = log_probs[torch.arange(len(target_ids)), target_tensor]
    return token_logps.mean().item()


def score_draw(
    model,
    tokenizer,
    fewshot_messages: List[dict],
    eval_docs: List[dict],
    max_context: int,
    device: str = "cuda",
) -> float:
    logps = []
    for doc in eval_docs:
        logp = score_doc(model, tokenizer, fewshot_messages, doc, max_context, device=device)
        if logp is not None:
            logps.append(logp)
    if not logps:
        return float("nan")
    return float(sum(logps) / len(logps))


def score_config(
    model,
    tokenizer,
    herded_rows: List[List[dict]],
    eval_docs: List[dict],
    m_draws: int = 10,
    draw_seed_base: int = 3000,
    max_context: int = 32768,
    few_shot_margin: int = 4096,
    device: str = "cuda",
) -> dict:
    row_token_lengths = [len(encode(tokenizer, row, False)) for row in herded_rows]
    longest_doc = 0
    for doc in eval_docs:
        user_content, target = eval_doc_query_target(doc)
        length = len(encode(
            tokenizer,
            [{"role": "user", "content": user_content}, {"role": "assistant", "content": target}],
            False,
        ))
        longest_doc = max(longest_doc, length)
    token_budget = max_context - longest_doc - few_shot_margin

    draw_scores = []
    for d in range(m_draws):
        fewshot_messages = build_fewshot_draw(herded_rows, row_token_lengths, token_budget, seed=draw_seed_base + d)
        draw_scores.append(score_draw(model, tokenizer, fewshot_messages, eval_docs, max_context, device=device))

    return {
        "draw_scores": draw_scores,
        "worst_of_m": min(draw_scores),
        "mean_of_m": sum(draw_scores) / len(draw_scores),
    }

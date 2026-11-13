"""Shared constants and helpers for Task 1 (resumable fine-tuning pipeline)."""
import json
import os
import re
import shutil

import torch

MODEL = "Qwen/Qwen3-0.6B"
SEED = 6304
N_TRAIN = 40_000
MAX_LEN = 2048
MICRO_BS = 2
ACCUM = 16
BATCH = MICRO_BS * ACCUM            # 32
TOTAL_UPDATES = N_TRAIN // BATCH    # 1250
WARMUP = 38
MILESTONES = [312, 625, 937, 1250]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# method -> (peak lr, lambda, adapter rank, alpha, oplora)
METHODS = {
    "full":    dict(lr=2e-5, lam=0.0,  r=None, alpha=None, oplora=False),
    "l2sp":    dict(lr=2e-5, lam=1e-3, r=None, alpha=None, oplora=False),
    "lora4":   dict(lr=2e-4, lam=0.0,  r=4,    alpha=8,    oplora=False),
    "lora16":  dict(lr=2e-4, lam=0.0,  r=16,   alpha=32,   oplora=False),
    "oplora":  dict(lr=2e-4, lam=0.0,  r=16,   alpha=32,   oplora=True),
}
PROT_K = 128


def atomic_save(obj, path):
    """Write to a temp file then rename, so a disconnect mid-write never corrupts a checkpoint."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def dir_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def norm_text(t):
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def build_dataset(tok, cache_path):
    """First 40k eligible MetaMathQA examples in source order (cached so every run/resume is identical)."""
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    from datasets import load_dataset
    gsm_test = {norm_text(q) for q in load_dataset("openai/gsm8k", "main", split="test")["question"]}
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    out, dropped = [], dict(empty=0, leak=0, long=0)
    for ex in ds:
        q, r = (ex["query"] or "").strip(), (ex["response"] or "").strip()
        if not q or not r:
            dropped["empty"] += 1
            continue
        if norm_text(q) in gsm_test or norm_text(ex.get("original_question") or "") in gsm_test:
            dropped["leak"] += 1
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(r + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(r_ids) > MAX_LEN:
            dropped["long"] += 1
            continue
        out.append((p_ids, r_ids))
        if len(out) == N_TRAIN:
            break
    data = dict(examples=out, dropped=dropped)
    print("dataset built:", len(out), "dropped:", dropped)
    atomic_save(data, cache_path)
    return data


def collate(examples, pad_id):
    L = max(len(p) + len(r) for p, r in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for i, (p, r) in enumerate(examples):
        n = len(p) + len(r)
        ids[i, :n] = torch.tensor(p + r)
        labels[i, len(p):n] = torch.tensor(r)
        att[i, :n] = 1
    return ids, att, labels

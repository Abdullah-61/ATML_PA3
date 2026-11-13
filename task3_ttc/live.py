"""Step 4: single long response vs 3 parallel samples vs sequential refinement on 50 MATH eval questions.

    python live.py --data ATML-PA3-Data --out OUT --shard 0 --n_shards 2

Each shard (one GPU) handles a contiguous block of the 50 questions; all strategies of a question run on the same GPU.
Writes OUT/live_calls_<shard>.jsonl (one row per generation call, resumable) and OUT/live_env_<shard>.json.
"""
import argparse
import hashlib
import json
import os
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as C

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
SUFFIX = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, retain its answer. "
          "Give your final answer inside \\boxed{}.")
SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)
CAPS = dict(single=3072, parallel=1024, refine=2048)


def last_boxed(text):
    """Last literal \\boxed{ with balanced braces; absent, empty or unclosed -> None (README rule)."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                s = text[start:j].strip()
                return s or None
        j += 1
    return None


def seed_for(qid, strategy, k):
    return int(hashlib.sha256(f"{C.SEED}|{qid}|{strategy}|{k}".encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    g = pq.read_table(f"{args.data}/qwen25-math/generations.parquet",
                      columns=["sample_id", "rollout_id", "question_text", "gold_answer"]).to_pandas()
    g = g[g.rollout_id == 0].set_index("sample_id")
    split = C.make_split(list(g.index))
    live = split["eval"][:C.N_LIVE]
    per = -(-len(live) // args.n_shards)
    mine = live[args.shard * per:(args.shard + 1) * per]
    path = f"{args.out}/live_calls_{args.shard}.jsonl"
    done = set()
    if os.path.exists(path):
        done = {(r["sample_id"], r["strategy"], r["k"]) for r in map(json.loads, open(path))}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).cuda().eval()
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    env = dict(model=MODEL, revision=REVISION, dtype="float16", batch_size=1, gpu=torch.cuda.get_device_name(0),
               torch=torch.__version__, transformers=__import__("transformers").__version__, eos_ids=sorted(eos),
               sampling=SAMPLING, caps=CAPS, questions=mine)
    json.dump(env, open(f"{args.out}/live_env_{args.shard}.json", "w"), indent=1)

    def gen(messages, cap, seed):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=cap,
                                 pad_token_id=tok.pad_token_id, **SAMPLING)
        new = out[0, ids.shape[1]:].tolist()
        stopped = len(new) > 0 and new[-1] in eos
        kept = new[:-1] if stopped else new
        text = tok.decode(kept, skip_special_tokens=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        return dict(prompt=prompt, input_tokens=int(ids.shape[1]), output_tokens=len(new), retained_tokens=len(kept),
                    truncated=bool(not stopped and len(new) >= cap), cap=cap, seed=seed, seconds=dt,
                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, response=text,
                    extracted=last_boxed(text))

    gen([{"role": "user", "content": "1+1?"}], 8, 0)   # warm-up, not recorded
    f = open(path, "a")
    for qid in mine:
        prob, gold = g.loc[qid, "question_text"], g.loc[qid, "gold_answer"]
        base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prob + "\n\n" + SUFFIX}]
        calls = [("single", 0, base, CAPS["single"])] + [("parallel", k, base, CAPS["parallel"]) for k in range(3)]
        for strategy, k, msgs, cap in calls:
            if (qid, strategy, k) in done:
                continue
            r = gen(msgs, cap, seed_for(qid, strategy, k))
            f.write(json.dumps(dict(sample_id=qid, strategy=strategy, k=k, gold=gold, **r)) + "\n"); f.flush()
        if (qid, "refine", 0) not in done:
            first = [json.loads(l) for l in open(path)]
            first = next(r for r in first if r["sample_id"] == qid and r["strategy"] == "parallel" and r["k"] == 0)
            msgs = base + [{"role": "assistant", "content": first["response"]}, {"role": "user", "content": REVIEW}]
            r = gen(msgs, CAPS["refine"], seed_for(qid, "refine", 0))
            f.write(json.dumps(dict(sample_id=qid, strategy="refine", k=0, gold=gold, **r)) + "\n"); f.flush()
        print("done", qid, flush=True)
    print("live done", flush=True)


def grade(out):
    """Correctness and per-strategy accounting (gold answers used only here, after all generation)."""
    import glob
    import grading as G
    rows = [json.loads(l) for p in sorted(glob.glob(f"{out}/live_calls_*.jsonl")) for l in open(p)]
    by = {}
    for r in rows:
        by.setdefault(r["sample_id"], {})[(r["strategy"], r["k"])] = r
    res = []
    for qid, d in by.items():
        gold = d[("single", 0)]["gold"]
        cor = lambda s: G.math_groups([(0, s)], gold)[1][G.math_groups([(0, s)], gold)[0][0]]
        par = [d[("parallel", k)] for k in range(3)]
        a, c, reps = G.math_groups([(k, p["extracted"]) for k, p in enumerate(par)], gold)
        votes = {}
        for k in range(3):
            votes.setdefault(a[k], []).append(k)
        win = max(votes, key=lambda x: (len(votes[x]), -votes[x][0]))
        ref = d[("refine", 0)]
        res.append(dict(sample_id=qid, gold=gold,
                        single_answer=d[("single", 0)]["extracted"], single_correct=bool(cor(d[("single", 0)]["extracted"])),
                        parallel_answers=[p["extracted"] for p in par], parallel_groups=[a[k] for k in range(3)],
                        parallel_answer=reps[win], parallel_correct=bool(c[win]),
                        parallel_coverage=bool(any(c[a[k]] for k in range(3))),
                        initial_answer=par[0]["extracted"], initial_correct=bool(c[a[0]]),
                        refine_answer=ref["extracted"], refine_correct=bool(cor(ref["extracted"]))))
    json.dump(res, open(f"{out}/live_graded.json", "w"), indent=1)
    print("graded", len(res), flush=True)


if __name__ == "__main__":
    import sys
    if "--grade" in sys.argv:
        grade(sys.argv[sys.argv.index("--grade") + 1])
    else:
        main()

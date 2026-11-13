"""Resumable lm-eval runs. Each (condition, task) result is written once; finished ones are skipped."""
import argparse
import os

import common as C

TASKS = ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]


def path_for(cond, root):
    return {"baseline": C.MODEL,
            "full_mid": f"{root}/runs/full/export_step625",
            "full": f"{root}/runs/full/export_final"}.get(cond, f"{root}/runs/{cond}/export_final")


def nothink_tokenizer(root):
    """Local tokenizer whose chat template defaults to enable_thinking=False (no lm_eval kwarg needed)."""
    d = os.path.join(root, "tok_nothink")
    if not os.path.exists(os.path.join(d, "tokenizer_config.json")):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(C.MODEL)
        tok.chat_template = "{%- set enable_thinking = false %}\n" + tok.chat_template
        tok.save_pretrained(d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True,
                    choices=["baseline", "full_mid", "full", "l2sp", "lora4", "lora16", "oplora"])
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--batch_size", default="32", help="fixed; 'auto' picked 1 for generation and was ~20x slower")
    args = ap.parse_args()
    from lm_eval import simple_evaluate
    out = os.path.join(args.root, "evals", args.cond)
    os.makedirs(out, exist_ok=True)
    tok_dir = nothink_tokenizer(args.root)
    for task in TASKS:
        f = os.path.join(out, f"{task}.json")
        if os.path.exists(f):
            print(f"skip {args.cond}/{task} (done)"); continue
        gen = task == "gsm8k_cot"
        kw = dict(apply_chat_template=True,
                  gen_kwargs="temperature=0,do_sample=False,max_gen_toks=1024") if gen else {}
        res = simple_evaluate(
            model="hf", model_args=f"pretrained={path_for(args.cond, args.root)},tokenizer={tok_dir},dtype=float16",
            tasks=[task], num_fewshot=0, log_samples=True,
            # loglik tasks with long passages (boolq) OOM a 15 GB T4 at 32; batch size does not change loglik scores
            batch_size=(int(args.batch_size) if gen else 8) if args.batch_size.isdigit() else args.batch_size,
            random_seed=C.SEED, numpy_random_seed=C.SEED, torch_random_seed=C.SEED,
            fewshot_random_seed=C.SEED, **kw)
        C.atomic_json(dict(results=res["results"], samples=res["samples"], config=res.get("config")), f)
        print(args.cond, task, res["results"][task])


if __name__ == "__main__":
    main()

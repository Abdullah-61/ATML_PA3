"""Final-test evaluation of the completed Stage-3 model under four image conditions.

    python final_eval.py --data DATA --out OUT/run196

Writes OUT/final/<suite>_<condition>.jsonl (raw references, generations, per-example metrics),
OUT/final/mismatch_map.json and OUT/final/summary.json (tables, advantages, pair-bootstrap 95% CIs).
"""
import argparse
import collections
import json
import os
import random
import time

import numpy as np
import torch

import common as C

CONDS = ["correct", "complementary", "mismatched", "none"]
N_BOOT = 4000


def score(ex, pred):
    """(accuracy, exact match to the canonical answer) for one prediction; empty outputs score 0."""
    p = C.normalize_answer(pred)
    em = float(p == ex["canonical"])
    if ex["suite"] == "vqa":
        return C.vqa_soft_acc(p, [C.normalize_answer(a) for a in ex["answers"]]), em
    return em, em


def image_keys(pairs, mismatch):
    from vlm import key
    k = {"correct": [], "complementary": [], "mismatched": [], "none": []}
    for i, p in enumerate(pairs):
        for j in range(2):
            k["correct"].append(key(p[j]))
            k["complementary"].append(key(p[1 - j]))
            k["mismatched"].append(key(pairs[mismatch[i]][j]))
            k["none"].append(None)
    return k


def make_mismatch(pairs, tag):
    """Fixed before generation: pair i borrows the same-side image of another pair (never its own two images)."""
    r = random.Random(f"{C.SEED}-mismatch-{tag}")
    order = list(range(len(pairs)))
    r.shuffle(order)
    m = {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}
    assert all(m[i] != i for i in m)
    return [m[i] for i in range(len(pairs))]


def boot_ci(per_pair, rng_idx):
    v = per_pair[rng_idx].mean(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarize(pairs, recs, gen_time):
    """recs[cond] = per-example dicts in pair order (2 per pair)."""
    P = len(pairs)
    idx = np.random.default_rng(C.SEED).integers(0, P, size=(N_BOOT, P))   # same resampled pair ids for all conditions
    types = sorted({p[0]["type"] for p in pairs})
    out = {}
    per_pair = {}
    for c in CONDS:
        r = recs[c]
        acc = np.array([x["acc"] for x in r]).reshape(P, 2).mean(1)
        em = np.array([x["em"] for x in r]).reshape(P, 2)
        nll = np.array([x["nll"] for x in r]).reshape(P, 2).mean(1)
        pair_ok = em.min(1)
        ident = np.array([r[2 * i]["pred_norm"] == r[2 * i + 1]["pred_norm"] for i in range(P)], dtype=float)
        empty = np.array([x["pred_norm"] == "" for x in r], dtype=float).reshape(P, 2).mean(1)
        per_pair[c] = dict(acc=acc, pair_acc=pair_ok, nll=nll)
        d = dict(acc=float(acc.mean()), acc_ci=boot_ci(acc, idx), pair_acc=float(pair_ok.mean()),
                 pair_acc_ci=boot_ci(pair_ok, idx), identical_answer_rate=float(ident.mean()),
                 empty_rate=float(empty.mean()), ref_nll=float(nll.mean()), ref_nll_ci=boot_ci(nll, idx),
                 gen_seconds_per_answer=gen_time[c] / len(r), by_type={})
        for t in types:
            ti = [i for i, p in enumerate(pairs) if p[0]["type"] == t]
            d["by_type"][t] = dict(n_pairs=len(ti), acc=float(acc[ti].mean()), pair_acc=float(pair_ok[ti].mean()),
                                   ref_nll=float(nll[ti].mean()), identical_answer_rate=float(ident[ti].mean()))
        if c == "complementary":  # does the answer follow the partner image, keep the original, or neither?
            follow = np.array([x["follows_partner"] for x in r], dtype=float)
            keep = np.array([x["em"] for x in r], dtype=float)
            d["follows_partner_answer"] = float(follow.mean())
            d["keeps_original_answer"] = float(keep.mean())
            d["other_answer"] = float(1 - follow.mean() - keep.mean())
            d["partner_acc"] = float(np.mean([x["partner_acc"] for x in r]))
            for t in types:
                ti = [2 * i + j for i, p in enumerate(pairs) if p[0]["type"] == t for j in range(2)]
                d["by_type"][t].update(follows_partner_answer=float(follow[ti].mean()),
                                       keeps_original_answer=float(keep[ti].mean()))
        out[c] = d
    adv = {}
    for c in ["complementary", "mismatched", "none"]:
        a = {}
        for m, sign in [("acc", 1), ("pair_acc", 1), ("nll", -1)]:
            diff = sign * (per_pair["correct"][m] - per_pair[c][m])   # positive = correct image better
            a[m] = dict(mean=float(diff.mean()), ci=boot_ci(diff, idx))
        adv[f"correct_vs_{c}"] = a
    return dict(n_pairs=P, conditions=out, advantages=adv, bootstrap=dict(resamples=N_BOOT, unit="pair", seed=C.SEED))


def main():
    from vlm import VLM, Features
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--weights", default=None, help="default: OUT/stage3_weights.pt (the final model)")
    ap.add_argument("--limit", type=int, default=None, help="debug only")
    args = ap.parse_args()
    fdir = f"{args.out}/final"
    os.makedirs(fdir, exist_ok=True)
    if os.path.exists(f"{fdir}/summary.json"):
        print("final eval already done"); return
    sel = C.load_json(f"{args.data}/selection.json")
    feats = Features(args.data)
    torch.manual_seed(C.SEED)
    model = VLM(pool=args.pool)
    model.load_adapter_state(torch.load(args.weights or f"{args.out}/stage3_weights.pt", map_location="cpu"))
    model.connector.scale.fill_(C.load_json(f"{args.out}/calibration.json")["scale"])
    model.eval()
    summary = {}
    for suite in ["ground_test", "vqa_test"]:
        pairs = sel[suite][:args.limit] if args.limit else sel[suite]
        mm = make_mismatch(pairs, suite)
        C.atomic_json(dict(zip([p[0]["pair_id"] for p in pairs], [pairs[j][0]["pair_id"] for j in mm])),
                      f"{fdir}/mismatch_map_{suite}.json")
        exs = [e for p in pairs for e in p]
        keys = image_keys(pairs, mm)
        recs, gen_time = {}, {}
        for c in CONDS:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            preds = model.generate(exs, keys[c], feats)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_time[c] = time.time() - t0
            nll = model.nll(exs, keys[c], feats)   # original references, termination excluded
            rows = []
            for i, (e, pr, n, k) in enumerate(zip(exs, preds, nll, keys[c])):
                partner = exs[i ^ 1]
                acc, em = score(e, pr)
                pn = C.normalize_answer(pr)
                rows.append(dict(qid=e["qid"], pair_id=e["pair_id"], type=e["type"], question=e["question"],
                                 image=k, target=e["target"], canonical=e["canonical"],
                                 answers=e.get("answers"), prediction=pr, pred_norm=pn, acc=acc, em=em, nll=n,
                                 partner_canonical=partner["canonical"], follows_partner=float(pn == partner["canonical"]),
                                 partner_acc=score(partner, pr)[0]))
            with open(f"{fdir}/{suite}_{c}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            recs[c] = rows
            print(suite, c, f"acc={np.mean([r['acc'] for r in rows]):.4f} nll={np.mean(nll):.4f} "
                            f"{gen_time[c]:.1f}s", flush=True)
        summary[suite] = summarize(pairs, recs, gen_time)
    summary["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    C.atomic_json(summary, f"{fdir}/summary.json")
    print("final eval done", flush=True)


if __name__ == "__main__":
    main()

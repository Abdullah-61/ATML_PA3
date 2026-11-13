"""Offline experiments on the supplied rollouts (Steps 1, 2, 3, 5 and the data for 6).

    python offline.py --data ATML-PA3-Data --meta testmini.json --out OUT

Writes to OUT:
  splits.json, orders_<ds>.npy                 dev/eval split and the 10 rollout orders per question
  rollouts_<ds>.parquet                        per rollout: answer group, correctness, length, L, K, V, N=32 VisConf ...
  groups_<ds>.json                             group representatives / correctness per question
  selections_<ds>.parquet                      per (question, order, N, method): selected rollout + answer, correct, cost
  adaptive_state_<ds>.parquet                  per (question, order, n): SC / RWC state used by the stopping rules
  adaptive_grid_<ds>.csv, adaptive_<ds>.json   dev grid, frozen settings, held-out per-(question, order) outcomes
  trajectories.json                            token-level signals of fixed illustrative MathVista rollouts
  scorer.json                                  scorer version, package versions, normalization coverage
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

import common as C
import grading as G


# ----------------------------------------------------------------------------- loading
def load(data, ds):
    d = f"{data}/{C.DATASETS[ds]}"
    gen = pq.read_table(f"{d}/generations.parquet", columns=["dataset", "sample_id", "rollout_id", "question_text",
                        "generated_token_ids", "n_generated_text_tokens", "hit_max_new_tokens", "gold_answer", "generated_text",
                        "extracted_answer"]).to_pandas()
    prob = pq.read_table(f"{d}/probability_metrics.parquet").to_pandas()
    df = gen.merge(prob, on=["dataset", "sample_id", "rollout_id"], how="inner", validate="one_to_one")
    if ds == "mathvista":
        att = pq.read_table(f"{d}/attention_metrics.parquet").to_pandas()
        df = df.merge(att, on=["dataset", "sample_id", "rollout_id"], how="inner", validate="one_to_one",
                      suffixes=("", "_att"))
    assert len(df) == 32000, len(df)
    # token-list alignment checks
    T = df["n_generated_text_tokens"].to_numpy()
    lens = {c: df[c].map(len).to_numpy() for c in ["generated_token_ids", "logp", "kl_u_p"] +
            (["img_attn_total", "prompt_text_attn_total", "generated_text_attn_total", "n_generated_text_tokens_att"]
             if ds == "mathvista" else [])}
    align = {c: bool((v == T).all()) for c, v in lens.items()}
    if ds == "mathvista":
        align["att_prefix_counts_0..T-1"] = bool(all((np.asarray(x) == np.arange(len(x))).all()
                                                     for x in df["n_generated_text_tokens_att"]))
    assert all(align.values()), align
    df = df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)
    return df, align


# ----------------------------------------------------------------------------- grouping
def _group_math(args):
    q, ex, gold = args
    a, c, r = G.math_groups(ex, gold)
    return q, a, c, r


def add_groups(df, ds, meta):
    rows = [(q, list(zip(g.rollout_id, g.extracted_answer)), g.gold_answer.iloc[0]) for q, g in df.groupby("sample_id")]
    if ds == "math":
        with mp.Pool(max(1, os.cpu_count())) as pool:
            res = pool.map(_group_math, rows, chunksize=8)
    else:
        res = [(q, *G.mathvista_groups(ex, meta[q])) for q, ex, _ in rows]
    assign, corr, reps = {}, {}, {}
    for q, a, c, r in res:
        assign[q], corr[q], reps[q] = a, c, r
    df["group"] = [assign[q][r] for q, r in zip(df.sample_id, df.rollout_id)]
    df["correct"] = [corr[q][g] for q, g in zip(df.sample_id, df.group)]
    gold_norm = {q: G.mathvista_gold(meta[q]) for q in assign} if ds == "mathvista" else \
        dict(zip(df.sample_id, df.gold_answer))
    return df, dict(reps=reps, correct=corr, gold_norm=gold_norm)


# ----------------------------------------------------------------------------- scores
def rollout_scores(df, ds):
    T = df.n_generated_text_tokens.to_numpy().astype(float)
    mean = lambda col: np.array([np.mean(x) if len(x) else np.nan for x in col])
    df["L"] = mean(df.logp)
    df["K"] = mean(df.kl_u_p)
    if ds == "mathvista":
        vt = []
        for a, b, c in zip(df.img_attn_total, df.prompt_text_attn_total, df.generated_text_attn_total):
            a, b, c = (np.asarray(x, dtype=np.float64) for x in (a, b, c))
            vt.append(a / np.maximum(a + b + c, 1e-12))
        df["v_tok"] = vt
        df["V"] = mean(df.v_tok)
    df["T"] = T
    return df


def visconf(kl_list, v_list):
    """Candidate-set calibration (pooled 5th/95th linear percentiles of token KL over the given rollouts)."""
    pool = np.concatenate(kl_list)
    q05, q95 = np.percentile(pool, [5, 95])
    den = max(q95 - q05, 1e-8)
    Cs, Zs, Zts, cs = [], [], [], []
    for k, v in zip(kl_list, v_list):
        c = np.clip((k - q05) / den, 0.0, 1.0)
        if len(c):
            Cs.append(c.mean()); Zs.append(c.mean() * v.mean()); Zts.append((c * v).mean())
        else:
            Cs.append(np.nan); Zs.append(np.nan); Zts.append(np.nan)
        cs.append(c)
    return np.array(Cs), np.array(Zs), np.array(Zts), cs, (q05, q95)


# ----------------------------------------------------------------------------- selection rules
def sc(idx, grp):
    """Self-consistency: one vote per rollout; ties -> group with the earliest supporting rollout in this order."""
    votes, first = collections.Counter(), {}
    for p, i in enumerate(idx):
        votes[grp[i]] += 1
        first.setdefault(grp[i], p)
    win = max(votes, key=lambda g: (votes[g], -first[g]))
    return idx[first[win]], win, votes, first


def argmax_first(idx, q):
    q = np.where(np.isnan(q), -np.inf, q)
    p = int(np.argmax(q))          # first maximum = earliest rollout in the current order
    return idx[p]


def rwc(idx, grp, q):
    """Rank-weighted consensus; average ranks for ties, rank 1 = best, w = (N - r)/(N - 1)."""
    N = len(idx)
    q = np.where(np.isnan(q), -np.inf, q)
    r = rankdata(-q, method="average")
    w = (N - r) / (N - 1)
    W, first = collections.defaultdict(float), {}
    for p, i in enumerate(idx):
        W[grp[i]] += w[p]
        first.setdefault(grp[i], p)
    Wr = {g: round(v, 9) for g, v in W.items()}
    win = max(Wr, key=lambda g: (Wr[g], -first[g]))
    tot = sum(W.values())
    vals = sorted(W.values(), reverse=True)
    margin = (vals[0] - (vals[1] if len(vals) > 1 else 0.0)) / tot if tot > 0 else 0.0
    return idx[first[win]], win, margin


METHODS = {"math": ["FirstAns", "SC", "MaxLogP", "MaxSC", "SC_RWC", "Shortest"],
           "mathvista": ["FirstAns", "SC", "MaxLogP", "MaxSC", "SC_RWC", "Shortest",
                         "MaxVis", "Vis_RWC", "MaxVisConf", "VisConf_RWC", "MaxVisConfTok", "VisConfTok_RWC"]}


def run_question(q, sub, orders, ds):
    """All budgets x orders x methods for one question. sub: the 32 rollouts sorted by rollout_id."""
    grp = sub.group.tolist()
    cor = dict(zip(sub.group, sub.correct))
    T, L, K = sub["T"].to_numpy(), sub.L.to_numpy(), sub.K.to_numpy()
    if ds == "mathvista":
        V = sub.V.to_numpy()
        kl = [np.asarray(x, dtype=np.float64) for x in sub.kl_u_p]
        vt = list(sub.v_tok)
    sel, state = [], []
    for o, order in enumerate(orders):
        for N in C.BUDGETS:
            idx = list(order[:N])
            cost = float(T[idx].sum())
            cover = any(sub.correct.iloc[i] for i in idx)
            ngr = len({grp[i] for i in idx})
            out = {}
            out["FirstAns"] = (idx[0], float(T[idx[0]]))
            s_r, s_g, votes, first = sc(idx, grp)
            out["SC"] = (s_r, cost)
            if N == 1:
                for m in METHODS[ds]:
                    out[m] = (idx[0], cost)          # every rule returns the only candidate
            else:
                out["MaxLogP"] = (argmax_first(idx, L[idx]), cost)
                out["MaxSC"] = (argmax_first(idx, K[idx]), cost)
                out["SC_RWC"] = (rwc(idx, grp, K[idx])[0], cost)
                out["Shortest"] = (argmax_first(idx, -T[idx]), cost)
                if ds == "mathvista":
                    Cc, Z, Zt, _, _ = visconf([kl[i] for i in idx], [vt[i] for i in idx])
                    out["MaxVis"] = (argmax_first(idx, V[idx]), cost)
                    out["Vis_RWC"] = (rwc(idx, grp, V[idx])[0], cost)
                    out["MaxVisConf"] = (argmax_first(idx, Z), cost)
                    out["VisConf_RWC"] = (rwc(idx, grp, Z)[0], cost)
                    out["MaxVisConfTok"] = (argmax_first(idx, Zt), cost)
                    out["VisConfTok_RWC"] = (rwc(idx, grp, Zt)[0], cost)
            for m in METHODS[ds]:
                r, c = out[m]
                sel.append((q, o, N, m, int(r), grp[r], bool(cor[grp[r]]), c, cover, ngr))
            if N in C.STOP_AT + [32]:
                qual = K[idx] if ds == "math" else visconf([kl[i] for i in idx], [vt[i] for i in idx])[1]
                rr, rg, margin = rwc(idx, grp, qual) if N > 1 else (idx[0], grp[idx[0]], 1.0)
                top = max(votes.values())
                state.append((q, o, N, s_g, bool(cor[s_g]), top / N, sum(v == top for v in votes.values()) == 1,
                              s_g == C.NA, rg, margin, cost))
    return sel, state


# ----------------------------------------------------------------------------- adaptive stopping
def simulate(st, tau, gamma, quality):
    """st: adaptive_state rows of one dataset (all n). Returns per (q, o) outcome for one setting (gamma None = agreement)."""
    st = st.sort_values(["sample_id", "order", "n"])
    ok = (st.lead_frac >= tau - 1e-12) & st.lead_unique & ~st.lead_is_na
    if quality:
        ok &= (st.sc_answer == st.rwc_answer) & (st.rwc_margin >= gamma - 1e-12)
    ok |= st.n == 32
    first = st[ok].groupby(["sample_id", "order"], sort=False).head(1)
    return first[["sample_id", "order", "n", "sc_answer", "sc_correct", "cost"]].reset_index(drop=True)


def adaptive(st, split, ds):
    dev, ev = set(split["dev"]), set(split["eval"])
    sd = st[st.sample_id.isin(dev)]
    f32 = simulate(sd, 2.0, None, False)      # tau > 1 never satisfied -> always 32
    acc32 = f32.groupby("sample_id").sc_correct.mean().mean()
    grid = [("fixed32", None, None)] + [("agreement", t, None) for t in C.TAUS] + \
           [("quality", t, g) for t in C.TAUS for g in C.GAMMAS]
    rows = []
    for fam, t, g in grid:
        r = f32 if fam == "fixed32" else simulate(sd, t, g, fam == "quality")
        pq_ = r.groupby("sample_id").agg(acc=("sc_correct", "mean"), cost=("cost", "mean"), n=("n", "mean")).mean()
        rows.append(dict(family=fam, tau=t, gamma=g, dev_acc=pq_.acc, dev_cost=pq_.cost, dev_rollouts=pq_.n))
    grid_df = pd.DataFrame(rows)
    chosen = {}
    for fam in ["agreement", "quality"]:
        cand = grid_df[grid_df.family.isin([fam, "fixed32"]) & (grid_df.dev_acc >= acc32 - 0.01 - 1e-12)].copy()
        cand["t"] = cand.tau.fillna(9); cand["g"] = cand.gamma.fillna(9)
        best = cand.sort_values(["dev_cost", "dev_acc", "t", "g"], ascending=[True, False, True, True]).iloc[0]
        chosen[fam] = dict(family=best.family, tau=None if pd.isna(best.tau) else float(best.tau),
                           gamma=None if pd.isna(best.gamma) else float(best.gamma), dev_acc=float(best.dev_acc),
                           dev_cost=float(best.dev_cost))
    se = st[st.sample_id.isin(ev)]
    outs = {}
    for fam, c in chosen.items():
        if c["family"] == "fixed32":
            r = simulate(se, 2.0, None, False)
        else:
            r = simulate(se, c["tau"], c["gamma"], fam == "quality")
        outs[fam] = r
    outs["fixed32"] = simulate(se, 2.0, None, False)
    return grid_df, dict(dev_fixed32_acc=float(acc32), chosen=chosen), outs


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--meta", required=True, help="MathVista testmini.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(args.meta))
    splits, info = {}, {}
    import math_verify
    info["scorer_version"] = G.SCORER_VERSION
    info["math_verify_version"] = getattr(math_verify, "__version__", None)
    try:
        from importlib.metadata import version
        info["math_verify_version"] = version("math-verify")
    except Exception:
        pass
    traj = {}
    for ds in ["math", "mathvista"]:
        t0 = time.time()
        df, align = load(args.data, ds)
        ids = sorted(df.sample_id.unique())
        assert len(ids) == 1000 and (df.groupby("sample_id").size() == 32).all()
        split = C.make_split(ids)
        splits[ds] = split
        orders = C.make_orders(ids)
        np.save(f"{args.out}/orders_{ds}.npy", np.stack([orders[q] for q in ids]))
        df, groups = add_groups(df, ds, meta)
        df = rollout_scores(df, ds)
        df["split"] = df.sample_id.map(lambda q: "dev" if q in set(split["dev"]) else "eval")
        info[ds] = dict(alignment=align, rows=len(df), load_group_seconds=time.time() - t0,
                        missing_extraction=int(df.extracted_answer.map(G.is_missing).sum()),
                        no_answer_after_norm=int((df.group == C.NA).sum()), capped=int(df.hit_max_new_tokens.sum()),
                        zero_length=int((df["T"] == 0).sum()), rollout_correct_rate=float(df.correct.mean()))
        if ds == "mathvista":   # rollouts whose non-missing extraction failed numeric normalization (dev split only)
            bad = df[(df.group == C.NA) & ~df.extracted_answer.map(G.is_missing) & (df.split == "dev")]
            info[ds]["unparsable_dev_examples"] = bad.extracted_answer.head(40).tolist()
            info[ds]["unparsable_total"] = int(((df.group == C.NA) & ~df.extracted_answer.map(G.is_missing)).sum())
        print(ds, json.dumps(info[ds])[:2000], flush=True)
        json.dump(groups, open(f"{args.out}/groups_{ds}.json", "w"))
        # per-question runs
        sel, state, extra = [], [], []
        for qi, (q, sub) in enumerate(df.groupby("sample_id", sort=True)):
            sub = sub.sort_values("rollout_id")
            s, st = run_question(q, sub, orders[q], ds)
            sel += s; state += st
            if ds == "mathvista":   # N = 32 signals (order-invariant)
                kl = [np.asarray(x, dtype=np.float64) for x in sub.kl_u_p]
                Cc, Z, Zt, cs, qq = visconf(kl, list(sub.v_tok))
                corr = [float(np.corrcoef(c, v)[0, 1]) if len(c) > 2 and np.std(c) > 0 and np.std(v) > 0 else np.nan
                        for c, v in zip(cs, sub.v_tok)]
                extra += [dict(sample_id=q, rollout_id=int(r), C32=a, Z32=b, Ztok32=c, cv_corr=d)
                          for r, a, b, c, d in zip(sub.rollout_id, Cc, Z, Zt, corr)]
            if qi % 200 == 0:
                print(ds, qi, f"{time.time() - t0:.0f}s", flush=True)
        sel = pd.DataFrame(sel, columns=["sample_id", "order", "N", "method", "sel_rollout", "sel_answer", "correct",
                                         "cost", "coverage", "n_groups"])
        sel["split"] = sel.sample_id.map(lambda q: "dev" if q in set(split["dev"]) else "eval")
        sel.to_parquet(f"{args.out}/selections_{ds}.parquet", index=False)
        st = pd.DataFrame(state, columns=["sample_id", "order", "n", "sc_answer", "sc_correct", "lead_frac",
                                          "lead_unique", "lead_is_na", "rwc_answer", "rwc_margin", "cost"])
        st.to_parquet(f"{args.out}/adaptive_state_{ds}.parquet", index=False)
        keep = ["sample_id", "rollout_id", "split", "extracted_answer", "group", "correct", "gold_answer",
                "hit_max_new_tokens", "T", "L", "K"] + (["V"] if ds == "mathvista" else [])
        roll = df[keep].copy()
        if ds == "mathvista":
            roll = roll.merge(pd.DataFrame(extra), on=["sample_id", "rollout_id"])
        roll.to_parquet(f"{args.out}/rollouts_{ds}.parquet", index=False)
        # adaptive stopping
        grid, chosen, outs = adaptive(st, split, ds)
        grid.to_csv(f"{args.out}/adaptive_grid_{ds}.csv", index=False)
        for fam, r in outs.items():
            r.to_parquet(f"{args.out}/adaptive_eval_{ds}_{fam}.parquet", index=False)
        C.atomic_json(chosen, f"{args.out}/adaptive_{ds}.json")
        print(ds, "adaptive", json.dumps(chosen), flush=True)
        # question texts + generated text for qualitative examples (eval questions only, truncated)
        df[["sample_id", "rollout_id", "question_text", "generated_text"]].assign(
            generated_text=lambda x: x.generated_text.str.slice(0, 4000)).to_parquet(f"{args.out}/texts_{ds}.parquet",
                                                                                    index=False)
        if ds == "mathvista":
            # fixed rule: first 4 eval questions (split order) with mixed correctness; per question the highest-Z32
            # correct and the highest-Z32 incorrect rollout
            rz = roll.set_index(["sample_id", "rollout_id"])
            n = 0
            for q in split["eval"]:
                r = roll[roll.sample_id == q]
                if r.correct.all() or not r.correct.any():
                    continue
                sub = df[df.sample_id == q].sort_values("rollout_id")
                kl = [np.asarray(x, dtype=np.float64) for x in sub.kl_u_p]
                _, _, _, cs, qq = visconf(kl, list(sub.v_tok))
                for lab in [True, False]:
                    rid = int(r[r.correct == lab].sort_values("Z32", ascending=False).rollout_id.iloc[0])
                    row = sub[sub.rollout_id == rid].iloc[0]
                    traj[f"{q}/{rid}"] = dict(sample_id=q, rollout_id=rid, correct=lab, group=row.group,
                                              question=row.question_text[:600], text=row.generated_text[:3000],
                                              c=cs[rid].round(4).tolist(), v=np.asarray(row.v_tok).round(5).tolist(),
                                              logp=np.asarray(row.logp).round(4).tolist(), q05_q95=list(qq),
                                              Z32=float(rz.loc[(q, rid)].Z32), cv_corr=float(rz.loc[(q, rid)].cv_corr))
                n += 1
                if n == 4:
                    break
        info[ds]["seconds"] = time.time() - t0
    C.atomic_json(splits, f"{args.out}/splits.json")
    C.atomic_json(info, f"{args.out}/scorer.json")
    C.atomic_json(traj, f"{args.out}/trajectories.json")
    print("offline done", flush=True)


if __name__ == "__main__":
    main()

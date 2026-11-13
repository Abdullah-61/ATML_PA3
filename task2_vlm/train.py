"""Three-stage curriculum (caption alignment -> visual instruction tuning -> natural VQA), resumable.

    python train.py --data DATA --out OUT/run196 --pool 1      # 196 visual tokens (main run)
    python train.py --data DATA --out OUT/run49  --pool 2      # optional 49-token run

Re-running the same command continues from OUT/ckpt_latest.pt (saved every 250 updates; holds connector/LoRA,
optimizer, scheduler, grad scaler, sampler position (= update index into the fixed, seeded draw schedule),
stage progress and RNG states). Connector/adapter weights are also saved after each stage.
"""
import argparse
import collections
import json
import math
import os
import random
import signal
import sys
import time

import numpy as np
import torch

import common as C
from vlm import VLM, Features, key

STOP = False


def _on_signal(*_):
    global STOP
    STOP = True
    print("signal received: checkpoint after the current update, then exit", flush=True)


# ----------------------------------------------------------------------------- data schedule
def build_schedule(sel):
    """Fixed list of 4500 updates x 16 examples. Each draw gives 2 examples (a full pair, or 2 captions)."""
    caps = sel["caption_train"]
    pools = {"presence": sel["presence_train_pool"], "counting": sel["counting_train_pool"], "vqa": sel["vqa_train"]}
    sched, mix = [], []
    r = random.Random(f"{C.SEED}-stage1")
    order, need = [], C.STAGES[0]["updates"] * C.BATCH
    while len(order) < need:  # full data: exactly two passes over the 18,000 caption examples
        p = list(range(len(caps)))
        r.shuffle(p)
        order += p
    assert C.os.environ.get("VLM_SMOKE") or len(order) == need == 2 * len(caps)
    order = order[:need]
    for u in range(C.STAGES[0]["updates"]):
        sched.append([caps[i] for i in order[u * C.BATCH:(u + 1) * C.BATCH]])
        mix.append(["caption"] * (C.BATCH // 2))
    for s, st in enumerate(C.STAGES[1:], start=2):
        r = random.Random(f"{C.SEED}-stage{s}")
        comps, probs = zip(*st["mix"].items())
        for _ in range(st["updates"]):
            exs, m = [], []
            for _ in range(C.BATCH // 2):
                c = r.choices(comps, probs)[0]
                exs += r.sample(caps, 2) if c == "caption" else r.choice(pools[c])
                m.append(c)
            sched.append(exs)
            mix.append(m)
    assert len(sched) == C.TOTAL_UPDATES and all(len(x) == C.BATCH for x in sched)
    return sched, mix


def stage_of(update):
    """Stage index (0..2) that the update with 0-based index `update` belongs to."""
    return next(i for i, e in enumerate(C.STAGE_ENDS) if update < e)


def lr_factor(t, n):
    w = math.ceil(C.WARMUP_FRAC * n)
    if t < w:
        return (t + 1) / w
    p = (t - w) / max(1, n - w)
    return C.MIN_LR_FRAC + (1 - C.MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p))


def rng_state():
    s = dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"].cpu())
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


# ----------------------------------------------------------------------------- dev evaluation
def flat(pairs):
    return [e for p in pairs for e in p]


def partner_keys(pairs):
    return [key(p[1 - j]) for p in pairs for j in range(2)]


def dev_eval(model, sel, feats, update):
    from final_eval import score  # same scoring as the final evaluation
    model.eval()
    t0 = time.time()
    caps = sel["caption_dev"]
    out = dict(update=update, caption_loss=float(np.mean(model.nll(caps, [key(e) for e in caps], feats, label_end=True))))
    for suite in ["ground_dev", "vqa_dev"]:
        exs = flat(sel[suite])
        conds = {"correct": [key(e) for e in exs], "complementary": partner_keys(sel[suite]), "none": [None] * len(exs)}
        nll = {c: model.nll(exs, k, feats) for c, k in conds.items()}
        for c, v in nll.items():
            out[f"{suite}/nll_{c}"] = float(np.mean(v))
        delta = np.array(nll["complementary"]) - np.array(nll["correct"])
        out[f"{suite}/delta_nll"] = float(delta.mean())
        for t in sorted({e["type"] for e in exs}):
            idx = [i for i, e in enumerate(exs) if e["type"] == t]
            out[f"{suite}/delta_nll/{t}"] = float(delta[idx].mean())
        preds = model.generate(exs, conds["correct"], feats)
        out[f"{suite}/acc_correct"] = float(np.mean([score(e, p)[0] for e, p in zip(exs, preds)]))
    out["eval_seconds"] = time.time() - t0
    model.train()
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=1, choices=[1, 2])
    ap.add_argument("--max_updates", type=int, default=C.TOTAL_UPDATES, help="debug only")
    ap.add_argument("--dev_limit", type=int, default=None, help="debug only: truncate dev sets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(f"{args.out}/DONE.json"):
        print("already finished"); return
    signal.signal(signal.SIGTERM, _on_signal); signal.signal(signal.SIGINT, _on_signal)
    dev = C.dev_type()

    sel = C.load_json(f"{args.data}/selection.json")
    if args.dev_limit:
        for k in ["caption_dev", "ground_dev", "vqa_dev"]:
            sel[k] = sel[k][:args.dev_limit]
    feats = Features(args.data)
    sched, mix = build_schedule(sel)

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    model = VLM(pool=args.pool)
    calib_path = f"{args.out}/calibration.json"
    if not os.path.exists(calib_path):
        r = random.Random(f"{C.SEED}-calib")
        pool_imgs = sorted({key(e) for e in sel["caption_train"]})
        cal_imgs = r.sample(pool_imgs, min(512, len(pool_imgs)))
        text = [model.tok(e["target"], add_special_tokens=False)["input_ids"] for e in sel["caption_train"][:1000]]
        text += [model.tok(model.user_text(e), add_special_tokens=False)["input_ids"]
                 for p in sel["presence_train_pool"][:250] + sel["counting_train_pool"][:250] + sel["vqa_train"][:250] for e in p]
        C.atomic_json(model.calibrate(feats.get(cal_imgs), [t for x in text for t in x]), calib_path)
    calib = C.load_json(calib_path)
    model.connector.scale.fill_(calib["scale"])
    model.train()

    cfg = dict(pool=args.pool, visual_tokens=model.connector.n_tokens, stages=C.STAGES, batch=C.BATCH,
               micro_bs=C.MICRO_BS, accum=C.ACCUM, wd=C.WD, clip=C.CLIP, warmup_frac=C.WARMUP_FRAC,
               min_lr_frac=C.MIN_LR_FRAC, caps=dict(question=C.Q_CAP, caption=C.CAP_CAP, answer=C.ANS_CAP),
               lora=C.LORA, seed=C.SEED, ckpt_every=C.CKPT_EVERY, dev_eval_at=C.DEV_EVAL_AT,
               caption_prompt=C.CAPTION_PROMPT, answer_suffix=C.ANSWER_SUFFIX, chat_template=model.template)
    C.atomic_json(cfg, f"{args.out}/config.json")
    mixture = {C.STAGES[s]["name"]: dict(collections.Counter(c for u in range(C.STAGE_ENDS[s] - C.STAGES[s]["updates"],
                                                                              C.STAGE_ENDS[s]) for c in mix[u]))
               for s in range(3)}
    C.atomic_json(mixture, f"{args.out}/mixture_counts.json")

    # ------------------------------------------------------------ resume
    latest = f"{args.out}/ckpt_latest.pt"
    update, ck = 0, None
    stats = dict(stages={})
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_adapter_state(ck["adapters"])
        update, stats = ck["update"], ck["stats"]
        print(f"RESUMED at update {update}", flush=True)
        for f in ["train_log.jsonl", "dev_metrics.jsonl"]:  # drop lines written after the checkpoint
            p = f"{args.out}/{f}"
            if os.path.exists(p):
                with open(p) as fh:
                    keep = [l for l in fh if json.loads(l)["update"] <= update]
                with open(p, "w") as fh:
                    fh.writelines(keep)
    tlog = open(f"{args.out}/train_log.jsonl", "a")
    dlog_path = f"{args.out}/dev_metrics.jsonl"
    done_evals = set()
    if os.path.exists(dlog_path):
        done_evals = {json.loads(l)["update"] for l in open(dlog_path)}
    if 0 not in done_evals and update == 0:
        r = dev_eval(model, sel, feats, 0)
        with open(dlog_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print("dev @0", json.dumps(r), flush=True)

    end = min(args.max_updates, C.TOTAL_UPDATES)
    while update < end:
        s = stage_of(update)
        st = C.STAGES[s]
        s_start = C.STAGE_ENDS[s] - st["updates"]
        if s >= 1 and not model.lora:
            torch.manual_seed(C.SEED + 1)
            model.add_lora()
        params_conn = list(model.connector.parameters())
        params_lora = [p for m in model.lora.values() for p in (m.A, m.B)]
        for p in params_conn:
            p.requires_grad_(True)
        for p in params_lora:
            p.requires_grad_(s >= 1)
        groups = [dict(params=[p for p in params_conn if p.ndim >= 2], lr=st["lr_conn"], weight_decay=C.WD),
                  dict(params=[p for p in params_conn if p.ndim < 2], lr=st["lr_conn"], weight_decay=0.0)]
        if s >= 1:
            groups.append(dict(params=params_lora, lr=st["lr_lora"], weight_decay=C.WD))
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, lambda t, n=st["updates"]: lr_factor(t, n))
        scaler = torch.amp.GradScaler(dev, init_scale=2.0 ** 10, enabled=dev == "cuda")  # 2^16 overflows fp16 grads
        trainable = sum(p.numel() for g in groups for p in g["params"])
        sstat = stats["stages"].setdefault(st["name"], dict(trainable_params=trainable, train_seconds=0.0,
                                                            peak_gpu_mem_gb=0.0, skipped_updates=[]))
        if ck is not None:
            if ck["stage"] == s and ck["update"] < C.STAGE_ENDS[s]:  # mid-stage: restore optimizer state too
                opt.load_state_dict(ck["opt"]); sched_lr.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            set_rng_state(ck["rng"])
        ck = None
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"stage {s + 1} {st['name']}: updates {update}->{C.STAGE_ENDS[s]}, trainable {trainable:,}", flush=True)

        def save_ckpt():
            sstat["peak_gpu_mem_gb"] = max(sstat["peak_gpu_mem_gb"], torch.cuda.max_memory_allocated() / 2**30
                                           if dev == "cuda" else 0.0)
            C.atomic_save(dict(adapters=model.adapter_state(), opt=opt.state_dict(), sched=sched_lr.state_dict(),
                               scaler=scaler.state_dict(), rng=rng_state(), update=update, stage=s, stats=stats),
                          latest)

        while update < min(C.STAGE_ENDS[s], end):
            t0 = time.time()
            exs = sched[update]
            opt.zero_grad(set_to_none=True)
            losses = []
            for m in range(C.ACCUM):
                mb = exs[m * C.MICRO_BS:(m + 1) * C.MICRO_BS]
                with torch.autocast(dev, dtype=C.amp_dtype()):
                    b = model.batch(mb, [key(e) for e in mb], feats)
                    per_ex = model.token_losses(*b)
                loss = per_ex.sum() / C.BATCH            # mean over examples of the effective batch
                scaler.scale(loss).backward()
                losses += per_ex.detach().tolist()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], C.CLIP).item()
            lrs = [g["lr"] for g in opt.param_groups]
            prev = scaler.get_scale()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            skipped = dev == "cuda" and scaler.get_scale() < prev
            sched_lr.step()
            update += 1
            if skipped:
                sstat["skipped_updates"].append(update)
            dt = time.time() - t0
            sstat["train_seconds"] += dt
            by_comp = collections.defaultdict(list)
            for i, c in enumerate(mix[update - 1]):
                by_comp[c] += losses[2 * i:2 * i + 2]
            rec = dict(update=update, stage=s + 1, loss=float(np.mean(losses)), grad_norm=gn, lr_conn=lrs[0],
                       lr_lora=lrs[2] if len(lrs) > 2 else None, scale=scaler.get_scale(), skipped=skipped,
                       seconds=dt, comp_loss={c: float(np.mean(v)) for c, v in by_comp.items()},
                       comp_draws=dict(collections.Counter(mix[update - 1])))
            tlog.write(json.dumps(rec) + "\n"); tlog.flush()
            if update % 25 == 0 or update == 1:
                print(f"[{update}/{C.TOTAL_UPDATES}] s{s + 1} loss={rec['loss']:.4f} gn={gn:.2f} lr={lrs[0]:.2e} "
                      f"{dt:.2f}s/upd", flush=True)
            if update in C.DEV_EVAL_AT and update not in done_evals:
                r = dev_eval(model, sel, feats, update)
                with open(dlog_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                done_evals.add(update)
                print(f"dev @{update}", json.dumps(r), flush=True)
            if update % C.CKPT_EVERY == 0 or update == C.STAGE_ENDS[s] or STOP or update == end:
                save_ckpt()
            if update == C.STAGE_ENDS[s]:
                C.atomic_save(model.adapter_state(), f"{args.out}/stage{s + 1}_weights.pt")
                print(f"stage {s + 1} done; weights saved", flush=True)
            if STOP:
                print("exiting for resume"); sys.exit(0)
    tlog.close()
    if update < C.TOTAL_UPDATES:
        C.atomic_save(model.adapter_state(), f"{args.out}/debug_weights.pt")
        print("max_updates reached (debug)"); return
    stats["total_train_seconds"] = sum(v["train_seconds"] for v in stats["stages"].values())
    C.atomic_json(stats, f"{args.out}/DONE.json")
    print("DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()

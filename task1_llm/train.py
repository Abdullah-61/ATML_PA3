"""Resumable SFT of Qwen3-0.6B on MetaMathQA: full, l2sp, lora4, lora16, oplora.

Re-running the same command after a disconnect continues from the newest checkpoint:
  * ckpt_latest.pt  - rolling checkpoint every --save_every updates (atomic write)
  * ckpt_step{312,625,937,1250}.pt - quarter-epoch recovery checkpoints
Each holds trainable weights, optimizer, scheduler, grad scaler, all RNG states and the
data position (the data order is fixed, so position == update index).
"""
import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as C

STOP = False


def _on_signal(*_):
    global STOP
    STOP = True
    print("signal received: will checkpoint after the current update and exit", flush=True)


# ----------------------------------------------------------------------------- adapters
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.Uk = self.Vk = None  # OPLoRA protected bases (fp32), set by attach_bases

    def effective(self):
        with torch.autocast("cuda", enabled=False):
            A, B = self.A, self.B
            if self.Uk is not None:
                B = B - self.Uk @ (self.Uk.T @ B)
                A = A - (A @ self.Vk) @ self.Vk.T
        return A, B

    def forward(self, x):
        A, B = self.effective()
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), A), B)


def add_adapters(model, r, alpha, dropout=0.05):
    mods = {}
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if cname in C.TARGETS and isinstance(child, nn.Linear):
                w = LoRALinear(child, r, alpha, dropout)
                setattr(mod, cname, w)
                mods[f"{name}.{cname}"] = w
    return mods


def attach_bases(mods, path, k):
    """Leading singular bases of W0, computed once in FP32 and cached on disk."""
    if os.path.exists(path):
        bases = torch.load(path, map_location="cpu")
        pre = bases.pop("_seconds")
    else:
        t0, bases = time.time(), {}
        for n, m in mods.items():
            U, S, Vh = torch.linalg.svd(m.base.weight.float(), full_matrices=False)
            bases[n] = (U[:, :k].contiguous().cpu(), Vh[:k].T.contiguous().cpu())
        pre = time.time() - t0
        C.atomic_save({**bases, "_seconds": pre}, path)
    for n, m in mods.items():
        m.Uk, m.Vk = (t.to(m.A.device) for t in bases[n])
    return pre, sum(u.numel() + v.numel() for u, v in bases.values()) * 4


# ----------------------------------------------------------------------------- helpers
def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"]); torch.cuda.set_rng_state_all(s["cuda"])


def lr_lambda(step):
    if step < C.WARMUP:
        return (step + 1) / C.WARMUP
    p = (step - C.WARMUP) / max(1, C.TOTAL_UPDATES - C.WARMUP)
    return 0.5 * (1 + math.cos(math.pi * p))


def export_full(named_fp32, path, tok, p0=None):
    """Save an HF fp16 model dir from a {name: tensor} dict; returns displacement ||theta-theta0||_2."""
    from transformers import AutoConfig
    sd = {k: v.detach().half().cpu() for k, v in named_fp32.items()}
    cfg = AutoConfig.from_pretrained(C.MODEL)
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
    m.load_state_dict({k: v for k, v in sd.items()}, strict=False)
    m.tie_weights()
    m.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    disp = None
    if p0 is not None:
        disp = math.sqrt(sum(((v.detach().float().cpu() - p0[k].float().cpu()) ** 2).sum().item()
                             for k, v in named_fp32.items()))
    return disp


def _ce_chunk(head, h, y):
    with torch.autocast("cuda", dtype=torch.float16):
        logits = head(h)
    return F.cross_entropy(logits.float(), y, reduction="sum")


def chunked_ce(head, h, y, rows=512):
    """Summed CE over response positions only, lm_head+CE recomputed per chunk in backward.
    Same math as full-vocab logits for every position, but never holds (B, L, V) fp16+fp32 copies
    (that peak OOMs full FT on a 15 GB T4)."""
    from torch.utils.checkpoint import checkpoint
    return sum(checkpoint(_ce_chunk, head, h[i:i + rows], y[i:i + rows], use_reentrant=False)
               for i in range(0, h.shape[0], rows))


def export_adapter(mods, run_dir):
    """Export the effective (projected, for OPLoRA) factors used for merging and inference."""
    factors = {}
    for n, m in mods.items():
        with torch.no_grad():
            A, B = m.effective()
        factors[n] = (A.detach().cpu(), B.detach().cpu(), m.scale)
    C.atomic_save(factors, os.path.join(run_dir, "adapters_effective.pt"))
    return factors


def merge_into_fresh(factors, path, tok):
    model = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.float16).cuda().eval()
    with torch.no_grad():
        for n, (A, B, s) in factors.items():
            lin = model.get_submodule(n)
            lin.weight.copy_((lin.weight.float() + s * (B.cuda() @ A.cuda())).half())
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    return model


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(C.METHODS))
    ap.add_argument("--root", default="outputs", help="persistent dir (put on Drive/Kaggle output)")
    ap.add_argument("--save_every", type=int, default=50, help="rolling checkpoint interval (updates)")
    ap.add_argument("--light_milestones", action="store_true",
                    help="milestone ckpts = fp16 weights only, ckpt_latest removed after DONE (fits free Kaggle 20 GB)")
    ap.add_argument("--max_updates", type=int, default=C.TOTAL_UPDATES, help="debug only")
    args = ap.parse_args()
    cfg = C.METHODS[args.method]
    run_dir = os.path.join(args.root, "runs", args.method)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "DONE.json")):
        print(f"[{args.method}] already finished, nothing to do"); return
    signal.signal(signal.SIGTERM, _on_signal); signal.signal(signal.SIGINT, _on_signal)

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED); torch.cuda.manual_seed_all(C.SEED)
    tok = AutoTokenizer.from_pretrained(C.MODEL)
    data = C.build_dataset(tok, os.path.join(args.root, "train_tokenized.pt"))["examples"]
    adapter = cfg["r"] is not None

    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL, torch_dtype=torch.float16 if adapter else torch.float32).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    info = dict(method=args.method, **cfg)
    mods = {}
    if adapter:
        for p in model.parameters():
            p.requires_grad_(False)
        mods = add_adapters(model, cfg["r"], cfg["alpha"])
        if cfg["oplora"]:
            info["basis_seconds"], info["basis_bytes"] = attach_bases(
                mods, os.path.join(args.root, "oplora_bases_k128.pt"), C.PROT_K)
    model.train()

    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    params = list(named.values())
    info["trainable_params"] = sum(p.numel() for p in params)
    # theta_0 (kept for L2-SP on GPU; for plain full FT on CPU, only used for displacement)
    p0 = None
    if not adapter:
        p0 = {n: p.detach().clone() for n, p in named.items()}
        if not cfg["lam"]:
            p0 = {n: v.cpu() for n, v in p0.items()}

    decay = [p for n, p in named.items() if p.ndim >= 2 and not n.endswith(".bias")]
    nodecay = [p for n, p in named.items() if not (p.ndim >= 2 and not n.endswith(".bias"))]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.01), dict(params=nodecay, weight_decay=0.0)],
                            lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------ resume
    step, train_time, skipped = 0, 0.0, []
    latest = os.path.join(run_dir, "ckpt_latest.pt")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cuda", weights_only=False)
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(ck["params"][n])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        r = ck["rng_cpu"]
        r["torch"] = r["torch"].cpu()
        r["cuda"] = [t.cpu() for t in r["cuda"]]
        set_rng_state(r)
        step, train_time, skipped = ck["step"], ck["train_time"], ck["skipped"]
        info.update(ck.get("info_extra", {}))
        del ck
        torch.cuda.empty_cache()
        print(f"[{args.method}] RESUMED from update {step}/{C.TOTAL_UPDATES}", flush=True)
        if os.path.exists(metrics_path):  # drop lines logged after the checkpoint
            with open(metrics_path) as f:
                keep = [l for l in f if int(json.loads(l)["step"]) <= step]
            with open(metrics_path, "w") as f:
                f.writelines(keep)
    else:
        print(f"[{args.method}] starting fresh", flush=True)
        torch.cuda.reset_peak_memory_stats()
    peak_prev = info.get("peak_mem_gb", 0.0)

    def save_ckpt(step_now):
        t0 = time.time()
        st = dict(params={n: p.detach() for n, p in named.items()}, opt=opt.state_dict(),
                  sched=sched.state_dict(), scaler=scaler.state_dict(), rng_cpu=rng_state(),
                  step=step_now, train_time=train_time, skipped=skipped,
                  info_extra=dict(peak_mem_gb=max(peak_prev, torch.cuda.max_memory_allocated() / 2**30)))
        C.atomic_save(st, latest)
        if step_now in C.MILESTONES:
            if args.light_milestones:
                C.atomic_save(dict(params={n: p.detach().half().cpu() for n, p in named.items()}, step=step_now),
                              os.path.join(run_dir, f"ckpt_step{step_now}.pt"))
            else:
                shutil.copyfile(latest, os.path.join(run_dir, f"ckpt_step{step_now}.pt"))
        print(f"  checkpoint saved @ {step_now} ({time.time() - t0:.1f}s)", flush=True)

    def export_mid_if_needed():
        d = os.path.join(run_dir, "export_step625")
        if args.method == "full" and not os.path.exists(os.path.join(d, "OK")):
            src = os.path.join(run_dir, "ckpt_step625.pt")
            if args.light_milestones and step == 625:  # live fp32 weights (the light milestone file is fp16)
                disp = export_full(named, d, tok, p0)
                C.atomic_json(dict(displacement_l2=disp, step=625), os.path.join(d, "OK"))
            elif os.path.exists(src) and not args.light_milestones:
                ck = torch.load(src, map_location="cpu", weights_only=False)
                disp = export_full(ck["params"], d, tok, p0)
                C.atomic_json(dict(displacement_l2=disp, step=625), os.path.join(d, "OK"))
                del ck

    mfile = open(metrics_path, "a")
    pad = tok.pad_token_id
    end = min(args.max_updates, C.TOTAL_UPDATES)
    while step < end:
        t0 = time.time()
        chunk = data[step * C.BATCH:(step + 1) * C.BATCH]
        micro = [C.collate(chunk[i:i + C.MICRO_BS], pad) for i in range(0, C.BATCH, C.MICRO_BS)]
        ntok = sum((lb[:, 1:] != -100).sum().item() for _, _, lb in micro)
        sft = 0.0
        for ids, att, lb in micro:
            ids, att, lb = ids.cuda(), att.cuda(), lb.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
            tgt = lb[:, 1:]
            m = tgt != -100
            loss = chunked_ce(model.lm_head, h[:, :-1][m], tgt[m]) / ntok
            scaler.scale(loss).backward()
            sft += loss.item()
        scaler.unscale_(opt)
        omega = lam_omega = None
        if cfg["lam"]:
            with torch.no_grad():
                omega = 0.0
                for n, p in named.items():
                    d = p - p0[n]
                    omega += 0.5 * d.pow(2).sum().item()
                    p.grad.add_(d, alpha=cfg["lam"])
                lam_omega = cfg["lam"] * omega
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        lr_now = sched.get_last_lr()[0]
        prev_scale = scaler.get_scale()
        scaler.step(opt); scaler.update()
        was_skipped = scaler.get_scale() < prev_scale
        sched.step()
        step += 1
        if was_skipped:
            skipped.append(step)
        train_time += time.time() - t0
        rec = dict(step=step, sft_loss=sft, lr=lr_now, grad_norm=gn, skipped=was_skipped,
                   scale=scaler.get_scale(), omega=omega, lam_omega=lam_omega,
                   total_loss=sft + (lam_omega or 0.0), train_time=train_time)
        mfile.write(json.dumps(rec) + "\n"); mfile.flush()
        if step % 10 == 0 or step == 1:
            print(f"[{args.method}] {step}/{C.TOTAL_UPDATES} sft={sft:.4f} lr={lr_now:.2e} gn={gn:.2f} "
                  f"{train_time / 60:.1f}min", flush=True)
        if step % args.save_every == 0 or step in C.MILESTONES or STOP or step == end:
            save_ckpt(step)
            if step == 625:
                export_mid_if_needed()
        if STOP:
            print("exiting for resume"); sys.exit(0)
    mfile.close()
    if step < C.TOTAL_UPDATES:
        print("max_updates reached (debug) - not exporting"); return
    export_mid_if_needed()

    # ------------------------------------------------------------ final export
    info["peak_mem_gb"] = max(peak_prev, torch.cuda.max_memory_allocated() / 2**30)
    info["train_time_s"], info["skipped_updates"] = train_time, skipped
    out = os.path.join(run_dir, "export_final")
    t0 = time.time()
    if not adapter:
        info["displacement_l2"] = export_full(named, out, tok, p0)
    else:
        factors = export_adapter(mods, run_dir)
        # merged-vs-unmerged logit check on a small batch
        ids, att, _ = C.collate(data[:2], pad)
        ids, att = ids.cuda(), att.cuda()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ref = model(input_ids=ids, attention_mask=att).logits.float()
        del opt, model
        torch.cuda.empty_cache()
        merged = merge_into_fresh(factors, out, tok)
        with torch.no_grad():
            got = merged(input_ids=ids, attention_mask=att).logits.float()
        mask = att.bool()
        info["merge_max_abs_logit_diff"] = (ref - got)[mask].abs().max().item()
        info["adapter_file_bytes"] = os.path.getsize(os.path.join(run_dir, "adapters_effective.pt"))
    info["export_seconds"] = time.time() - t0
    info["export_dir_bytes"] = C.dir_size(out)
    info["ckpt_latest_bytes"] = os.path.getsize(latest)
    C.atomic_json(info, os.path.join(run_dir, "DONE.json"))
    if args.light_milestones:
        os.remove(latest)
    print(f"[{args.method}] DONE", info, flush=True)


if __name__ == "__main__":
    main()

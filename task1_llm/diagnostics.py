"""Task 1 diagnostics + qualitative study on Kaggle (reads the weights/adapters saved by the training kernels).
Outputs (/kaggle/working/diag): weights.json, activations.json, qualitative.json, probe_set.json"""
import glob, json, math, os, re, sys, time, traceback
import numpy as np, torch, yaml
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
OUT = "/kaggle/working/diag"; os.makedirs(OUT, exist_ok=True)
MODEL = "Qwen/Qwen3-0.6B"; LAYERS = [0, 13, 27]; K = 128
CONDS = ["baseline", "full_mid", "full", "l2sp", "lora4", "lora16", "oplora"]
dev = "cuda"
def find(pat):
    r = sorted(glob.glob("/kaggle/input/**/" + pat, recursive=True))
    return r[0] if r else None
PATHS = {"full_mid": "runs/full/export_step625/model.safetensors", "full": "runs/full/export_final/model.safetensors",
         "l2sp": "runs/l2sp/export_final/model.safetensors", "lora4": "runs/lora4/export_final/model.safetensors",
         "lora16": "runs/lora16/export_final/model.safetensors", "oplora": "runs/oplora/export_final/model.safetensors"}
ADAPT = {"lora4": "runs/lora4/adapters_effective.pt", "lora16": "runs/lora16/adapters_effective.pt", "oplora": "runs/oplora/adapters_effective.pt"}
FOUND = {c: find(p) for c, p in PATHS.items()}; print("found:", FOUND, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
key = lambda L: f"model.layers.{L}.mlp.down_proj"
base_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
W0 = {L: base_model.get_submodule(key(L)).weight.detach().float() for L in LAYERS}

# ---------------------------------------------------------------- weight-space diagnostics
def delta_w(cond, L):
    if cond in ADAPT:
        f = torch.load(find(ADAPT[cond]), map_location="cpu")
        A, B, s = f[key(L)]
        return (s * (B.float() @ A.float())).to(dev)
    with safe_open(FOUND[cond], "pt", device="cpu") as sf:
        return sf.get_tensor(key(L) + ".weight").float().to(dev) - W0[L]
def weights():
    res = []
    for L in LAYERS:
        U, S, Vh = torch.linalg.svd(W0[L], full_matrices=False); Uk, Vk = U[:, :K], Vh[:K].T
        for c in CONDS[1:]:
            try:
                d = delta_w(c, L); n2 = (d ** 2).sum()
                res.append(dict(cond=c, layer=L, rho_k=((Uk @ (Uk.T @ d)) ** 2).sum().item() / n2.item(),
                                rho_k_input=((d @ Vk) ** 2).sum().item() / n2.item(),
                                eta=(n2.sqrt() / W0[L].norm()).item(), delta_fro=n2.sqrt().item(), w0_fro=W0[L].norm().item(),
                                top_energy_share_of_W0=(S[:K] ** 2).sum().item() / (S ** 2).sum().item()))
            except Exception as e: print("weights fail", c, L, e, flush=True); traceback.print_exc()
        del U, S, Vh
    json.dump(res, open(f"{OUT}/weights.json", "w"), indent=1); print("weights done", flush=True); return res

# ---------------------------------------------------------------- activation diagnostics
def activations():
    probe = json.load(open("/kaggle/working/probe_set.json")); json.dump(probe, open(f"{OUT}/probe_set.json", "w"))
    cache = {"math": {L: [] for L in LAYERS}, "nonmath": {L: [] for L in LAYERS}}; hooks = []
    cur = {}
    for L in LAYERS:
        hooks.append(base_model.get_submodule(key(L)).register_forward_pre_hook(lambda m, a, L=L: cur.__setitem__(L, a[0][0, -1].detach().float().clone())))
    for grp in ["math", "nonmath"]:
        for ex in probe[grp]:
            ids = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)  # batch 1: last token = last non-padding token
            with torch.no_grad(): base_model(input_ids=ids)
            for L in LAYERS: cache[grp][L].append(cur[L].clone())
    for h in hooks: h.remove()
    res = []
    for c in CONDS[1:]:
        for L in LAYERS:
            d = delta_w(c, L)
            for grp in ["math", "nonmath"]:
                z = torch.stack(cache[grp][L])                       # [32, 3072] FP32
                R = ((z @ d.T).norm(dim=1) / ((z @ W0[L].T).norm(dim=1) + 1e-8)).cpu().numpy()
                res.append(dict(cond=c, layer=L, group=grp, mean=float(R.mean()), median=float(np.median(R)), std=float(R.std()), values=R.tolist()))
    json.dump(res, open(f"{OUT}/activations.json", "w"), indent=1); print("activations done", flush=True)

# ---------------------------------------------------------------- qualitative
def qual():
    qs = yaml.safe_load(open("/kaggle/working/first_plot_questions.yaml"))
    items = []
    for q in qs:
        fmt = "json" if q["id"].endswith("_json") else "template" if q["id"].endswith("_template") else "plain"
        base = re.sub(r"_(json|template)$", "", q["id"])
        msgs = ([{"role": "system", "content": q["system"]}] if q.get("system") else []) + [{"role": "user", "content": q["paraphrases"][0]}]
        items.append(dict(id=q["id"], question=base, format=fmt, messages=msgs))
    tok.padding_side = "left"; res = []
    for c in CONDS:
        path = MODEL if c == "baseline" else os.path.dirname(FOUND[c])
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).to(dev).eval()
        for i in range(0, len(items), 8):
            b = items[i:i + 8]
            texts = [tok.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in b]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=512, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            for x, o in zip(b, out):
                gen = o[enc.input_ids.shape[1]:]; n = int((gen != tok.pad_token_id).sum())
                eos = bool(((gen == tok.convert_tokens_to_ids("<|im_end|>")) | (gen == tok.eos_token_id)).any())
                res.append(dict(cond=c, id=x["id"], question=x["question"], format=x["format"], response=tok.decode(gen, skip_special_tokens=True),
                                new_tokens=n, truncated=(not eos) and n >= 512))
        del m; torch.cuda.empty_cache(); print("qual done", c, flush=True)
        json.dump(res, open(f"{OUT}/qualitative.json", "w"), indent=1)

for fn in (weights, activations, qual):
    t0 = time.time()
    try: fn()
    except Exception: traceback.print_exc()
    print(fn.__name__, "seconds", round(time.time() - t0), flush=True)

"""SigLIP features -> MLP connector -> Qwen3-0.6B (frozen, optional LoRA). Batching, losses, NLL, greedy decoding."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as C

IMG = -1  # placeholder id for a visual-token position inside a prompt


class Features:
    """FP16 SigLIP cache (memory-mapped); key = 'train2014/<image_id>' or 'val2014/<image_id>'."""

    def __init__(self, data):
        self.arr = np.load(f"{data}/features.npy", mmap_mode="r")
        self.index = C.load_json(f"{data}/feature_index.json")

    def get(self, keys):
        return torch.from_numpy(np.stack([self.arr[self.index[k]] for k in keys]))


def key(ex):
    return f"{ex['split']}/{ex['image_id']}"


class Connector(nn.Module):
    """Linear 768->1024, GELU, Linear 1024->1024. pool=2 averages each 2x2 block of the 14x14 grid first (49 tokens)."""

    def __init__(self, pool=1):
        super().__init__()
        self.pool = pool
        self.fc1, self.act, self.fc2 = nn.Linear(C.D_VIS, C.D_LM), nn.GELU(), nn.Linear(C.D_LM, C.D_LM)
        self.register_buffer("scale", torch.tensor(1.0))  # fixed output-magnitude calibration (not trained)

    @property
    def n_tokens(self):
        return (C.GRID // self.pool) ** 2

    def forward(self, f):
        f = f.float()
        if self.pool > 1:
            B = f.shape[0]
            g = f.view(B, C.GRID, C.GRID, C.D_VIS).permute(0, 3, 1, 2)       # keep 14 x 14 row-major order
            f = F.avg_pool2d(g, self.pool).permute(0, 2, 3, 1).reshape(B, -1, C.D_VIS)
        return self.fc2(self.act(self.fc1(f))) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class VLM(nn.Module):
    def __init__(self, pool=1, device=None):
        super().__init__()
        self.dev = device or C.dev_type()
        self.tok = AutoTokenizer.from_pretrained(C.LM)
        lm_dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(C.LM, dtype=lm_dtype).to(self.dev)
        self.lm.config.use_cache = False
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.connector = Connector(pool).to(self.dev)
        self.lora = {}
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id
        tmpl = self.tok.apply_chat_template(
            [{"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>\nQQQ"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pre, rest = tmpl.split("<|image_pad|>")
        mid, post = rest.split("QQQ")
        enc = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]
        self.pre, self.mid, self.post = enc(pre), enc(mid), enc(post)
        self.template = tmpl

    # ------------------------------------------------------------------ parameters
    def add_lora(self):
        for name, mod in list(self.lm.named_modules()):
            for cname, child in list(mod.named_children()):
                if cname in C.LORA["targets"] and isinstance(child, nn.Linear):
                    w = LoRALinear(child, C.LORA["r"], C.LORA["alpha"], C.LORA["dropout"])
                    setattr(mod, cname, w)
                    self.lora[f"{name}.{cname}"] = w

    def trainable(self):
        return [p for p in self.connector.parameters()] + [p for m in self.lora.values() for p in (m.A, m.B)]

    def adapter_state(self):
        return dict(connector={k: v.detach().cpu() for k, v in self.connector.state_dict().items()},
                    lora={n: (m.A.detach().cpu(), m.B.detach().cpu()) for n, m in self.lora.items()})

    def load_adapter_state(self, st):
        self.connector.load_state_dict(st["connector"])
        if st["lora"] and not self.lora:
            self.add_lora()
        with torch.no_grad():
            for n, (A, B) in st["lora"].items():
                self.lora[n].A.copy_(A); self.lora[n].B.copy_(B)

    # ------------------------------------------------------------------ calibration (training data only)
    @torch.no_grad()
    def calibrate(self, feats, text_ids):
        """Rule: scale = mean per-token L2 norm of Qwen input embeddings over training text tokens /
        mean per-token L2 norm of the freshly initialized connector output on training images."""
        e = self.lm.get_input_embeddings()(torch.tensor(text_ids, device=self.dev)).float()
        self.connector.scale.fill_(1.0)
        v = self.connector(feats.to(self.dev))
        t_norm, v_norm = e.norm(dim=-1).mean().item(), v.norm(dim=-1).mean().item()
        self.connector.scale.fill_(t_norm / v_norm)
        after = self.connector(feats.to(self.dev)).norm(dim=-1).mean().item()
        return dict(rule="scale = mean||text embedding|| / mean||connector(F)|| at init (training data only)",
                    text_tokens=len(text_ids), images=feats.shape[0], text_embed_norm=t_norm,
                    visual_norm_before=v_norm, scale=t_norm / v_norm, visual_norm_after=after)

    # ------------------------------------------------------------------ encoding
    def user_text(self, ex):
        return C.CAPTION_PROMPT if ex["suite"] == "caption" else ex["question"] + C.ANSWER_SUFFIX

    def encode(self, ex, image=True, target=True, label_end=True):
        q = self.tok(self.user_text(ex), add_special_tokens=False)["input_ids"][:C.Q_CAP]
        ids = self.pre + ([IMG] * self.connector.n_tokens if image else []) + self.mid + q + self.post
        labels = [-100] * len(ids)
        if target:
            cap = C.CAP_CAP if ex["suite"] == "caption" else C.ANS_CAP
            t = self.tok(ex["target"], add_special_tokens=False)["input_ids"][:cap]
            ids = ids + t + [self.im_end]
            labels = labels + t + ([self.im_end] if label_end else [-100])
        return ids, labels

    def batch(self, exs, keys, feats, target=True, label_end=True, left_pad=False):
        """keys[i] is the image to show example i (None = no image: visual tokens omitted, masks/positions rebuilt)."""
        enc = [self.encode(e, image=k is not None, target=target, label_end=label_end) for e, k in zip(exs, keys)]
        L = max(len(i) for i, _ in enc)
        B = len(exs)
        ids = torch.full((B, L), self.pad, dtype=torch.long)
        lab = torch.full((B, L), -100, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for b, (i, l) in enumerate(enc):
            s = L - len(i) if left_pad else 0
            ids[b, s:s + len(i)] = torch.tensor(i)
            lab[b, s:s + len(i)] = torch.tensor(l)
            att[b, s:s + len(i)] = 1
        img_mask = ids == IMG
        ids[img_mask] = self.pad
        ids, lab, att, img_mask = (t.to(self.dev) for t in (ids, lab, att, img_mask))
        emb = self.lm.get_input_embeddings()(ids)
        rows = [k for k in keys if k is not None]
        if rows:
            vis = self.connector(feats.get(rows).to(self.dev))                       # (n_img, n_tok, 1024)
            emb = emb.masked_scatter(img_mask.unsqueeze(-1), vis.to(emb.dtype))     # in row-major order
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        return emb, att, pos, lab

    # ------------------------------------------------------------------ losses
    def token_losses(self, emb, att, pos, lab):
        """Per-example mean CE over supervised tokens -> (B,) ; only supervised positions go through lm_head."""
        h = self.lm.model(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state
        y = lab[:, 1:]
        m = y != -100
        rows = torch.arange(y.shape[0], device=y.device)[:, None].expand_as(y)[m]
        logits = self.lm.lm_head(h[:, :-1][m]).float()
        tl = F.cross_entropy(logits, y[m], reduction="none")
        n = m.sum(1).clamp(min=1)
        return torch.zeros(y.shape[0], device=y.device).index_add_(0, rows, tl) / n

    @torch.no_grad()
    def nll(self, exs, keys, feats, label_end=False, bs=C.GEN_BS):
        """Teacher-forced reference NLL per example (token mean). label_end=False excludes the termination token."""
        out = []
        for s in range(0, len(exs), bs):
            with torch.autocast(self.dev, dtype=C.amp_dtype()):
                b = self.batch(exs[s:s + bs], keys[s:s + bs], feats, target=True, label_end=label_end)
                out += self.token_losses(*b).tolist()
        return out

    @torch.no_grad()
    def generate(self, exs, keys, feats, bs=C.GEN_BS, max_new=C.GEN_TOKENS):
        """Greedy decoding (thinking disabled by the template), left padding, KV cache, stop at <|im_end|>."""
        outs = []
        for s in range(0, len(exs), bs):
            with torch.autocast(self.dev, dtype=C.amp_dtype()):
                emb, att, pos, _ = self.batch(exs[s:s + bs], keys[s:s + bs], feats, target=False, left_pad=True)
                o = self.lm(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
                past, nxt = o.past_key_values, o.logits[:, -1].argmax(-1)
                toks, done = [], torch.zeros_like(nxt, dtype=torch.bool)
                for step in range(max_new):
                    toks.append(nxt)
                    done |= nxt == self.im_end
                    if done.all() or step == max_new - 1:
                        break
                    att = torch.cat([att, torch.ones_like(att[:, :1])], 1)
                    pos = pos[:, -1:] + 1
                    o = self.lm(input_ids=nxt[:, None], attention_mask=att, position_ids=pos,
                                past_key_values=past, use_cache=True)
                    past, nxt = o.past_key_values, o.logits[:, -1].argmax(-1)
            T = torch.stack(toks, 1).tolist()
            for row in T:
                row = row[:row.index(self.im_end)] if self.im_end in row else row
                outs.append(self.tok.decode(row, skip_special_tokens=True).strip())
        return outs

"""Data selection, image download and SigLIP feature caching.

    python prep.py --data DATA select      # deterministic selection (seed 6304) -> DATA/selection.json
    python prep.py --data DATA fetch       # download only the selected COCO images
    python prep.py --data DATA cache       # SigLIP features, FP16, 196 x 768 per image -> DATA/features.npy

All evaluation images come from val2014 and every val image is used by at most one development/final-test
example set (captions, grounding, VQAv2), so the sets are image-disjoint from each other and from training.
"""
import argparse
import collections
import concurrent.futures as cf
import os
import random
import subprocess
import time

import numpy as np

import common as C

RAW_URLS = [
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    *[f"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_{k}_mscoco.zip"
      for k in ["Questions_Train", "Annotations_Train", "Complementary_Pairs_Train",
                "Questions_Val", "Annotations_Val", "Complementary_Pairs_Val"]],
]
VQA_TYPES = {"yes/no": "yes/no", "number": "number", "other": "other"}
VQA_COUNTS = {"train": {"yes/no": 640, "number": 256, "other": 640},
              "dev": {"yes/no": 12, "number": 8, "other": 12},
              "test": {"yes/no": 36, "number": 24, "other": 36}}
GROUND_COUNTS = {"dev": {"presence": 12, "counting": 12}, "test": {"presence": 32, "counting": 32}}
N_CAP_TRAIN, CAPS_PER_TRAIN, N_CAP_DEV = 6000, 3, 96
N_GROUND_POOL_IMAGES = 10000
PRESENCE_PAIRS_PER_CAT, COUNTING_PAIRS_PER_CAT = 250, 150
PRESENCE_MIN_AREA, COUNT_MIN_AREA = 0.01, 0.005   # fraction of image area
IRREGULAR = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis",
             "scissors": "scissors", "broccoli": "broccoli", "skateboard": "skateboards"}


def rng(tag):
    return random.Random(f"{C.SEED}-{tag}")


def download_raw(raw):
    os.makedirs(raw, exist_ok=True)
    for u in RAW_URLS:
        z = os.path.join(raw, os.path.basename(u))
        if not os.path.exists(z):
            subprocess.run(["curl", "-sSL", "-o", z + ".part", u], check=True)
            os.replace(z + ".part", z)
        subprocess.run(["unzip", "-oq", z, "-d", raw], check=True)


def plural(name):
    if name in IRREGULAR:
        return IRREGULAR[name]
    head, _, last = name.rpartition(" ")
    if last in IRREGULAR:
        p = IRREGULAR[last]
    elif last.endswith(("s", "sh", "ch", "x")):
        p = last + "es"
    else:
        p = last + "s"
    return (head + " " if head else "") + p


def article(name):
    return ("an " if name[0] in "aeiou" else "a ") + name


# ----------------------------------------------------------------------------- VQAv2
def consensus(ann):
    """(normalized consensus answer, agreement count, original target string from the consensus group)."""
    raw = [a["answer"].strip() for a in ann["answers"]]
    norm = [C.normalize_answer(a) for a in raw]
    cons, cnt = collections.Counter(norm).most_common(1)[0]
    target = collections.Counter(r for r, n in zip(raw, norm) if n == cons).most_common(1)[0][0]
    return cons, cnt, target, raw, norm


def eligible_vqa_pairs(raw, split):
    tag = "train2014" if split == "train" else "val2014"
    qs = {q["question_id"]: q for q in C.load_json(f"{raw}/v2_OpenEnded_mscoco_{tag}_questions.json")["questions"]}
    anns = {a["question_id"]: a for a in C.load_json(f"{raw}/v2_mscoco_{tag}_annotations.json")["annotations"]}
    pairs = C.load_json(f"{raw}/v2_mscoco_{tag}_complementary_pairs.json")
    out = []
    for qa, qb in sorted(pairs):
        A, B = anns[qa], anns[qb]
        if A["answer_type"] != B["answer_type"] or A["image_id"] == B["image_id"]:
            continue
        ca, na, ta, ra, nra = consensus(A)
        cb, nb, tb, rb, nrb = consensus(B)
        if na < 8 or nb < 8 or ca == cb:
            continue
        pid = f"vqa-{qa}-{qb}"
        ex = []
        for q, a, c, t, r, nr in [(qa, A, ca, ta, ra, nra), (qb, B, cb, tb, rb, nrb)]:
            ex.append(dict(qid=f"vqa-{q}", pair_id=pid, split=tag, image_id=a["image_id"],
                           question=qs[q]["question"], target=t, canonical=c, answers=r, answers_norm=nr,
                           type=VQA_TYPES[a["answer_type"]], suite="vqa"))
        out.append(ex)
    return out


# ----------------------------------------------------------------------------- COCO instances -> grounding
class Instances:
    def __init__(self, raw, tag):
        d = C.load_json(f"{raw}/annotations/instances_{tag}.json")
        self.tag = tag
        self.cats = {c["id"]: c for c in d["categories"]}
        self.imgs = {i["id"]: i for i in d["images"]}
        self.by_img = collections.defaultdict(list)
        for a in d["annotations"]:
            self.by_img[a["image_id"]].append(a)

    def presence_sets(self, image_ids):
        """category -> (positives, negatives); negatives have no annotation in the category's supercategory."""
        out = {}
        for cid, c in sorted(self.cats.items()):
            same_super = {k for k, v in self.cats.items() if v["supercategory"] == c["supercategory"]}
            pos, neg = [], []
            for i in image_ids:
                im, an = self.imgs[i], self.by_img[i]
                area = im["width"] * im["height"]
                if any(a["category_id"] == cid and not a["iscrowd"] and a["area"] >= PRESENCE_MIN_AREA * area
                       for a in an):
                    pos.append(i)
                elif not any(a["category_id"] in same_super for a in an):
                    neg.append(i)
            out[cid] = (pos, neg)
        return out

    def count_buckets(self, image_ids):
        """category -> {count: [images]}, counts 1..4, no crowd and no tiny instances of that category."""
        out = collections.defaultdict(lambda: collections.defaultdict(list))
        for i in image_ids:
            im, an = self.imgs[i], self.by_img[i]
            area = im["width"] * im["height"]
            per = collections.defaultdict(list)
            for a in an:
                per[a["category_id"]].append(a)
            for cid, lst in per.items():
                if any(a["iscrowd"] or a["area"] < COUNT_MIN_AREA * area for a in lst):
                    continue
                if 1 <= len(lst) <= 4:
                    out[cid][len(lst)].append(i)
        return out


def presence_example(ins, cid, img, ans, pid, k):
    name = ins.cats[cid]["name"]
    return dict(qid=f"{pid}-{k}", pair_id=pid, split=ins.tag, image_id=img, question=f"Does this image contain {article(name)}?",
                target=ans, canonical=C.normalize_answer(ans), type="presence", category=name, suite="ground")


def counting_example(ins, cid, img, n, pid, k):
    name = ins.cats[cid]["name"]
    return dict(qid=f"{pid}-{k}", pair_id=pid, split=ins.tag, image_id=img, question=f"How many {plural(name)} can you see?",
                target=str(n), canonical=C.normalize_answer(str(n)), type="counting", category=name, suite="ground")


def make_presence_pair(ins, r, cid, pos, neg, pid):
    a = [presence_example(ins, cid, pos, "yes", pid, 0), presence_example(ins, cid, neg, "no", pid, 1)]
    if r.random() < 0.5:  # random order within the pair
        a = a[::-1]
    for k, e in enumerate(a):
        e["qid"] = f"{pid}-{k}"
    return a


def train_pools(ins, image_ids):
    r = rng("ground-pools")
    pres, cnt, seen = [], [], set()
    for cid, (pos, neg) in ins.presence_sets(image_ids).items():
        if len(pos) < 20 or len(neg) < 20:
            continue
        for _ in range(PRESENCE_PAIRS_PER_CAT):
            p, n = r.choice(pos), r.choice(neg)
            if (cid, p, n) in seen:
                continue
            seen.add((cid, p, n))
            pres.append(make_presence_pair(ins, r, cid, p, n, f"pres-tr-{cid}-{p}-{n}"))
    for cid, buckets in sorted(ins.count_buckets(image_ids).items()):
        ks = sorted(k for k, v in buckets.items() if len(v) >= 3)
        if len(ks) < 2:
            continue
        for _ in range(COUNTING_PAIRS_PER_CAT):
            ka, kb = r.sample(ks, 2)
            a, b = r.choice(buckets[ka]), r.choice(buckets[kb])
            if (cid, a, b) in seen:
                continue
            seen.add((cid, a, b))
            pid = f"cnt-tr-{cid}-{a}-{b}"
            cnt.append([counting_example(ins, cid, a, ka, pid, 0), counting_example(ins, cid, b, kb, pid, 1)])
    return pres, cnt


def eval_grounding(ins, used, counts, tag):
    """Image-unique presence/counting pairs from val2014, cycling over categories for diversity."""
    r = rng(f"ground-{tag}")
    free = sorted(i for i in ins.imgs if i not in used)
    pres_sets, buckets = ins.presence_sets(free), ins.count_buckets(free)
    out = []
    cats = [c for c, (p, n) in pres_sets.items() if len(p) >= 5 and len(n) >= 5]
    r.shuffle(cats)
    k = 0
    while sum(p[0]["type"] == "presence" for p in out) < counts["presence"]:
        cid = cats[k % len(cats)]; k += 1
        pos = [i for i in pres_sets[cid][0] if i not in used]
        neg = [i for i in pres_sets[cid][1] if i not in used]
        if not pos or not neg:
            continue
        p, n = r.choice(pos), r.choice(neg)
        used.update([p, n])
        out.append(make_presence_pair(ins, r, cid, p, n, f"pres-{tag}-{cid}-{p}-{n}"))
    ccats = sorted(c for c, b in buckets.items() if sum(len(v) >= 2 for v in b.values()) >= 2)
    r.shuffle(ccats)
    k = 0
    while sum(p[0]["type"] == "counting" for p in out) < counts["counting"]:
        cid = ccats[k % len(ccats)]; k += 1
        avail = {n: [i for i in v if i not in used] for n, v in buckets[cid].items()}
        ks = sorted(n for n, v in avail.items() if v)
        if len(ks) < 2:
            continue
        ka, kb = r.sample(ks, 2)
        a, b = r.choice(avail[ka]), r.choice(avail[kb])
        used.update([a, b])
        pid = f"cnt-{tag}-{cid}-{a}-{b}"
        out.append([counting_example(ins, cid, a, ka, pid, 0), counting_example(ins, cid, b, kb, pid, 1)])
    return out


# ----------------------------------------------------------------------------- selection
def select(data):
    raw = os.path.join(data, "raw")
    download_raw(raw)
    t0 = time.time()
    sel = {}
    used_val = set()

    # natural VQAv2: train from train2014; final test then dev from val2014, image-unique
    tr_cand, va_cand = eligible_vqa_pairs(raw, "train"), eligible_vqa_pairs(raw, "val")
    for split, cand in [("train", tr_cand), ("test", va_cand), ("dev", va_cand)]:
        r = rng(f"vqa-{split}")
        order = list(range(len(cand)))
        r.shuffle(order)
        picked, need = [], dict(VQA_COUNTS[split])
        for i in order:
            p = cand[i]
            t = p[0]["type"]
            if need[t] == 0:
                continue
            imgs = {p[0]["image_id"], p[1]["image_id"]}
            if split != "train" and imgs & used_val:
                continue
            if split != "train":
                used_val |= imgs
            picked.append(p)
            need[t] -= 1
            if not any(need.values()):
                break
        assert not any(need.values()), (split, need)
        sel[f"vqa_{split}"] = picked
        print(f"vqa {split}: {len(picked)} pairs from {len(cand)} eligible", flush=True)

    # annotation-derived grounding
    ins_tr = Instances(raw, "train2014")
    r = rng("ground-pool-images")
    with_anns = sorted(i for i in ins_tr.imgs if ins_tr.by_img[i])
    pool_imgs = sorted(r.sample(with_anns, N_GROUND_POOL_IMAGES))
    sel["presence_train_pool"], sel["counting_train_pool"] = train_pools(ins_tr, pool_imgs)
    del ins_tr
    ins_va = Instances(raw, "val2014")
    for split in ["test", "dev"]:
        sel[f"ground_{split}"] = eval_grounding(ins_va, used_val, GROUND_COUNTS[split], split)
    val_info = ins_va.imgs
    print("grounding pools:", len(sel["presence_train_pool"]), "presence /", len(sel["counting_train_pool"]),
          "counting pairs", flush=True)

    # COCO captions: 6000 train images x 3 captions, 96 dev images x 1 caption
    caps = {}
    for tag in ["train2014", "val2014"]:
        d = C.load_json(f"{raw}/annotations/captions_{tag}.json")
        by = collections.defaultdict(list)
        for a in sorted(d["annotations"], key=lambda a: a["id"]):
            by[a["image_id"]].append(a)
        caps[tag] = (by, {i["id"]: i for i in d["images"]})
    r = rng("captions")
    by, _ = caps["train2014"]
    tr_imgs = r.sample(sorted(i for i in by if len(by[i]) >= CAPS_PER_TRAIN), N_CAP_TRAIN)
    sel["caption_train"] = [dict(qid=f"cap-{a['id']}", image_id=i, split="train2014", target=a["caption"].strip(),
                                 suite="caption") for i in tr_imgs for a in r.sample(by[i], CAPS_PER_TRAIN)]
    by, _ = caps["val2014"]
    free = [i for i in sorted(by) if i not in used_val]
    dev_imgs = r.sample(free, N_CAP_DEV)
    used_val.update(dev_imgs)
    sel["caption_dev"] = [dict(qid=f"cap-{a['id']}", image_id=i, split="val2014", target=a["caption"].strip(),
                               suite="caption") for i in dev_imgs for a in [r.choice(by[i])]]

    # image list (train / val) with file names
    names = {**{("train2014", i): v["file_name"] for i, v in caps["train2014"][1].items()},
             **{("val2014", i): v["file_name"] for i, v in caps["val2014"][1].items()}}
    images = set()
    for k, v in sel.items():
        for item in v:
            for e in (item if isinstance(item, list) else [item]):
                images.add((e["split"], e["image_id"]))
    sel["images"] = sorted([s, i, names[(s, i)]] for s, i in images)
    checks = isolation_checks(sel)
    sel["meta"] = dict(seed=C.SEED, select_seconds=time.time() - t0, checks=checks,
                       counts={k: len(v) for k, v in sel.items() if isinstance(v, list)},
                       rules=dict(presence_min_area=PRESENCE_MIN_AREA, count_min_area=COUNT_MIN_AREA,
                                  presence_negatives="no annotation of any category in the same supercategory",
                                  counting="1-4 non-crowd instances, none smaller than count_min_area",
                                  pool_images=N_GROUND_POOL_IMAGES))
    C.atomic_json(sel, os.path.join(data, "selection.json"))
    ids = {k: ([[e["qid"] for e in p] for p in v] if v and isinstance(v[0], list) else [e["qid"] for e in v])
           for k, v in sel.items() if isinstance(v, list) and k != "images"}
    ids["images"] = sel["images"]
    C.atomic_json(ids, os.path.join(data, "selected_ids.json"))
    print("checks:", checks, flush=True)
    print({k: len(v) for k, v in sel.items() if isinstance(v, list)}, flush=True)


def isolation_checks(sel):
    def imgs(key):
        s = set()
        for item in sel[key]:
            for e in (item if isinstance(item, list) else [item]):
                s.add((e["split"], e["image_id"]))
        return s
    evals = ["vqa_dev", "vqa_test", "ground_dev", "ground_test", "caption_dev"]
    train = ["vqa_train", "presence_train_pool", "counting_train_pool", "caption_train"]
    tr = set().union(*[imgs(k) for k in train])
    out = dict(train_images_all_train2014=all(s == "train2014" for s, _ in tr),
               eval_images_all_val2014=all(s == "val2014" for k in evals for s, _ in imgs(k)),
               eval_train_overlap=len(tr & set().union(*[imgs(k) for k in evals])))
    for i, a in enumerate(evals):
        for b in evals[i + 1:]:
            out[f"overlap_{a}_{b}"] = len(imgs(a) & imgs(b))
    for k in evals[:-1]:  # images unique within each eval suite as well
        n = sum(2 for _ in sel[k])
        out[f"unique_images_{k}"] = len(imgs(k)) == n
    for k in ["vqa_train", "vqa_dev", "vqa_test"]:
        out[f"{k}_pairs_differ_and_agree8"] = all(p[0]["canonical"] != p[1]["canonical"] and
                                                   max(collections.Counter(e["answers_norm"]).values()) >= 8
                                                   for p in sel[k] for e in p)
    return out


# ----------------------------------------------------------------------------- download + cache
def fetch(data, workers=64):
    sel = C.load_json(os.path.join(data, "selection.json"))
    out = os.path.join(data, "images")
    os.makedirs(out, exist_ok=True)
    import requests
    sess = requests.Session()
    sess.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=workers, max_retries=5))

    def get(item):
        split, _, fn = item
        p = os.path.join(out, fn)
        if os.path.exists(p):
            return 0
        for attempt in range(6):
            try:
                r = sess.get(f"http://images.cocodataset.org/{split}/{fn}", timeout=60)
                r.raise_for_status()
                with open(p + ".part", "wb") as f:
                    f.write(r.content)
                os.replace(p + ".part", p)
                return len(r.content)
            except Exception:
                time.sleep(2 ** attempt)
        raise RuntimeError(f"failed {fn}")
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        nbytes = sum(ex.map(get, sel["images"]))
    C.atomic_json(dict(images=len(sel["images"]), bytes=nbytes, seconds=time.time() - t0),
                  os.path.join(data, "fetch_stats.json"))
    print(f"fetched {len(sel['images'])} images, {nbytes / 2**30:.2f} GB, {time.time() - t0:.0f}s", flush=True)


def cache(data, bs=128):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel
    sel = C.load_json(os.path.join(data, "selection.json"))
    items = sel["images"]
    dev = C.dev_type()
    proc = AutoImageProcessor.from_pretrained(C.VISION, use_fast=False)   # resize 224 bicubic, rescale, normalize
    model = SiglipVisionModel.from_pretrained(C.VISION, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    feats = np.lib.format.open_memmap(os.path.join(data, "features.npy.tmp"), mode="w+", dtype=np.float16,
                                      shape=(len(items), C.GRID * C.GRID, C.D_VIS))
    index = {f"{s}/{i}": k for k, (s, i, _) in enumerate(items)}

    def load(it):
        with Image.open(os.path.join(data, "images", it[2])) as im:
            return proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ln_check = None
    with cf.ThreadPoolExecutor(8) as ex, torch.no_grad():
        for s in range(0, len(items), bs):
            px = torch.stack(list(ex.map(load, items[s:s + bs]))).to(dev, model.dtype)
            out = model(pixel_values=px, output_hidden_states=ln_check is None)
            if ln_check is None:  # last_hidden_state = post_layernorm(encoder output), applied exactly once
                post_ln = next(m for n, m in model.named_modules() if n.endswith("post_layernorm"))
                ref = post_ln(out.hidden_states[-1])
                ln_check = (out.last_hidden_state - ref).abs().max().item()
                assert out.last_hidden_state.shape[1:] == (C.GRID * C.GRID, C.D_VIS), out.last_hidden_state.shape
            feats[s:s + bs] = out.last_hidden_state.half().cpu().numpy()
    feats.flush()
    del feats
    os.replace(os.path.join(data, "features.npy.tmp"), os.path.join(data, "features.npy"))
    stats = dict(images=len(items), seconds=time.time() - t0, batch_size=bs, dtype="float16",
                 shape=[len(items), C.GRID * C.GRID, C.D_VIS], post_layernorm_max_abs_diff=ln_check,
                 peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None,
                 bytes=os.path.getsize(os.path.join(data, "features.npy")),
                 vision_params=sum(p.numel() for p in model.parameters()))
    C.atomic_json(index, os.path.join(data, "feature_index.json"))
    C.atomic_json(stats, os.path.join(data, "cache_stats.json"))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()  # vision encoder removed from GPU before any Qwen training
    print("cached", stats, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("cmd", choices=["select", "fetch", "cache"])
    a = ap.parse_args()
    {"select": select, "fetch": fetch, "cache": cache}[a.cmd](a.data)

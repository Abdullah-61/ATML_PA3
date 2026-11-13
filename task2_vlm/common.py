"""Shared constants, VQA answer normalization / scoring and small IO helpers for Task 2 (SigLIP-Qwen VLM)."""
import json
import os
import re

import torch

SEED = 6304
LM = "Qwen/Qwen3-0.6B"
VISION = "google/siglip-base-patch16-224"
D_VIS, D_LM = 768, 1024
GRID = 14                       # 14 x 14 = 196 patch features

MICRO_BS, ACCUM = 8, 2
BATCH = MICRO_BS * ACCUM        # 16 examples per update
WD, CLIP = 0.01, 1.0
WARMUP_FRAC, MIN_LR_FRAC = 0.05, 0.10
Q_CAP, CAP_CAP, ANS_CAP = 80, 64, 16
GEN_TOKENS, GEN_BS = 16, 8

# stage -> (updates, connector peak lr, lora peak lr, draw probabilities)
STAGES = [
    dict(name="caption_alignment", updates=2250, lr_conn=1e-3, lr_lora=None, mix=None),
    dict(name="visual_instruction", updates=1500, lr_conn=1e-4, lr_lora=2e-5,
         mix={"presence": 0.50, "counting": 0.25, "caption": 0.25}),
    dict(name="natural_vqa", updates=750, lr_conn=5e-5, lr_lora=1e-5,
         mix={"vqa": 0.70, "presence": 0.10, "counting": 0.10, "caption": 0.10}),
]
STAGE_ENDS = [2250, 3750, 4500]
TOTAL_UPDATES = 4500
CKPT_EVERY = 250
DEV_EVAL_AT = sorted({0, *range(500, TOTAL_UPDATES + 1, 500), *STAGE_ENDS})
if os.environ.get("VLM_SMOKE"):  # tiny end-to-end test of the full flow (stages, LoRA switch, resume, final eval)
    for _s, _n in zip(STAGES, [4, 3, 2]):
        _s["updates"] = _n
    STAGE_ENDS, TOTAL_UPDATES, CKPT_EVERY = [4, 7, 9], 9, 2
    DEV_EVAL_AT = [0, 4, 7, 9]
LORA = dict(r=16, alpha=32, dropout=0.05, targets=("q_proj", "k_proj", "v_proj", "o_proj"))

CAPTION_PROMPT = "Write a short caption for this image."
ANSWER_SUFFIX = "\nAnswer with a single word or number."

# ----------------------------------------------------------------------------- VQA answer normalization
# Port of the official VQA evaluation normalization (vqaEval.py): punctuation, digits, articles, contractions.
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
              "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def normalize_answer(ans):
    """Same normalization for targets, references and predictions everywhere."""
    t = ans.replace("\n", " ").replace("\t", " ").strip()
    out = t
    for p in PUNCT:
        if (p + " " in t or " " + p in t) or COMMA_STRIP.search(t) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = PERIOD_STRIP.sub("", out)
    words = []
    for w in out.lower().split():
        w = MANUAL_MAP.get(w, w)
        if w not in ARTICLES:
            words.append(CONTRACTIONS.get(w, w))
    return " ".join(words)


def vqa_soft_acc(pred_norm, refs_norm):
    """Local VQA consensus accuracy: mean over the 10 leave-one-out subsets of min(1, matches/3)."""
    n = len(refs_norm)
    return sum(min(1.0, sum(pred_norm == refs_norm[j] for j in range(n) if j != i) / 3.0) for i in range(n)) / n


# ----------------------------------------------------------------------------- IO
def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dev_type():
    return "cuda" if torch.cuda.is_available() else "cpu"


def amp_dtype():
    # FP16 autocast on GPU (as specified); bf16 only for CPU smoke tests
    return torch.float16 if torch.cuda.is_available() else torch.bfloat16

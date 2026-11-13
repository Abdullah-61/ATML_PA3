"""Answer normalization, answer groups and correctness (scorer v1, fixed before final evaluation).

MATH: math-verify equivalence. Distinct extracted strings of a question are clustered greedily in rollout order:
a string joins the first existing group whose representative it is equivalent to (verify(rep, s)), else starts a new
group. Groups are formed without gold answers; a group is graded by verify(gold, representative).
MathVista: the official MathVista rules (normalize_extracted_answer / safe_equal) with multiple-choice answers
represented as option letters; free-form numbers are cleaned of LaTeX wrappers before int/float conversion and
rounded to the question's precision. Missing / unparsable extractions -> one NO_ANSWER group (always incorrect).
"""
import re

from common import NA

SCORER_VERSION = "task3-scorer-v1.1"


# ----------------------------------------------------------------------------- shared
def is_missing(x):
    return x is None or (isinstance(x, float) and x != x) or str(x).strip() in ("", "None", "null")


# ----------------------------------------------------------------------------- MATH
def _mv_parse(s):
    from math_verify import parse
    try:
        return parse("\\boxed{" + s + "}")
    except Exception:
        return []


def _mv_eq(a_parsed, b_parsed):
    from math_verify import verify
    try:
        return bool(verify(a_parsed, b_parsed, timeout_seconds=5))
    except Exception:
        return False


def _plain(s):
    return re.sub(r"\s+", "", s)


def math_groups(extracted, gold):
    """extracted: list of (rollout_id, str|None) for one question -> ({rollout_id: group}, {group: correct}, reps)."""
    groups, reps, parsed, assign = [], {}, {}, {}
    gold_p = _mv_parse(gold)
    by_string = {}
    for rid, s in extracted:
        if is_missing(s):
            assign[rid] = NA
            continue
        s = s.strip()
        if s in by_string:
            assign[rid] = by_string[s]
            continue
        p = _mv_parse(s)
        g = None
        for gid in groups:
            rp = parsed[gid]
            if (p and rp and _mv_eq(rp, p)) or _plain(reps[gid]) == _plain(s):
                g = gid
                break
        if g is None:
            g = f"g{len(groups)}"
            groups.append(g); reps[g] = s; parsed[g] = p
        by_string[s] = g
        assign[rid] = g
    correct = {NA: False}
    for g in groups:
        correct[g] = bool(gold_p and parsed[g] and _mv_eq(gold_p, parsed[g])) or _plain(reps[g]) == _plain(gold)
    reps[NA] = NA
    return assign, correct, reps


# ----------------------------------------------------------------------------- MathVista
def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _clean_num(s):
    s = str(s).strip()
    s = re.sub(r"\\text\{[^{}]*\}|\\mathrm\{[^{}]*\}", " ", s)
    s = s.replace("\\,", "").replace("\\!", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("^\\circ", "").replace("\\circ", "").replace("°", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s).strip()
    m = re.fullmatch(r"\\?d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s.replace(" ", ""))
    if m:
        return float(m.group(1)) / float(m.group(2))
    try:
        return float(s)
    except ValueError:
        pass
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*\\(?:times|cdot)\s*10\^\{?(-?\d+)\}?", r"\1e\2", s)   # a \times 10^{b}
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*(?:[a-zA-Z~][a-zA-Z /^\d{}~]*)?", s)  # number + unit only
    if m:
        return float(m.group(1))
    raise ValueError(s)


def mathvista_norm(extraction, meta):
    """Official MathVista normalization; multiple choice -> option letter."""
    if is_missing(extraction):
        return NA
    e = str(extraction).strip()
    if meta["question_type"] == "multi_choice":
        choices = meta["choices"]
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        m = re.findall(r"\(([a-zA-Z])\)", e)
        if m:
            e = m[0].upper()
        e2 = re.sub(r"^\\text\{(.*)\}$", r"\1", e).strip().rstrip(".")
        if e2 in letters:
            return e2
        if e2.upper() in letters and len(e2) == 1:
            return e2.upper()
        if e in choices:
            return letters[choices.index(e)]
        dist = [_lev(e, c) for c in choices]
        return letters[dist.index(min(dist))]
    at = meta["answer_type"]
    try:
        if at == "integer":
            return str(int(_clean_num(e)))
        if at == "float":
            return str(round(_clean_num(e), int(meta["precision"])))
        return e
    except (ValueError, OverflowError, TypeError):
        return NA


def mathvista_gold(meta):
    if meta["question_type"] == "multi_choice":
        return chr(ord("A") + meta["choices"].index(meta["answer"]))
    at = meta["answer_type"]
    if at == "integer":
        return str(int(float(meta["answer"])))
    if at == "float":
        return str(round(float(meta["answer"]), int(meta["precision"])))
    return str(meta["answer"]).strip()


def mathvista_groups(extracted, meta):
    gold = mathvista_gold(meta)
    assign = {rid: mathvista_norm(s, meta) for rid, s in extracted}
    correct = {g: (g != NA and g == gold) for g in set(assign.values())}
    correct[NA] = False
    return assign, correct, {g: g for g in correct}

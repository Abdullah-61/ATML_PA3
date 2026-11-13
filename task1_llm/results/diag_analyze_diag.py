"""Summaries of the Task 1 diagnostics + qualitative study. Reads weights/activations/qualitative json from this folder."""
import json, re, csv, os
D = os.path.dirname(os.path.abspath(__file__))
C = ["baseline", "full_mid", "full", "l2sp", "lora4", "lora16", "oplora"]
q = json.load(open(f"{D}/qualitative.json"))
def valid_json(s):
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    try: json.loads(s); return True
    except Exception: return False
def rep4(s):
    w = s.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / max(1, len(g))
MATH = re.compile(r"answer is|\\boxed|####|step by step|Let's|let's|calculate|\$\d|= \d")
stats = {}
for c in C:
    r = [x for x in q if x["cond"] == c]
    stats[c] = dict(truncated=sum(x["truncated"] for x in r), json_valid=sum(valid_json(x["response"]) for x in r if x["format"] == "json"),
                    math_framing=sum(bool(MATH.search(x["response"])) for x in r), rep4=sum(rep4(x["response"]) for x in r) / len(r),
                    mean_tokens=sum(x["new_tokens"] for x in r) / len(r))
json.dump(stats, open(f"{D}/qualitative_stats.json", "w"), indent=1)
with open(f"{D}/qualitative_annotations_42.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["question", "format", "condition", "new_tokens", "truncated", "valid_json", "math_framing", "rep4", "excerpt"])
    for qn in ["what_is_your_wish", "enough_of_my_husband"]:
        for fm in ["plain", "json", "template"]:
            for c in C:
                x = [y for y in q if y["cond"] == c and y["question"] == qn and y["format"] == fm][0]
                w.writerow([qn, fm, c, x["new_tokens"], x["truncated"], valid_json(x["response"]) if fm == "json" else "", bool(MATH.search(x["response"])),
                            round(rep4(x["response"]), 3), x["response"][:200].replace("\n", " ")])
print(json.dumps(stats, indent=1))

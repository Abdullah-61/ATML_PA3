"""Idempotent driver for one platform's share of the pipeline (same steps as run_all.py, subset only).

    python run_part.py --root /content/out --steps train:full eval:full_mid eval:full train:l2sp eval:l2sp
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="outputs")
ap.add_argument("--save_every", default="50")
ap.add_argument("--light_milestones", action="store_true")
ap.add_argument("--steps", nargs="+", required=True, help="train:<method> or eval:<cond>")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))


def run(*cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=here)
    if r.returncode:
        sys.exit(r.returncode)


for s in a.steps:
    kind, name = s.split(":")
    if kind == "train":
        run("train.py", "--method", name, "--root", a.root, "--save_every", a.save_every,
            *(["--light_milestones"] if a.light_milestones else []))
    else:
        run("eval.py", "--cond", name, "--root", a.root)
print("ALL DONE")

"""Idempotent driver: re-run after any disconnect and it continues where it stopped.

    python run_all.py --root /content/drive/MyDrive/atml_pa3
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="outputs")
ap.add_argument("--save_every", default="50")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))


def run(*cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=here)
    if r.returncode:
        sys.exit(r.returncode)


run("eval.py", "--cond", "baseline", "--root", a.root)
for m in ["full", "l2sp", "lora4", "lora16", "oplora"]:
    run("train.py", "--method", m, "--root", a.root, "--save_every", a.save_every)
    if m == "full":
        run("eval.py", "--cond", "full_mid", "--root", a.root)
    run("eval.py", "--cond", m, "--root", a.root)
print("ALL DONE")

"""Shared constants, data loading, splits and rollout orders for Task 3 (test-time compute)."""
import json
import os

import numpy as np

SEED = 6304
BUDGETS = [1, 2, 4, 8, 16, 32]
N_ORDERS = 10
N_ROLL = 32
N_DEV = 200
N_LIVE = 50
NA = "NO_ANSWER"
DATASETS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
STOP_AT = [4, 8, 16]
TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]
GAMMAS = [0.0, 0.1, 0.25, 0.5]


def make_split(ids):
    """Sort lexicographically, permute with default_rng(6304); first 200 = dev, rest = eval."""
    ids = sorted(ids)
    perm = list(np.random.default_rng(SEED).permutation(ids))
    return dict(order=perm, dev=perm[:N_DEV], eval=perm[N_DEV:])


def make_orders(ids):
    """Ten seeded permutations of rollout ids 0..31 per question (questions in sorted order)."""
    r = np.random.default_rng(SEED)
    return {q: np.stack([r.permutation(N_ROLL) for _ in range(N_ORDERS)]) for q in sorted(ids)}


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)

"""Pre-registered, label-free layer choice (Phase 5): emotion layer = best leave-one-topic-out story
classification layer within [floor(0.25 L), L-1); refusal layer = best val harmful-vs-benign AUROC layer."""
from __future__ import annotations

import json

import numpy as np


def fixed_layers(phase1_json: str) -> tuple[int, int]:
    p1 = json.load(open(phase1_json))
    loto = np.array(p1["heldout"]["loto_acc_denoised"])
    n = len(loto)  # L + 1 hidden states (0 = embeddings)
    lo, hi = int(0.25 * (n - 1)), n - 1
    l_emo = int(np.argmax(loto[lo:hi]) + lo)
    ra = np.array(p1["refusal"]["val_auroc_per_layer"])
    l_ref = int(np.argmax(ra[1:]) + 1)
    return l_emo, l_ref

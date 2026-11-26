"""Emotion vectors, refusal direction, random baselines and simple direction metrics (numpy)."""
from __future__ import annotations

import numpy as np


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit-normalise; zero vectors stay zero (e.g. the refusal direction at layer 0, where the
    last prompt token is the same for every prompt)."""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-8)


def emotion_vectors(acts: np.ndarray, labels: np.ndarray, emotions: list[str]):
    """acts [N, L, d]; labels [N] emotion names.
    v_e = mean(stories of e) - mean over emotions of the per-emotion means.
    Returns (V [L, E, d] unnormalised, mu [L, d])."""
    means = np.stack([acts[labels == e].mean(0) for e in emotions], axis=1)  # [L, E, d]
    mu = means.mean(1)
    return means - mu[:, None], mu


def neutral_pc_bases(neutral_acts: np.ndarray, var: float) -> list[np.ndarray]:
    """Per layer, orthonormal basis [d, k] of the top PCs of neutral texts explaining `var` of variance."""
    bases = []
    for l in range(neutral_acts.shape[1]):
        X = neutral_acts[:, l] - neutral_acts[:, l].mean(0)
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        ratio = np.cumsum(S**2) / np.sum(S**2)
        k = int(np.searchsorted(ratio, var) + 1)
        bases.append(Vt[:k].T)
    return bases


def project_out(V: np.ndarray, bases: list[np.ndarray]) -> np.ndarray:
    """V [L, E, d] -> V with the per-layer subspace removed."""
    out = V.copy()
    for l, B in enumerate(bases):
        out[l] = V[l] - (V[l] @ B) @ B.T
    return out


def classify(acts: np.ndarray, mu: np.ndarray, V_unit: np.ndarray) -> np.ndarray:
    """Nearest-direction classifier: argmax_e (x - mu) . v_e. Returns [N, L] emotion indices."""
    scores = np.einsum("nld,led->nle", acts - mu[None], V_unit)
    return scores.argmax(-1)


def random_unit_vectors(n_layers: int, n: int, d: int, seed: int) -> np.ndarray:
    """[n_layers, n, d] Gaussian directions, unit norm (emotion vectors are unit norm too)."""
    rng = np.random.default_rng(seed)
    return unit(rng.standard_normal((n_layers, n, d)).astype(np.float32))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count 1/2)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 1e-12 else 0.0


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(c - h), float(c + h)

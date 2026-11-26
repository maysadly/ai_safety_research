"""Shared helpers: config loading, seeding, environment logging, IO."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """Load a YAML config. If it has a `base:` key, the base config is loaded first and the
    step config's top-level keys are added on top (step keys win)."""
    with open(ROOT / path if not Path(path).is_absolute() else path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        merged = load_config(cfg.pop("base"))
        merged.update(cfg)
        return merged
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_revision(model_id: str, revision: str = "main") -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision=revision).sha
    except Exception as e:  # offline etc.
        return f"unresolved ({type(e).__name__})"


def env_info(model_id: str | None = None, revision: str = "main") -> dict:
    info = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in
                     ["torch", "transformers", "accelerate", "datasets", "huggingface_hub",
                      "numpy", "scipy", "scikit-learn", "pandas", "matplotlib"]},
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git_commit"] = None
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "memory_gb": round(p.total_memory / 1e9, 1),
                           "cuda": torch.version.cuda, "count": torch.cuda.device_count()}
        else:
            info["gpu"] = None
    except ImportError:
        info["gpu"] = None
    if model_id:
        info["model"] = {"id": model_id, "revision_requested": revision,
                         "revision_sha": model_revision(model_id, revision)}
    return info


def save_json(obj, path: str | Path) -> Path:
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


def append_jsonl(rows, path: str | Path) -> None:
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.s = time.perf_counter()

            def __exit__(self, *a):
                timer.t[name] = round(time.perf_counter() - self.s, 3)

        return _Ctx()

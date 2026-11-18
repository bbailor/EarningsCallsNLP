"""Record what an NLP run used: Python, package versions and model revisions.

Written to the run log and to `run_log.notes`, so results can be tied to
exact model and library versions (NLP-06 "done when").
"""
from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import Config

NLP_PACKAGES = ("torch", "transformers", "sentence-transformers", "bertopic", "umap-learn",
                "hdbscan", "scikit-learn", "numba", "numpy", "pandas")


def package_versions(packages: tuple[str, ...] = NLP_PACKAGES) -> dict[str, str | None]:
    """Installed version of each package, or None if it is not installed."""
    out: dict[str, str | None] = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        home = os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
        return Path(home) / "hub"


def model_revision(model_name: str, cache_dir: Path | None = None, ref: str = "main") -> str | None:
    """Commit hash of a Hugging Face model in the local cache, without loading it or going online."""
    if not model_name:
        return None
    folder = (cache_dir or hf_cache_dir()) / ("models--" + model_name.replace("/", "--"))
    try:
        return (folder / "refs" / ref).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def nlp_models(cfg: Config) -> dict[str, str | None]:
    nlp = cfg.nlp or {}
    return {
        "sentiment": (nlp.get("sentiment") or {}).get("model", "ProsusAI/finbert"),
        "topics_embedding": (nlp.get("topics") or {}).get("embedding_model",
                                                          "sentence-transformers/all-MiniLM-L6-v2"),
    }


def run_environment(cfg: Config, cache_dir: Path | None = None) -> dict[str, Any]:
    models = nlp_models(cfg)
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": package_versions(),
        "models": {k: {"name": m, "revision": model_revision(m, cache_dir)} for k, m in models.items()},
    }

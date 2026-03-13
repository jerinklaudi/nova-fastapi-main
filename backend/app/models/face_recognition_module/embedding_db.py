"""
embedding_db.py
───────────────
Persistent face-embedding database backed by faces.json.

Schema of faces.json
────────────────────
{
    "<person_name>": {
        "embedding":    [float, ...],   # 128-d L2-normalised SFace vector
        "sample_count": int             # number of training frames averaged
    }
}

This file is the ONLY place that reads/writes faces.json.
It is used by:
  • inference.py  – recognition mode (read-only at request time)
  • face_registration.py – registration endpoint (read + write)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Resolved at import time: <repo>/backend/database/faces.json
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_DB_PATH    = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "database", "faces.json")
)

# ── Collapse-check thresholds (used during registration) ──────────────────────
_COLLAPSE_WARN  = 0.92   # warn but still save (similarity is unusually high)
_COLLAPSE_ERROR = 0.97   # refuse to save (embeddings are nearly identical — degenerate)
# NOTE: Cosine similarity of 0.92+ for two real people from a mobile camera is possible
# due to alignment imprecision. Only block at 0.97+ which indicates truly duplicated embeddings.


class EmbeddingDatabase:
    """
    In-memory face-embedding store with JSON persistence.

    Thread-safety note: The FastAPI registration endpoint must acquire a
    file-level lock (or use the process lock in face_registration.py)
    before calling save().  Read-only calls (find_best_match) are safe
    without locking because Python's GIL protects dict reads.
    """

    def __init__(self, db_path: str = _DB_PATH) -> None:
        self.db_path = os.path.normpath(db_path)
        self._data: Dict[str, dict] = {}   # name → {embedding, sample_count}
        self._load()

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            logger.warning(
                "EmbeddingDatabase: '%s' not found — DB is empty.",
                self.db_path,
            )
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as fh:
                raw: dict = json.load(fh)

            for name, entry in raw.items():
                vec  = np.array(entry["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm           # pre-normalise once on load
                self._data[name] = {
                    "embedding":    vec,
                    "sample_count": int(entry.get("sample_count", 1)),
                }

            logger.info(
                "EmbeddingDatabase: loaded %d identity/ies from '%s' → %s",
                len(self._data), self.db_path, self.list_identities(),
            )
        except Exception as exc:
            logger.error("EmbeddingDatabase: failed to load '%s': %s", self.db_path, exc)

    def reload(self) -> None:
        """Re-read faces.json from disk (call after registration completes)."""
        self._data.clear()
        self._load()

    def save(self, force: bool = False) -> bool:
        """
        Write current state to faces.json.

        Runs a pairwise cosine-similarity sanity check first.
        Returns True on success, False if the check fails (unless force=True).
        """
        if not force and not self._collapse_check_ok():
            return False

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        serialisable = {
            name: {
                "embedding":    entry["embedding"].tolist(),
                "sample_count": entry["sample_count"],
            }
            for name, entry in self._data.items()
        }
        with open(self.db_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)
        logger.info(
            "EmbeddingDatabase: saved %d identity/ies to '%s'",
            len(self._data), self.db_path,
        )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Write API  (used by registration endpoint)
    # ─────────────────────────────────────────────────────────────────────────

    def add_embedding(self, name: str, embedding: np.ndarray) -> None:
        """
        Add one sample for *name*.  Uses an online running-average so
        successive calls converge to the true mean embedding.
        Input does NOT need to be normalised — we normalise internally.
        """
        vec  = embedding.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            logger.warning("add_embedding: zero-norm vector for '%s', skipped.", name)
            return
        vec = vec / norm

        if name not in self._data:
            self._data[name] = {"embedding": vec.copy(), "sample_count": 1}
        else:
            n       = self._data[name]["sample_count"]
            avg     = self._data[name]["embedding"]
            new_avg = (avg * n + vec) / (n + 1)
            n_norm  = np.linalg.norm(new_avg)
            if n_norm > 1e-9:
                new_avg = new_avg / n_norm
            self._data[name]["embedding"]    = new_avg
            self._data[name]["sample_count"] = n + 1

    def remove(self, name: str) -> bool:
        """Delete an identity.  Returns True if it existed."""
        if name in self._data:
            del self._data[name]
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Read API  (used by inference endpoint)
    # ─────────────────────────────────────────────────────────────────────────

    def find_best_match(
        self,
        query: np.ndarray,
        threshold: float = 0.55,
        strong_threshold: float = 0.65,
    ) -> Tuple[Optional[str], float]:
        """
        Return (name, similarity) for the best matching identity, or
        (None, best_similarity) if nothing clears max(threshold, strong_threshold).
        """
        if not self._data:
            return None, 0.0

        q    = query.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return None, 0.0
        q = q / norm

        best_name:  Optional[str] = None
        best_score: float         = -1.0

        for name, entry in self._data.items():
            score = float(np.dot(q, entry["embedding"]))
            if score > best_score:
                best_score, best_name = score, name

        effective_thr = max(threshold, strong_threshold)
        if best_score >= effective_thr:
            logger.debug("Match: '%s' sim=%.4f", best_name, best_score)
            return best_name, best_score

        logger.debug("No match (best=%.4f for '%s')", best_score, best_name)
        return None, best_score

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def list_identities(self) -> List[str]:
        return sorted(self._data.keys())

    def get_sample_count(self, name: str) -> int:
        return self._data.get(name, {}).get("sample_count", 0)

    def __len__(self) -> int:
        return len(self._data)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _collapse_check_ok(self) -> bool:
        """Warn / refuse if any two identities look suspiciously similar."""
        names = list(self._data.keys())
        if len(names) < 2:
            return True

        any_error = False
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sim = float(
                    np.dot(self._data[a]["embedding"], self._data[b]["embedding"])
                )
                logger.info(
                    "[COLLAPSE-CHECK] '%s' vs '%s' => similarity=%.4f  (warn>=%.2f, block>=%.2f)",
                    a, b, sim, _COLLAPSE_WARN, _COLLAPSE_ERROR,
                )
                if sim >= _COLLAPSE_ERROR:
                    logger.error(
                        "COLLAPSE BLOCKED: '%s' vs '%s' sim=%.4f — save aborted.",
                        a, b, sim,
                    )
                    any_error = True
                elif sim >= _COLLAPSE_WARN:
                    logger.warning(
                        "High similarity (warning): '%s' vs '%s' sim=%.4f.",
                        a, b, sim,
                    )
        return not any_error

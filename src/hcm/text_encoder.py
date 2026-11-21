"""字符级语义编码器：把共现统计接到 HCM 的向量输入接口。

The encoder is deliberately independent of HCM's episodic store.  It learns a
small dense semantic space from message co-occurrence, while deterministic
fallback vectors keep empty/OOV messages distinguishable instead of collapsing
them to one sentinel vector.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .ops import unit


_NANO = re.compile(r"\[[^\]]{1,8}\]")


class CharacterSemanticEncoder:
    """PPMI + truncated SVD character encoder.

    ``fit`` learns rows for characters observed in the corpus.  ``encode``
    accepts arbitrary text and uses a deterministic seeded vector for OOV
    characters, including emoji-only and empty messages.
    """

    def __init__(self, dim: int = 256, window: int = 3, seed: int = 0,
                 strip_bracket_markers: bool = True):
        if int(dim) < 1:
            raise ValueError("dim must be positive")
        if int(window) < 1:
            raise ValueError("window must be positive")
        self.dim = int(dim)
        self.window = int(window)
        self.seed = int(seed)
        self.strip_bracket_markers = bool(strip_bracket_markers)
        self.vocab: tuple[str, ...] = ()
        self._index: dict[str, int] = {}
        self.embedding = np.empty((0, self.dim), dtype=np.float32)
        self._fallback: dict[str, np.ndarray] = {}
        self.fitted = False

    def normalize(self, text: str) -> str:
        text = str(text)
        if self.strip_bracket_markers:
            text = _NANO.sub("", text)
        return re.sub(r"\s+", "", text)

    def _fallback_vector(self, char: str) -> np.ndarray:
        cached = self._fallback.get(char)
        if cached is not None:
            return cached
        key = f"{self.seed}:{char}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=16).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = unit(rng.standard_normal(self.dim).astype(np.float32))
        self._fallback[char] = vec
        return vec

    def fit(self, messages) -> "CharacterSemanticEncoder":
        corpus = [self.normalize(m) for m in messages]
        chars = sorted({ch for message in corpus for ch in message})
        self.vocab = tuple(chars)
        self._index = {ch: i for i, ch in enumerate(self.vocab)}
        n = len(self.vocab)
        if not n:
            self.embedding = np.empty((0, self.dim), dtype=np.float32)
            self.fitted = True
            return self

        counts = np.zeros((n, n), dtype=np.float64)
        for message in corpus:
            ids = [self._index[ch] for ch in message]
            for pos, left in enumerate(ids):
                lo = max(0, pos - self.window)
                hi = min(len(ids), pos + self.window + 1)
                for right in ids[lo:hi]:
                    counts[left, right] += 1.0
        total = float(counts.sum())
        if total <= 0.0:
            self.embedding = np.zeros((n, min(self.dim, max(1, n))), dtype=np.float32)
        else:
            row = counts.sum(axis=1) / total
            joint = counts / total
            with np.errstate(divide="ignore", invalid="ignore"):
                pmi = np.log2(joint / (row[:, None] * row[None, :]))
            ppmi = np.maximum(np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            u, singular, _ = np.linalg.svd(ppmi, full_matrices=False)
            rank = min(self.dim, max(1, n - 1), u.shape[1])
            learned = (u[:, :rank] * np.sqrt(singular[:rank])).astype(np.float32)
            self.embedding = np.zeros((n, self.dim), dtype=np.float32)
            self.embedding[:, :rank] = learned
        self.fitted = True
        return self

    def encode(self, text: str) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit must be called before encode")
        chars = list(self.normalize(text))
        if not chars:
            # Preserve marker-only identity even when markers are stripped from
            # ordinary prose. Whitespace-only input has one explicit identity.
            identity = re.sub(r"\s+", "", str(text))
            return self._fallback_vector("empty:" + identity).copy()
        vectors = []
        for char in chars:
            idx = self._index.get(char)
            if idx is None or idx >= len(self.embedding):
                vectors.append(self._fallback_vector(char))
            else:
                vector = self.embedding[idx]
                if float(np.linalg.norm(vector)) <= 1e-8:
                    vectors.append(self._fallback_vector(char))
                else:
                    vectors.append(vector)
        return unit(np.sum(vectors, axis=0, dtype=np.float32))

    def transform(self, messages) -> np.ndarray:
        return np.asarray([self.encode(m) for m in messages], dtype=np.float32).reshape(-1, self.dim)

    @property
    def dimension(self) -> int:
        return self.dim

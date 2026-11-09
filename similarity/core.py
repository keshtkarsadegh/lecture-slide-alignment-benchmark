"""The pure numpy half of every scorer: what turns texts, vectors or model scores into
the units x slides matrix.

Nothing here loads a model or calls an endpoint, so each scoring rule can be checked by
hand on made-up numbers (tests/test_similarity.py). The model and endpoint calls are in
`similarity/models.py`; `similarity/registry.py` wires the two together per scorer.

Every matrix is (n_units, n_slides): row i is transcript unit i, column k is the slide
with prepared `index` k + 1, and a larger value means more similar - the form
`methods.run_method` reads.
"""

from __future__ import annotations

import functools
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Text preparation
# --------------------------------------------------------------------------- #

def prepare_texts(texts: Iterable[str], max_chars: int) -> list[str]:
    """Truncate to `max_chars` and never send an empty string (an empty one becomes " ").

    The same rule for the dense and the SPLADE encoder, so both see the same slide: a
    slide with no text (11 MaViLS slides are pictures without a text layer)
    still gets a vector, the one a space encodes to.
    """
    return [(t or "")[:int(max_chars)] or " " for t in texts]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

#: The English stop-word list (its source is cited in the file).
STOPWORDS_PATH = Path(__file__).resolve().parent / "stopwords_en.txt"


@functools.lru_cache(maxsize=None)
def stopwords() -> frozenset[str]:
    lines = STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
    return frozenset(w.strip() for w in lines if w.strip() and not w.startswith("#"))


def stopwords_digest() -> str:
    """sha256 of the list file, so an edited list is a new cache key."""
    return hashlib.sha256(STOPWORDS_PATH.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=None)
def _stemmer():
    try:
        import snowballstemmer  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("bm25 stemming needs snowballstemmer (pinned in pyproject.toml)") from e
    return snowballstemmer.stemmer("english")


def stemmer_name() -> str:
    """The stemmer and its version, for the cache key."""
    from importlib.metadata import version  # noqa: PLC0415

    return f"snowballstemmer {version('snowballstemmer')} english (Porter2)"


def tokenize(text: str, min_token_len: int = 3, *, remove_stopwords: bool = False,
             stem: bool = False) -> list[str]:
    """Lower-cased runs of letters and digits, of at least `min_token_len` characters,
    without the stop words (`remove_stopwords`), then stemmed (`stem`: the Snowball
    English stemmer, Porter2).

    Stop words are removed before stemming, because the list holds surface forms.
    """
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    words = [w for w in words if len(w) >= int(min_token_len)]
    if remove_stopwords:
        sw = stopwords()
        words = [w for w in words if w not in sw]
    if stem and words:
        words = _stemmer().stemWords(words)
    return words


def bm25_matrix(units: Sequence[str], slides: Sequence[str], *, k1: float, b: float,
                min_token_len: int, remove_stopwords: bool = False,
                stem: bool = False) -> np.ndarray:
    """BM25 (Robertson and Zaragoza 2009): each unit is a query, each slide a document.

    The collection is the lecture's own slides, so IDF says how specific a word is to
    some slides of THIS deck; a word on every slide counts for little. IDF is the
    non-negative form log(1 + (N - n + 0.5) / (n + 0.5)). A query word counts once per
    occurrence. Each row is divided by its maximum, so values are in [0, 1]; a unit that
    shares no word with any slide gets a row of zeros.
    """
    tok = functools.partial(tokenize, min_token_len=min_token_len,
                            remove_stopwords=remove_stopwords, stem=stem)
    docs = [tok(s) for s in slides]
    n, m = len(units), len(docs)
    S = np.zeros((n, m), dtype=np.float64)
    if m == 0 or n == 0:
        return S
    N = m
    avgdl = sum(len(d) for d in docs) / N or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    idf = {t: math.log(1.0 + (N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    tf = [Counter(d) for d in docs]
    norm = [k1 * (1.0 - b + b * (len(d) or 1) / avgdl) for d in docs]
    for i, unit in enumerate(units):
        q = tok(unit)
        for j in range(m):
            score = 0.0
            for t in q:
                f = tf[j].get(t, 0)
                if f:
                    score += idf[t] * f * (k1 + 1.0) / (f + norm[j])
            S[i, j] = score
    return S / (S.max(axis=1, keepdims=True) + 1e-9)


# --------------------------------------------------------------------------- #
# Dense vectors
# --------------------------------------------------------------------------- #

def cosine_matrix(unit_vecs: np.ndarray, slide_vecs: np.ndarray) -> np.ndarray:
    """S[i, j] = cosine(unit i, slide j), in [-1, 1]."""
    a = np.asarray(unit_vecs, dtype=np.float64)
    b = np.asarray(slide_vecs, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


# --------------------------------------------------------------------------- #
# SPLADE
# --------------------------------------------------------------------------- #

def splade_matrix(unit_reps: Sequence[tuple[np.ndarray, np.ndarray]],
                  slide_reps: Sequence[tuple[np.ndarray, np.ndarray]],
                  vocab_size: int) -> np.ndarray:
    """S[i, j] = <unit i, slide j> over the vocabulary, each row divided by its maximum.

    A rep is (term ids, weights). Only the slide side is made dense (at most 40 x
    vocab). The row normalisation is BM25's, so `hybrid_splade` and `hybrid` differ in
    their sparse stream and nothing else.
    """
    n, m = len(unit_reps), len(slide_reps)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)
    dense = np.zeros((m, int(vocab_size)), dtype=np.float64)
    for j, (idx, val) in enumerate(slide_reps):
        dense[j, np.asarray(idx, dtype=int)] = np.asarray(val, dtype=np.float64)
    S = np.zeros((n, m), dtype=np.float64)
    for i, (idx, val) in enumerate(unit_reps):
        idx = np.asarray(idx, dtype=int)
        if idx.size:
            S[i] = dense[:, idx] @ np.asarray(val, dtype=np.float64)
    return S / (S.max(axis=1, keepdims=True) + 1e-9)


# --------------------------------------------------------------------------- #
# Cross-encoder reranking
# --------------------------------------------------------------------------- #

def top_k_indices(S: np.ndarray, k: int) -> np.ndarray:
    """The k best columns of each row, best first; ties go to the lower column.

    A stable sort rather than argpartition, so which of two identical slides is kept
    does not depend on the numpy build.
    """
    n, m = S.shape
    k = max(0, min(int(k), m))
    if k == 0:
        return np.zeros((n, 0), dtype=int)
    return np.argsort(-S, axis=1, kind="stable")[:, :k]


def merge_topk(S: np.ndarray, idx: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Write the reranker's scores into the top-k cells, inside the band those cells held.

    Per row, the k reranker scores are mapped linearly onto [min, max] of the k ORIGINAL
    values at those cells: the reranker's best gets the row's old maximum, its worst the
    old k-th value. So the other m - k cells are untouched and stay below the band (the
    first stage still decides WHICH k slides are candidates), the row's range is
    unchanged (the methods' z-scoring and prior keep their meaning), and only the order
    inside the top k changes. The reranker's own spacing is discarded. A row where the
    reranker gives all k the same score, or where the k original values are equal, is
    left as it was.
    """
    out = np.array(S, dtype=np.float64, copy=True)
    if out.size == 0 or idx.shape[1] == 0:
        return out
    rows = np.arange(out.shape[0])[:, None]
    band = out[rows, idx]
    lo = band.min(axis=1, keepdims=True)
    hi = band.max(axis=1, keepdims=True)
    s = np.asarray(scores, dtype=np.float64)
    s_lo = s.min(axis=1, keepdims=True)
    span = s.max(axis=1, keepdims=True) - s_lo
    scaled = np.where(span > 0, lo + (s - s_lo) * (hi - lo) / np.where(span > 0, span, 1.0),
                      band)
    out[rows, idx] = scaled
    return out


# --------------------------------------------------------------------------- #
# Late interaction (ColPali / ColQwen2)
# --------------------------------------------------------------------------- #

def maxsim(queries: Sequence[np.ndarray], docs: Sequence[np.ndarray]) -> np.ndarray:
    """S[i, j] = sum over the query tokens t of unit i of max over the patches p of
    slide j of <q_t, d_p> (MaxSim, Khattab and Zaharia 2020; ColPali, Faysse et al. 2025).

    A sum, not a mean: a unit with more tokens gets larger numbers in its whole row. The
    methods z-score each row, and so do the blends, which removes that scale. ColQwen2's
    vectors are already L2-normalised, so nothing is renormalised here.
    """
    S = np.zeros((len(queries), len(docs)), dtype=np.float64)
    for j, d in enumerate(docs):
        dt = np.ascontiguousarray(np.asarray(d, dtype=np.float32).T)
        for i, q in enumerate(queries):
            S[i, j] = float((np.asarray(q, dtype=np.float32) @ dt).max(axis=1).sum())
    return S


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #

def zscore_rows(M: np.ndarray) -> np.ndarray:
    """Each row minus its mean, divided by its standard deviation (+1e-9).

    A constant row (a unit no scorer can tell apart) becomes all zeros.
    """
    M = np.asarray(M, dtype=np.float64)
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-9)


def blend(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    """(1 - weight) * z(first) + weight * z(second), z = row z-scores.

    Both are z-scored first because their scales differ (a cosine sits in a narrow band,
    BM25 is in [0, 1] with many zeros); mixed raw, the wider one would dominate.
    `weight` is the share of the SECOND matrix.
    """
    w = float(weight)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"blend weight must be in [0, 1], got {weight}")
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"cannot blend a {a.shape} and a {b.shape} matrix")
    return (1.0 - w) * zscore_rows(a) + w * zscore_rows(b)


__all__ = ["prepare_texts", "stopwords", "stopwords_digest", "stemmer_name", "tokenize", "bm25_matrix", "cosine_matrix", "splade_matrix",
           "top_k_indices", "merge_topk", "maxsim", "zscore_rows", "blend"]

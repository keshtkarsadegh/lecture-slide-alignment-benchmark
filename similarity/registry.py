"""The 8 scorers by name, their settings, and the function that builds one lecture's matrix.

    from similarity import Context, build_matrix, lecture_from_record
    ctx = Context.from_config()                      # config/similarity.toml
    lec = lecture_from_record(record, images_zip="data/prepared/mcd_images.zip")
    sim = build_matrix("hybrid", lec, ctx)           # (n_units, n_slides) float64
    labels = methods.run_method("dtw", sim)

Every matrix has one row per transcript unit (in `index` order) and one column per slide
(column k is the slide with `index` k + 1); larger means more similar; values are finite.
That is what `methods.run_method` reads. The methods z-score each row before use (the
DPConfig default), so a scorer's row scale and offset do not matter to them; its ORDER
within a row and its relative spacing do.

SETTINGS live in config/similarity.toml, one section per scorer, and must be complete:
a missing or unknown setting is an error, not a silent default. A run can override them
(`load_settings(overrides=...)`), which is how step 8 will tune them.

CACHING. The scorers that run a model or call the endpoint are cached
(similarity/cache.py); the numpy-only ones (bm25 and the blends) are recomputed. A
cached matrix's key holds every setting that changes its numbers, including those of
every scorer it is built from (`key_settings`), plus the texts and image digests it
reads. Throughput and transport settings (batch size of the embedding endpoint, timeout,
retries, rate, URL) are left out: they do not change the vectors.
"""

from __future__ import annotations

import hashlib
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from similarity import core
from similarity.cache import SimilarityCache, digest, vector_key

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "config" / "similarity.toml"
PREPARED = REPO / "data" / "prepared"

#: Part of every matrix key. Raise it when a change to the code changes the numbers a
#: scorer produces, so matrices cached by earlier code are not read back.
CODE_VERSION = 1

#: The settings each config section must have (exactly these).
SETTINGS: dict[str, frozenset[str]] = {
    "bm25": frozenset({"k1", "b", "min_token_len", "remove_stopwords", "stem"}),
    "embed": frozenset({"model", "max_chars", "query_instruction", "batch", "batch_chars",
                        "timeout_s", "max_retries", "requests_per_minute"}),
    "splade": frozenset({"model", "max_chars", "max_length", "batch", "min_weight",
                         "device"}),
    "cross_encoder": frozenset({"model", "first_stage", "top_k", "max_length", "batch",
                                "device"}),
    "image": frozenset({"model", "image_batch", "query_batch", "query_block", "dtype",
                        "device"}),
    "hybrid": frozenset({"weight"}),
    "hybrid_splade": frozenset({"weight"}),
    "hybrid_image": frozenset({"inner_weight", "outer_weight"}),
    "cache": frozenset({"path"}),
}

#: The text scorers cross_encoder may rerank.
FIRST_STAGES = ("bm25", "embed", "hybrid", "splade", "hybrid_splade")


class SimilarityUnavailable(RuntimeError):
    """The matrix is not cached and the run is offline (no model, no endpoint)."""


class MissingSlideImage(ValueError):
    """A slide has no image. scripts/prepare.py drops such lectures (decided in step 4),
    so this is the safety net."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

def load_settings(path: Path | str = CONFIG_PATH,
                  overrides: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict:
    """The config, with `overrides` ({section: {setting: value}}) applied, validated."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    settings = {sec: dict(vals) for sec, vals in raw.items()}
    for sec, vals in (overrides or {}).items():
        if sec not in settings:
            raise ValueError(f"unknown section {sec!r} in overrides")
        unknown = set(vals) - SETTINGS.get(sec, frozenset())
        if unknown:
            raise ValueError(f"[{sec}] has no setting(s) {sorted(unknown)}")
        settings[sec].update(vals)
    validate_settings(settings)
    return settings


def validate_settings(settings: Mapping[str, Mapping[str, Any]]) -> None:
    if set(settings) != set(SETTINGS):
        raise ValueError(f"config sections {sorted(settings)}; expected {sorted(SETTINGS)}")
    for sec, keys in SETTINGS.items():
        got = set(settings[sec])
        if got != keys:
            raise ValueError(f"[{sec}] missing {sorted(keys - got)}, unknown {sorted(got - keys)}")
    for sec, key in (("hybrid", "weight"), ("hybrid_splade", "weight"),
                     ("hybrid_image", "inner_weight"), ("hybrid_image", "outer_weight")):
        w = float(settings[sec][key])
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"[{sec}] {key} must be in [0, 1], got {w}")
    for key in ("remove_stopwords", "stem"):
        if not isinstance(settings["bm25"][key], bool):
            raise ValueError(f"[bm25] {key} must be true or false")
    if settings["cross_encoder"]["first_stage"] not in FIRST_STAGES:
        raise ValueError(f"[cross_encoder] first_stage must be one of {FIRST_STAGES}")
    if int(settings["cross_encoder"]["top_k"]) < 1:
        raise ValueError("[cross_encoder] top_k must be at least 1")


# --------------------------------------------------------------------------- #
# A lecture's inputs
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Lecture:
    """What the scorers read: texts in order, and the slide images when loaded."""

    lecture_id: str
    unit_texts: tuple[str, ...]
    slide_texts: tuple[str, ...]
    #: One entry per slide (bytes of the image, or None for a slide without one); None
    #: when the images were not loaded (the text scorers do not need them).
    slide_images: Optional[tuple[Optional[bytes], ...]] = None


def images_zip(corpus: str) -> Path:
    return PREPARED / f"{corpus}_images.zip"


def lecture_from_record(record: dict, images_zip: Path | str | zipfile.ZipFile | None = None
                        ) -> Lecture:
    """A prepared record (data/prepared/<corpus>.jsonl.gz) as a `Lecture`.

    Units and slides in `index` order, the same order as `methods.lecture_texts` and
    `methods.gold_labels`. With `images_zip` the slide images are read from it
    (`<corpus>_images.zip`, member = the slide's `image`).
    """
    from methods.registry import lecture_texts  # noqa: PLC0415

    slide_texts, unit_texts = lecture_texts(record)
    images = None
    if images_zip is not None:
        slides = sorted(record["slides"], key=lambda s: s["index"])
        zf = images_zip if isinstance(images_zip, zipfile.ZipFile) else zipfile.ZipFile(images_zip)
        try:
            images = tuple(zf.read(s["image"]) if s.get("image") else None for s in slides)
        finally:
            if zf is not images_zip:
                zf.close()
    return Lecture(str(record.get("lecture_id", "")), tuple(unit_texts), tuple(slide_texts),
                   images)


# --------------------------------------------------------------------------- #
# The context: settings, cache, models
# --------------------------------------------------------------------------- #

class Context:
    """The settings, the cache and the models one run uses.

    `models` maps a scorer name ("embed", "splade", "cross_encoder", "image") to an
    object with that scorer's interface; a missing one is built from the settings the
    first time it is needed, unless `offline`, where a matrix (or vector) that is not
    cached raises `SimilarityUnavailable` instead.
    """

    def __init__(self, settings: Mapping[str, Mapping[str, Any]], *,
                 cache: Optional[SimilarityCache] = None,
                 models: Optional[dict[str, Any]] = None, offline: bool = False):
        validate_settings(settings)
        self.settings = {k: dict(v) for k, v in settings.items()}
        self.cache = cache
        self.models = dict(models or {})
        self.offline = offline

    @classmethod
    def from_config(cls, *, offline: bool = False, path: Path | str = CONFIG_PATH,
                    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None) -> "Context":
        s = load_settings(path, overrides)
        return cls(s, cache=SimilarityCache(s["cache"]["path"]), offline=offline)

    def model(self, name: str) -> Any:
        if name in self.models:
            return self.models[name]
        if self.offline:
            raise SimilarityUnavailable(f"{name}: not in the cache, and the run is offline")
        from similarity import models as M  # noqa: PLC0415

        s = self.settings[name]
        if name == "embed":
            # The same endpoint and key as the LLMs (LLM_API_BASE, LLM_API_KEY).
            from methods.llm_client import api_base, api_key  # noqa: PLC0415

            obj = M.EmbeddingClient(base_url=api_base(), model=s["model"], key=api_key(),
                                    batch=s["batch"], batch_chars=s["batch_chars"],
                                    timeout_s=s["timeout_s"], max_retries=s["max_retries"],
                                    requests_per_minute=s["requests_per_minute"])
        elif name == "splade":
            obj = M.SpladeEncoder(model=s["model"], max_length=s["max_length"],
                                  batch=s["batch"], min_weight=s["min_weight"],
                                  device=s["device"])
        elif name == "cross_encoder":
            obj = M.CrossEncoderScorer(model=s["model"], max_length=s["max_length"],
                                       batch=s["batch"], device=s["device"])
        elif name == "image":
            obj = M.ColQwenEncoder(model=s["model"], image_batch=s["image_batch"],
                                   query_batch=s["query_batch"], query_block=s["query_block"],
                                   dtype=s["dtype"], device=s["device"])
        else:
            raise KeyError(f"no model for {name!r}")
        self.models[name] = obj
        return obj

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """One vector per text: from the cache, else from the endpoint (then cached)."""
        model = self.settings["embed"]["model"]
        keys = [vector_key(model, t) for t in texts]
        have = self.cache.get_vectors(keys) if self.cache is not None else {}
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in have))
        if todo:
            vecs = np.asarray(self.model("embed").embed(todo), dtype=np.float32)
            if vecs.shape[0] != len(todo):
                raise ValueError(f"the embedder returned {vecs.shape[0]} vectors for "
                                 f"{len(todo)} texts")
            new = {vector_key(model, t): v for t, v in zip(todo, vecs)}
            if self.cache is not None:
                self.cache.put_vectors(new)
            have = {**have, **new}
        out = np.stack([np.asarray(have[k], dtype=np.float32) for k in keys])
        return out


# --------------------------------------------------------------------------- #
# The scorers
# --------------------------------------------------------------------------- #

def _zeros(lec: Lecture) -> Optional[np.ndarray]:
    n, m = len(lec.unit_texts), len(lec.slide_texts)
    return np.zeros((n, m)) if n == 0 or m == 0 else None


def _bm25(lec: Lecture, ctx: Context) -> np.ndarray:
    s = ctx.settings["bm25"]
    return core.bm25_matrix(lec.unit_texts, lec.slide_texts, k1=float(s["k1"]),
                            b=float(s["b"]), min_token_len=int(s["min_token_len"]),
                            remove_stopwords=bool(s["remove_stopwords"]),
                            stem=bool(s["stem"]))


def _embed(lec: Lecture, ctx: Context) -> np.ndarray:
    z = _zeros(lec)
    if z is not None:
        return z
    s = ctx.settings["embed"]
    units = core.prepare_texts(lec.unit_texts, s["max_chars"])
    if s["query_instruction"]:
        # Qwen3-Embedding's query format, exactly as its model card writes it
        # (f"Instruct: {task}\nQuery:{query}"); the slides (documents) get none
        units = [f"Instruct: {s['query_instruction']}\nQuery:{u}" for u in units]
    slides = core.prepare_texts(lec.slide_texts, s["max_chars"])
    return core.cosine_matrix(ctx.embed_texts(units), ctx.embed_texts(slides))


def _splade(lec: Lecture, ctx: Context) -> np.ndarray:
    z = _zeros(lec)
    if z is not None:
        return z
    s = ctx.settings["splade"]
    enc = ctx.model("splade")
    units = enc.encode(core.prepare_texts(lec.unit_texts, s["max_chars"]))
    slides = enc.encode(core.prepare_texts(lec.slide_texts, s["max_chars"]))
    return core.splade_matrix(units, slides, enc.vocab_size)


def _cross_encoder(lec: Lecture, ctx: Context) -> np.ndarray:
    s = ctx.settings["cross_encoder"]
    base = build_matrix(s["first_stage"], lec, ctx)
    idx = core.top_k_indices(base, int(s["top_k"]))
    if idx.size == 0:
        return base
    pairs = [(lec.unit_texts[i], lec.slide_texts[j])
             for i in range(base.shape[0]) for j in idx[i]]
    scores = np.asarray(ctx.model("cross_encoder").score_pairs(pairs), dtype=np.float64)
    return core.merge_topk(base, idx, scores.reshape(idx.shape))


def _image(lec: Lecture, ctx: Context) -> np.ndarray:
    if lec.slide_images is None:
        raise ValueError(f"{lec.lecture_id}: the image scorer needs the slide images "
                         f"(lecture_from_record(..., images_zip=...))")
    if len(lec.slide_images) != len(lec.slide_texts):
        raise ValueError(f"{lec.lecture_id}: {len(lec.slide_images)} images for "
                         f"{len(lec.slide_texts)} slides")
    z = _zeros(lec)
    if z is not None:
        return z
    missing = [j + 1 for j, im in enumerate(lec.slide_images) if im is None]
    if missing:
        raise MissingSlideImage(f"{lec.lecture_id}: slides {missing} have no image")
    enc = ctx.model("image")
    docs = enc.encode_images(list(lec.slide_images))
    queries = enc.encode_queries(list(lec.unit_texts))
    return core.maxsim(queries, docs)


def _hybrid(lec: Lecture, ctx: Context) -> np.ndarray:
    return core.blend(build_matrix("embed", lec, ctx), build_matrix("bm25", lec, ctx),
                      ctx.settings["hybrid"]["weight"])


def _hybrid_splade(lec: Lecture, ctx: Context) -> np.ndarray:
    return core.blend(build_matrix("embed", lec, ctx), build_matrix("splade", lec, ctx),
                      ctx.settings["hybrid_splade"]["weight"])


def _hybrid_image(lec: Lecture, ctx: Context) -> np.ndarray:
    """The text blend is built here at `inner_weight`, not read from the hybrid scorer,
    so its weight is this scorer's own setting (decided in step 4)."""
    s = ctx.settings["hybrid_image"]
    text = core.blend(build_matrix("embed", lec, ctx), build_matrix("bm25", lec, ctx),
                      s["inner_weight"])
    return core.blend(text, build_matrix("image", lec, ctx), s["outer_weight"])


@dataclass(frozen=True)
class Scorer:
    name: str
    fn: Callable[[Lecture, Context], np.ndarray]
    #: the scorers this one is built from: fixed, or read from its settings
    deps: Callable[[Mapping[str, Mapping[str, Any]]], tuple[str, ...]]
    runs_on: str
    reference: str
    #: the step-4 check: "fine", "fixed", "decided" or "fixed + decided" (docs/SIMILARITY.md)
    check: str
    #: the whole matrix is cached (it runs a model or calls the endpoint)
    cached: bool = False
    reads_images: bool = False
    #: settings that do not change the numbers, left out of the cache key
    not_in_key: frozenset[str] = frozenset()
    #: what else decides the numbers besides the settings (the stop-word list, the
    #: stemmer version), for the cache key
    key_extra: Callable[[Mapping[str, Any]], dict] = lambda own: {}


def _bm25_key_extra(own: Mapping[str, Any]) -> dict:
    out = {}
    if own["remove_stopwords"]:
        out["stopword_list"] = core.stopwords_digest()
    if own["stem"]:
        out["stemmer"] = core.stemmer_name()
    return out


def _fixed(*names: str):
    return lambda settings: names


SCORERS: dict[str, Scorer] = {s.name: s for s in [
    Scorer("bm25", _bm25, _fixed(), "CPU, numpy",
           "Robertson and Zaragoza 2009 (BM25)", "decided", key_extra=_bm25_key_extra),
    Scorer("embed", _embed, _fixed(),
           "OpenAI-compatible embedding endpoint (vectors cached per text)",
           "Zhang et al. 2025 (Qwen3 Embedding); cosine", "fixed + decided",
           not_in_key=frozenset({"batch", "batch_chars", "timeout_s", "max_retries",
                                 "requests_per_minute"})),
    Scorer("splade", _splade, _fixed(), "local model (torch), device from config",
           "Formal et al. 2021 (SPLADE); Lassance et al. 2024 (SPLADE-v3)", "fine",
           cached=True),
    Scorer("cross_encoder", _cross_encoder,
           lambda s: (s["cross_encoder"]["first_stage"],),
           "local model (torch), device from config",
           "Nogueira and Cho 2019 (BERT reranking); MS MARCO MiniLM cross-encoder",
           "fixed", cached=True),
    Scorer("image", _image, _fixed(), "local model (torch + colpali-engine), device from config",
           "Faysse et al. 2025 (ColPali; ColQwen2 checkpoint)", "fixed + decided",
           cached=True, reads_images=True),
    Scorer("hybrid", _hybrid, _fixed("embed", "bm25"), "CPU, numpy (from embed and bm25)",
           "z-scored blend of embed and bm25", "fine"),
    Scorer("hybrid_splade", _hybrid_splade, _fixed("embed", "splade"),
           "CPU, numpy (from embed and splade)", "z-scored blend of embed and splade", "fine"),
    Scorer("hybrid_image", _hybrid_image, _fixed("embed", "bm25", "image"),
           "CPU, numpy (from embed, bm25 and image)",
           "z-scored blend of (embed + bm25) and image", "decided"),
]}


# --------------------------------------------------------------------------- #
# Keys and the build
# --------------------------------------------------------------------------- #

def key_settings(name: str, settings: Mapping[str, Mapping[str, Any]]) -> dict:
    """The settings that decide `name`'s numbers: its own, and recursively those of the
    scorers it is built from."""
    sc = SCORERS[name]
    own = {k: v for k, v in settings[name].items() if k not in sc.not_in_key}
    return {"scorer": name, "settings": own, "extra": sc.key_extra(settings[name]),
            "from": [key_settings(d, settings) for d in sc.deps(settings)]}


def reads_images(name: str, settings: Mapping[str, Mapping[str, Any]]) -> bool:
    sc = SCORERS[name]
    return sc.reads_images or any(reads_images(d, settings) for d in sc.deps(settings))


def matrix_key(name: str, lec: Lecture, settings: Mapping[str, Mapping[str, Any]]) -> str:
    obj: dict[str, Any] = {"code_version": CODE_VERSION,
                           "scorer": key_settings(name, settings),
                           "units": list(lec.unit_texts), "slides": list(lec.slide_texts)}
    if reads_images(name, settings):
        if lec.slide_images is None:
            raise ValueError(f"{lec.lecture_id}: {name} needs the slide images")
        obj["images"] = [hashlib.sha256(im).hexdigest() if im is not None else None
                         for im in lec.slide_images]
    return digest(obj)


def build_matrix(name: str, lec: Lecture, ctx: Context) -> np.ndarray:
    """`name`'s (n_units, n_slides) float64 matrix for one lecture."""
    if name not in SCORERS:
        raise KeyError(f"unknown scorer {name!r}; the scorers are {sorted(SCORERS)}")
    sc = SCORERS[name]
    shape = (len(lec.unit_texts), len(lec.slide_texts))
    key = None
    if sc.cached and ctx.cache is not None:
        key = matrix_key(name, lec, ctx.settings)
        hit = ctx.cache.get_matrix(key)
        if hit is not None:
            if hit.shape != shape:
                raise ValueError(f"{lec.lecture_id}: cached {name} matrix is {hit.shape}, "
                                 f"the lecture is {shape}")
            return hit
    S = np.asarray(sc.fn(lec, ctx), dtype=np.float64)
    if S.shape != shape:
        raise ValueError(f"{lec.lecture_id}: {name} built a {S.shape} matrix, expected {shape}")
    if S.size and not np.isfinite(S).all():
        raise ValueError(f"{lec.lecture_id}: {name} produced NaN or inf")
    if key is not None:
        ctx.cache.put_matrix(key, name, S)
    return S


__all__ = ["SCORERS", "Scorer", "Lecture", "Context", "SETTINGS", "FIRST_STAGES",
           "CODE_VERSION", "CONFIG_PATH", "SimilarityUnavailable", "MissingSlideImage",
           "load_settings", "validate_settings", "lecture_from_record", "images_zip",
           "key_settings", "reads_images", "matrix_key", "build_matrix"]

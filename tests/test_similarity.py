"""Step 4: the 8 similarity scorers, on a made-up lecture, with tiny fake models.

No test loads a model or calls the endpoint. The fakes keep each scorer's interface and
score by shared words, so the right slide must win:
  * FakeEmbed: bag-of-words vectors (embed);
  * FakeSplade: bag-of-words sparse reps (splade);
  * FakeCrossEncoder: the number of shared words of a pair (cross_encoder);
  * FakeColQwen: a slide "image" is the slide's text as bytes; every word is one
    one-hot token vector (image).
Every fake records what it was asked, so the tests can check what each scorer sends.
The tests named `test_fixed_*` pin a cache-key rule (said in each docstring).
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from methods import run_method
from similarity import core
from similarity import models as M
from similarity.cache import SimilarityCache, vector_key
from similarity.registry import (CONFIG_PATH, SCORERS, SETTINGS, Context, Lecture,
                                 MissingSlideImage, SimilarityUnavailable, build_matrix,
                                 key_settings, lecture_from_record, load_settings,
                                 matrix_key, validate_settings)

DIM = 512


def _hash(word: str) -> int:
    return int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM


def _words(text: str) -> list[str]:
    return core.tokenize(text, 3)


class FakeEmbed:
    def __init__(self, offset: float = 0.0):
        self.calls: list[list[str]] = []
        self.offset = offset

    def embed(self, texts):
        self.calls.append(list(texts))
        out = np.full((len(texts), DIM), self.offset, dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _words(t):
                out[i, _hash(w)] += 1.0
        return out


class FakeSplade:
    vocab_size = DIM

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        reps = []
        for t in texts:
            ids = sorted({_hash(w) for w in _words(t)})
            reps.append((np.array(ids, dtype=int), np.ones(len(ids))))
        return reps


class FakeCrossEncoder:
    def __init__(self):
        self.pairs: list[tuple[str, str]] = []

    def score_pairs(self, pairs):
        self.pairs.extend(pairs)
        return np.array([len(set(_words(a)) & set(_words(b))) for a, b in pairs], float)


class FakeColQwen:
    def __init__(self):
        self.images: list[bytes] = []
        self.queries: list[str] = []

    @staticmethod
    def _tokens(text: str) -> np.ndarray:
        ws = _words(text)
        out = np.zeros((max(len(ws), 1), DIM), dtype=np.float32)
        for k, w in enumerate(ws):
            out[k, _hash(w)] = 1.0
        return out

    def encode_images(self, images):
        self.images.extend(images)
        return [self._tokens(im.decode("utf-8")) for im in images]

    def encode_queries(self, texts):
        self.queries.extend(texts)
        return [self._tokens(t) for t in texts]


SLIDES = ("neural networks learn weights",
          "gradient descent optimisation step",
          "convolution kernels detect edges images",
          "")                                     # a slide without text
UNITS = ("the gradient descent takes a step",
         "neural networks learn their weights",
         "convolution kernels slide over images",
         "another gradient descent step",
         "networks and weights again")
GOLD = [1, 0, 2, 1, 0]


def lecture(units=UNITS, slides=SLIDES, images="text") -> Lecture:
    imgs = tuple(s.encode("utf-8") for s in slides) if images == "text" else images
    return Lecture("L1", tuple(units), tuple(slides), imgs)


def fakes() -> dict:
    return {"embed": FakeEmbed(), "splade": FakeSplade(),
            "cross_encoder": FakeCrossEncoder(), "image": FakeColQwen()}


def context(cache=None, overrides=None, models=None, offline=False) -> Context:
    return Context(load_settings(overrides=overrides), cache=cache,
                   models=fakes() if models is None else models, offline=offline)


ALL = sorted(SCORERS)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_the_eight_scorers():
    assert ALL == sorted(["bm25", "splade", "embed", "cross_encoder", "image", "hybrid",
                          "hybrid_splade", "hybrid_image"])
    assert {s.check for s in SCORERS.values()} == {"fine", "fixed", "decided",
                                                  "fixed + decided"}


def test_config_is_complete_and_holds_no_secret():
    s = load_settings()
    assert set(s) == set(SETTINGS)
    text = CONFIG_PATH.read_text()
    assert "api_key" not in text.replace("API key", "")
    assert s["hybrid"]["weight"] == 0.4 and s["embed"]["model"] == "qwen3-embedding-4b"
    assert s["hybrid_image"] == {"inner_weight": 0.4, "outer_weight": 0.4}
    assert (s["bm25"]["min_token_len"], s["bm25"]["remove_stopwords"], s["bm25"]["stem"]) == (2, True, False)


def test_unknown_or_missing_setting_is_an_error():
    with pytest.raises(ValueError, match="no setting"):
        load_settings(overrides={"bm25": {"k2": 1.0}})
    s = load_settings()
    del s["splade"]["batch"]
    with pytest.raises(ValueError, match="missing"):
        validate_settings(s)
    for bad in ({"hybrid": {"weight": 1.5}}, {"cross_encoder": {"first_stage": "image"}},
                {"hybrid_image": {"inner_weight": -1}}, {"bm25": {"stem": "yes"}},
                {"cross_encoder": {"top_k": 0}}):
        with pytest.raises(ValueError):
            load_settings(overrides=bad)


# --------------------------------------------------------------------------- #
# The contract, for every scorer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL)
def test_shape_orientation_and_easy_case(name):
    """Rows are units, columns slides, and each unit's own slide scores highest."""
    S = build_matrix(name, lecture(), context())
    assert S.shape == (len(UNITS), len(SLIDES)) and S.dtype == np.float64
    assert np.isfinite(S).all()
    assert S.argmax(axis=1).tolist() == GOLD


@pytest.mark.parametrize("name", ALL)
def test_methods_read_the_matrix(name):
    """The matrix is in the form methods/ reads: a path method recovers the gold."""
    S = build_matrix(name, lecture(), context())
    assert run_method("argmax", S) == GOLD
    assert len(run_method("dp", S)) == len(UNITS)


@pytest.mark.parametrize("name", ALL)
def test_empty_texts(name):
    """An empty slide, a unit with no word of 3+ letters, and a lecture with no units."""
    units = UNITS + ("ok so",)
    S = build_matrix(name, lecture(units=units), context())
    assert S.shape == (len(units), len(SLIDES)) and np.isfinite(S).all()
    E = build_matrix(name, lecture(units=()), context())
    assert E.shape == (0, len(SLIDES))


def test_empty_text_details():
    ctx = context()
    B = build_matrix("bm25", lecture(units=UNITS + ("ok so",)), ctx)
    assert (B[:, 3] == 0).all()                      # the empty slide matches nothing
    assert (B[-1] == 0).all()                        # the unit with no long word: zeros
    assert np.allclose(B.max(axis=1)[:-1], 1.0)      # rows divided by their maximum
    build_matrix("embed", lecture(), ctx)
    assert " " in ctx.models["embed"].calls[-1]      # an empty slide is sent as " "
    build_matrix("splade", lecture(), ctx)
    assert " " in ctx.models["splade"].calls[-1]


@pytest.mark.parametrize("name", ALL)
def test_deterministic_and_cache_is_exact(name, tmp_path):
    a = build_matrix(name, lecture(), context())
    b = build_matrix(name, lecture(), context())
    assert np.array_equal(a, b)
    cache = SimilarityCache(tmp_path / "c.sqlite3")
    c = build_matrix(name, lecture(), context(cache=cache))
    d = build_matrix(name, lecture(), context(cache=cache, models={}, offline=True))
    assert np.array_equal(a, c) and np.array_equal(c, d)


def test_offline_without_cache_raises_and_bm25_needs_nothing():
    ctx = context(models={}, offline=True)
    for name in ("embed", "splade", "cross_encoder", "image", "hybrid", "hybrid_image"):
        with pytest.raises(SimilarityUnavailable):
            build_matrix(name, lecture(), ctx)
    assert build_matrix("bm25", lecture(), ctx).shape == (5, 4)


# --------------------------------------------------------------------------- #
# Scoring rules, by hand
# --------------------------------------------------------------------------- #

def test_bm25_by_hand():
    k1, b = 1.5, 0.75
    docs = [["alpha", "beta"], ["beta", "beta", "gamma"]]
    avgdl = 2.5
    idf = {"alpha": np.log(1 + (2 - 1 + 0.5) / 1.5), "beta": np.log(1 + 0.5 / 2.5),
           "gamma": np.log(1 + 1.5 / 1.5)}

    def term(t, f, dl):
        return idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))

    s0 = term("alpha", 1, 2) + term("beta", 1, 2)
    s1 = term("beta", 2, 3)
    S_raw = np.array([[s0, s1]])
    S = core.bm25_matrix(["Alpha, beta!"], [" ".join(d) for d in docs], k1=k1, b=b,
                         min_token_len=3)
    assert np.allclose(S, S_raw / S_raw.max())


def test_tokenize():
    assert core.tokenize("AI and ML: 3D-CNNs, don't!", 3) == ["and", "cnns", "don"]
    assert core.tokenize("AI and ML: 3D-CNNs, don't!", 2, remove_stopwords=True) == [
        "ai", "ml", "3d", "cnns"]
    assert core.tokenize("Networks learning", 2, stem=True) == ["network", "learn"]


def test_stopword_list():
    sw = core.stopwords()
    assert len(sw) == 179 and {"the", "and", "of", "don", "t"} <= sw
    assert "NLTK" in core.STOPWORDS_PATH.read_text().splitlines()[1]


def _bm25_default(units, slides):
    return build_matrix("bm25", Lecture("L", tuple(units), tuple(slides)), context())


def test_decided_bm25_keeps_two_letter_tokens():
    """DECIDED: min_token_len 2. With 3, "AI" and "ML" would be dropped and this unit
    would match nothing."""
    S = _bm25_default(["AI and ML"], ["AI ML basics", "deep networks"])
    assert S[0, 0] > 0 and S[0, 1] == 0


def test_decided_bm25_removes_stop_words():
    """DECIDED: stop words do not count: "the", "and", "with" match nothing."""
    S = _bm25_default(["the and with"], ["the and with", "networks"])
    assert (S == 0).all()


def test_decided_bm25_stemming_is_a_setting_off_by_default():
    """DECIDED: Snowball stemming, off by default."""
    units, slides = ["networks learning"], ["network learned", "images"]
    assert (_bm25_default(units, slides) == 0).all()
    ctx = context(overrides={"bm25": {"stem": True}})
    S = build_matrix("bm25", Lecture("L", tuple(units), tuple(slides)), ctx)
    assert S[0, 0] > 0 and S[0, 1] == 0


def test_decided_embed_instruction_on_units_only():
    """DECIDED: the Qwen3 query instruction on the unit side, in the model card's format
    ("Instruct: <task>\nQuery:<text>"); slides plain."""
    ctx = context()
    build_matrix("embed", lecture(), ctx)
    sent = [x for call in ctx.models["embed"].calls for x in call]
    task = load_settings()["embed"]["query_instruction"]
    assert task == ("Given a sentence from a lecture transcript, retrieve the lecture "
                    "slide it refers to")
    assert [x for x in sent if x.startswith("Instruct:")] == [
        f"Instruct: {task}\nQuery:{u}" for u in UNITS]
    assert {x for x in sent if not x.startswith("Instruct:")} == set(SLIDES[:3]) | {" "}


def test_splade_and_maxsim_by_hand():
    units = [(np.array([0, 2]), np.array([1.0, 2.0]))]
    slides = [(np.array([0]), np.array([3.0])), (np.array([2, 3]), np.array([1.0, 5.0]))]
    S = core.splade_matrix(units, slides, 4)
    assert np.allclose(S, np.array([[3.0, 2.0]]) / 3.0)
    q = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    d = [np.array([[0.5, 0.5], [1.0, 0.0]]), np.array([[0.0, 0.2]])]
    assert np.allclose(core.maxsim(q, d), [[1.0 + 0.5, 0.0 + 0.2]])


def test_top_k_ties_go_to_the_lower_column():
    S = np.array([[1.0, 3.0, 3.0, 0.0]])
    assert core.top_k_indices(S, 2).tolist() == [[1, 2]]
    assert core.top_k_indices(S, 9).shape == (1, 4)


def test_merge_topk_reorders_inside_the_band():
    S = np.array([[0.9, 0.1, 0.5, 0.7, -0.2]])
    idx = core.top_k_indices(S, 3)                          # columns 0, 3, 2
    out = core.merge_topk(S, idx, np.array([[1.0, 5.0, 3.0]]))
    assert out[0, 3] == pytest.approx(0.9) and out[0, 0] == pytest.approx(0.5)
    assert out[0, 2] == pytest.approx(0.7)
    assert out[0, 1] == S[0, 1] and out[0, 4] == S[0, 4]    # the rest is untouched
    assert out.max() == S.max() and out.min() == S.min()    # the range is kept
    flat = core.merge_topk(S, idx, np.array([[2.0, 2.0, 2.0]]))
    assert np.array_equal(flat, S)                          # no order: row unchanged


def test_cross_encoder_reranks_only_the_top_k():
    ctx = context(overrides={"cross_encoder": {"top_k": 2}})
    S = build_matrix("cross_encoder", lecture(), ctx)
    H = build_matrix("hybrid", lecture(), ctx)
    pairs = ctx.models["cross_encoder"].pairs
    assert len(pairs) == len(UNITS) * 2
    rest = np.argsort(-H, axis=1, kind="stable")[:, 2:]
    assert np.array_equal(np.take_along_axis(S, rest, 1), np.take_along_axis(H, rest, 1))


# --------------------------------------------------------------------------- #
# Blends
# --------------------------------------------------------------------------- #

def test_blend_math():
    a = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    b = np.array([[3.0, 1.0, 2.0], [1.0, 0.0, 0.0]])
    z = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)  # noqa: E731
    for w in (0.0, 0.4, 1.0):
        assert np.allclose(core.blend(a, b, w), (1 - w) * z(a) + w * z(b))
    assert np.array_equal(core.zscore_rows(a)[1], np.zeros(3))   # a flat row becomes 0
    with pytest.raises(ValueError):
        core.blend(a, b, -0.1)
    with pytest.raises(ValueError):
        core.blend(a, b[:, :2], 0.5)


@pytest.mark.parametrize("name,first,second", [("hybrid", "embed", "bm25"),
                                               ("hybrid_splade", "embed", "splade")])
@pytest.mark.parametrize("w", [0.0, 0.25, 0.4, 1.0])
def test_blend_scorers(name, first, second, w):
    """weight is the share of the second matrix."""
    ctx = context(overrides={name: {"weight": w}})
    lec = lecture()
    got = build_matrix(name, lec, ctx)
    want = core.blend(build_matrix(first, lec, ctx), build_matrix(second, lec, ctx), w)
    assert np.allclose(got, want)


@pytest.mark.parametrize("inner,outer", [(0.4, 0.4), (0.0, 0.7), (0.9, 0.2)])
def test_decided_hybrid_image_has_two_weights(inner, outer):
    """DECIDED: hybrid_image has its own inner (bm25 share of the text blend) and outer
    (image share) weight, so [hybrid].weight does not move hybrid_image."""
    lec = lecture()
    ctx = context(overrides={"hybrid_image": {"inner_weight": inner, "outer_weight": outer},
                             "hybrid": {"weight": 0.9}})
    text = core.blend(build_matrix("embed", lec, ctx), build_matrix("bm25", lec, ctx), inner)
    want = core.blend(text, build_matrix("image", lec, ctx), outer)
    assert np.allclose(build_matrix("hybrid_image", lec, ctx), want)
    s = load_settings()
    t = load_settings(overrides={"hybrid": {"weight": 0.9}})
    assert matrix_key("hybrid_image", lec, s) == matrix_key("hybrid_image", lec, t)


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #

def test_image_reads_the_images_not_the_slide_text():
    imgs = tuple(s.encode() for s in SLIDES)
    lec = Lecture("L1", UNITS, ("x", "y", "z", "w"), imgs)
    ctx = context()
    assert build_matrix("image", lec, ctx).argmax(axis=1).tolist() == GOLD
    assert ctx.models["image"].images == list(imgs)
    assert ctx.models["image"].queries == list(UNITS)


def test_missing_image_is_always_an_error():
    """DECIDED: lectures with a slide without an image are dropped in step 2
    (scripts/prepare.py); the scorer only raises, as the safety net. The step-4 draft's
    "row_min" fill is gone, and so is its setting."""
    imgs = (SLIDES[0].encode(), None, SLIDES[2].encode(), SLIDES[3].encode())
    ctx = context()
    with pytest.raises(MissingSlideImage, match=r"\[2\]"):
        build_matrix("image", lecture(images=imgs), ctx)
    with pytest.raises(MissingSlideImage):
        build_matrix("hybrid_image", lecture(images=imgs), ctx)
    assert ctx.models["image"].images == []                 # nothing was encoded
    with pytest.raises(ValueError, match="no setting"):
        load_settings(overrides={"image": {"missing_image": "row_min"}})
    with pytest.raises(ValueError, match="needs the slide images"):
        build_matrix("image", lecture(images=None), context())


def test_lecture_from_record(tmp_path):
    zp = tmp_path / "c_images.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("L/01.png", b"one")
        zf.writestr("L/02.png", b"two")
    record = {"lecture_id": "L",
              "slides": [{"index": 2, "text": "s2", "image": "L/02.png"},
                         {"index": 1, "text": "s1", "image": "L/01.png"},
                         {"index": 3, "text": "s3", "image": None}],
              "units": [{"index": 1, "text": "u1", "true_slide": 1},
                        {"index": 0, "text": "u0", "true_slide": 2}]}
    lec = lecture_from_record(record, images_zip=zp)
    assert lec.slide_texts == ("s1", "s2", "s3") and lec.unit_texts == ("u0", "u1")
    assert lec.slide_images == (b"one", b"two", None)
    assert lecture_from_record(record).slide_images is None


# --------------------------------------------------------------------------- #
# Cache keys
# --------------------------------------------------------------------------- #

def _perturb(section: str, setting: str, value):
    if setting == "first_stage":
        return "bm25"
    if setting.endswith("weight"):
        return 0.5
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return str(value) + "-x"


def _tree_sections(name, settings) -> set[str]:
    out = {name}
    for d in SCORERS[name].deps(settings):
        out |= _tree_sections(d, settings)
    return out


@pytest.mark.parametrize("name", ALL)
def test_cache_key_covers_every_setting(name):
    """Every setting of the scorer and of everything it is built from changes the key,
    except the transport settings; a setting of an unrelated scorer does not."""
    base = load_settings()
    lec = lecture()
    k0 = matrix_key(name, lec, base)
    tree = _tree_sections(name, base)
    for sec, keys in SETTINGS.items():
        if sec == "cache":
            continue
        for setting in sorted(keys):
            s = load_settings(overrides={sec: {setting: _perturb(sec, setting,
                                                                 base[sec][setting])}})
            changed = matrix_key(name, lec, s) != k0
            expect = sec in tree and setting not in SCORERS[sec].not_in_key
            assert changed == expect, (name, sec, setting)


@pytest.mark.parametrize("name", ALL)
def test_cache_key_covers_the_inputs(name):
    s = load_settings()
    k0 = matrix_key(name, lecture(), s)
    assert matrix_key(name, lecture(units=UNITS[:-1] + ("changed",)), s) != k0
    assert matrix_key(name, lecture(slides=SLIDES[:-1] + ("changed",),
                                    images=tuple(x.encode() for x in SLIDES)), s) != k0
    other_images = tuple(x.encode() for x in SLIDES[:-1]) + (b"png",)
    reads = name in ("image", "hybrid_image")
    assert (matrix_key(name, lecture(images=other_images), s) != k0) == reads


def test_fixed_embed_vectors_are_keyed_by_model(tmp_path):
    """The dense cache key holds the model: after switching the model, the first
    model's vectors must not come back."""
    cache = SimilarityCache(tmp_path / "c.sqlite3")
    a, b = FakeEmbed(), FakeEmbed(offset=0.5)
    A = build_matrix("embed", lecture(), context(cache=cache, models={"embed": a}))
    ctx_b = context(cache=cache, models={"embed": b},
                    overrides={"embed": {"model": "another-embedding-model"}})
    B = build_matrix("embed", lecture(), ctx_b)
    assert b.calls, "the second model was never asked: the first model's vectors were reused"
    assert not np.array_equal(A, B)
    again = FakeEmbed()
    build_matrix("embed", lecture(), context(cache=cache, models={"embed": again}))
    assert again.calls == []                               # same model: all cached


def test_fixed_embed_key_is_the_text_sent():
    """Truncation and the instruction prefix change what is sent, and so the key."""
    long = "word " * 3000
    k = lambda mc: vector_key("m", core.prepare_texts([long], mc)[0])  # noqa: E731
    assert k(6000) != k(100)
    ctx = context(overrides={"embed": {"query_instruction": "Find the slide"}})
    build_matrix("embed", lecture(), ctx)
    sent = [t for call in ctx.models["embed"].calls for t in call]
    assert sum(t.startswith("Instruct: Find the slide\nQuery:") for t in sent) == len(UNITS)
    assert not any(t.startswith("Instruct") for t in sent if t in SLIDES or t == " ")


def test_fixed_cross_encoder_key_includes_its_first_stage(tmp_path):
    """The cross-encoder key holds the settings of the hybrid it reranks: a new
    embedding model, BM25 setting or blend weight there must not return the matrix
    reranked from another first stage."""
    cache = SimilarityCache(tmp_path / "c.sqlite3")
    build_matrix("cross_encoder", lecture(), context(cache=cache))
    for ov in ({"embed": {"model": "another"}}, {"bm25": {"k1": 1.2}},
               {"hybrid": {"weight": 0.6}}):
        ce = FakeCrossEncoder()
        build_matrix("cross_encoder", lecture(),
                     context(cache=cache, overrides=ov,
                             models={**fakes(), "cross_encoder": ce}))
        assert ce.pairs, f"{ov}: the cached matrix of another first stage was reused"


def test_fixed_image_key_includes_the_model(tmp_path):
    """The image matrix key holds the model."""
    cache = SimilarityCache(tmp_path / "c.sqlite3")
    build_matrix("image", lecture(), context(cache=cache))
    enc = FakeColQwen()
    build_matrix("image", lecture(), context(cache=cache, models={"image": enc},
                                             overrides={"image": {"model": "another"}}))
    assert enc.images, "the other model's cached matrix was reused"


def test_key_settings_tree():
    s = load_settings()
    t = key_settings("hybrid_image", s)
    assert [d["scorer"] for d in t["from"]] == ["embed", "bm25", "image"]
    assert "batch" not in t["from"][0]["settings"]       # embed transport
    assert "base_url" not in t["from"][0]["settings"]
    assert t["from"][1]["extra"] == {"stopword_list": core.stopwords_digest()}
    t2 = key_settings("bm25", load_settings(overrides={"bm25": {"stem": True}}))
    assert t2["extra"]["stemmer"].startswith("snowballstemmer ")


def test_bm25_key_covers_the_stopword_list(monkeypatch, tmp_path):
    s = load_settings()
    k0 = matrix_key("cross_encoder", lecture(), s)
    other = tmp_path / "sw.txt"
    other.write_text("the\n")
    monkeypatch.setattr(core, "STOPWORDS_PATH", other)
    assert matrix_key("cross_encoder", lecture(), s) != k0


# --------------------------------------------------------------------------- #
# The real model wrappers (no model is loaded)
# --------------------------------------------------------------------------- #

class _Resp:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def test_embedding_client(monkeypatch):
    sent = []
    fail = {"left": 1}

    def urlopen(req, timeout):
        payload = json.loads(req.data)
        sent.append((req.full_url, req.headers["Authorization"], payload))
        if fail["left"]:
            fail["left"] -= 1
            raise M.urllib.error.HTTPError(req.full_url, 429, "busy", {}, io.BytesIO(b""))
        data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(payload["input"]))]
        return _Resp({"data": data[::-1]})                      # out of order on purpose

    monkeypatch.setattr(M.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    c = M.EmbeddingClient(base_url="https://x/v1/", model="m", key="k", batch=2,
                          requests_per_minute=0)
    out = c.embed(["a", "b", "c"])
    assert out.tolist() == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert [p["input"] for _, _, p in sent] == [["a", "b"], ["a", "b"], ["c"]]
    assert sent[0][0] == "https://x/v1/embeddings" and sent[0][1] == "Bearer k"
    assert sent[0][2]["model"] == "m"


def test_embed_model_uses_the_llm_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_API_BASE", "https://example.org/v1")
    ctx = Context(load_settings())
    client = ctx.model("embed")
    assert isinstance(client, M.EmbeddingClient) and client.key == "secret"
    assert client.url == "https://example.org/v1/embeddings"
    assert client.model == "qwen3-embedding-4b" and client.requests == 0


def test_local_models_load_lazily():
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed")
    except ImportError:
        pass
    enc = M.SpladeEncoder(model="m", max_length=8, batch=2, min_weight=0.0)
    with pytest.raises(M.ModelDependencyMissing):
        enc.encode(["a"])
    with pytest.raises(M.ModelDependencyMissing):
        M.CrossEncoderScorer(model="m", max_length=8, batch=2).score_pairs([("a", "b")])
    with pytest.raises(M.ModelDependencyMissing):
        M.ColQwenEncoder(model="m", image_batch=1, query_batch=1,
                         query_block=4).encode_queries(["a"])

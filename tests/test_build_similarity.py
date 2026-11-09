"""Step 10: scripts/build_similarity.py on a made-up lecture set, with fake models.

No test loads a model or calls the endpoint. The fakes are those of
tests/test_similarity.py (they score by shared words, so the true slide wins); the
embedding client is the real one with a fake transport, sleep and clock, so the cost
rule (docs/TUNING_DESIGN.md 6) is checked on the code that runs on the cluster.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import urllib.error
import zipfile

import numpy as np
import pytest

import build_similarity as B
from similarity import models as M
from test_similarity import (DIM, GOLD, SLIDES, UNITS, FakeColQwen, FakeCrossEncoder,
                             FakeEmbed, FakeSplade, _hash, _words)


# --------------------------------------------------------------------------- #
# A tiny world: one corpus, two tuning lectures and one test lecture
# --------------------------------------------------------------------------- #

def _record(lid: str, units=UNITS, slides=SLIDES, gold=GOLD) -> dict:
    return {"corpus": "toy", "lecture_id": lid,
            "slides": [{"index": j + 1, "text": t, "image": f"{lid}/{j + 1:02d}.png"}
                       for j, t in enumerate(slides)],
            "units": [{"index": i, "text": t, "true_slide": g + 1}
                      for i, (t, g) in enumerate(zip(units, gold))]}


def _write_world(tmp_path, records):
    prepared = tmp_path / "prepared"
    prepared.mkdir(exist_ok=True)
    with gzip.open(prepared / "toy.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    with zipfile.ZipFile(prepared / "toy_images.zip", "w") as zf:
        for r in records:
            for s in r["slides"]:
                zf.writestr(s["image"], s["text"].encode("utf-8"))   # FakeColQwen reads text
    return prepared


@pytest.fixture
def world(tmp_path):
    # L2: the same slides, other units (so its slide vectors come from the cache)
    records = [_record("L1"),
               _record("L2", units=tuple(u + " indeed" for u in UNITS[::-1]), gold=GOLD[::-1]),
               _record("T1")]
    prepared = _write_world(tmp_path, records)
    selection = tmp_path / "selected_lectures.csv"
    with open(selection, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "lecture_id", "set"])
        w.writerows([["toy", "L1", "tuning"], ["toy", "T1", "test"], ["toy", "L2", "tuning"]])
    return {"tmp": tmp_path, "records": records,
            "kw": {"selection": selection, "prepared": prepared,
                   "out_root": tmp_path / "similarity",
                   "overrides": {"cache": {"path": str(tmp_path / "cache.sqlite3")}}}}


def fakes() -> dict:
    return {"embed": FakeEmbed(), "splade": FakeSplade(),
            "cross_encoder": FakeCrossEncoder(), "image": FakeColQwen()}


def quiet(_msg):
    pass


def build(world, measures=B.MEASURES, models=None, **kw):
    return B.build("tuning", measures, models=fakes() if models is None else models,
                   log=quiet, **{**world["kw"], **kw})


def store(world) -> B.Store:
    return B.Store.for_set("tuning", world["kw"]["out_root"])


# --------------------------------------------------------------------------- #
# Build, storage, idempotence
# --------------------------------------------------------------------------- #

def test_the_build_order_lists_all_eight_measures_inputs_first():
    assert sorted(B.MEASURES) == sorted(B.SCORERS)
    s = B.load_settings()
    for k, m in enumerate(B.MEASURES):
        assert all(B.MEASURES.index(d) < k for d in B.SCORERS[m].deps(s))


def test_the_set_is_the_selection_files_set_in_its_order(world):
    assert B.set_lectures("tuning", world["kw"]["selection"]) == [("toy", "L1"), ("toy", "L2")]
    assert B.set_lectures("test", world["kw"]["selection"]) == [("toy", "T1")]
    with pytest.raises(SystemExit):
        B.set_lectures("train", world["kw"]["selection"])


def test_storage_round_trip_is_bit_identical(tmp_path):
    rng = np.random.default_rng(0)
    st = B.Store(tmp_path / "t.sqlite3")
    mats = {"a": rng.normal(size=(7, 3)) * 1e-300, "b": rng.normal(size=(1, 9)),
            "c": np.zeros((0, 4))}
    for name, S in mats.items():
        st.put("bm25", "toy", name, "k", S, 1, {"wall_s": 0.5, "cost_s": 0.5})
    for name, S in mats.items():
        back = st.get("bm25", "toy", name)
        assert back.dtype == np.float64 and back.shape == S.shape
        assert back.tobytes() == S.tobytes()
    assert st.get("bm25", "toy", "missing") is None
    st.close()


def test_the_build_stores_the_matrices_build_matrix_returns(world):
    res = build(world)
    assert all(r == {"built": 2, "reused": 0, "failed": 0, "load_s": r["load_s"]}
               for r in res.values())
    ctx = B.Context(B.load_settings(overrides=world["kw"]["overrides"]), models=fakes())
    st = store(world)
    for m in B.MEASURES:
        for rec in world["records"][:2]:
            lec = B.lecture_from_record(rec, images_zip=world["kw"]["prepared"] /
                                        "toy_images.zip")
            want = B.build_matrix(m, lec, ctx)
            assert st.get(m, "toy", rec["lecture_id"]).tobytes() == want.tobytes()
    assert st.get("bm25", "toy", "T1") is None              # the test set is not touched
    st.close()


def test_a_rerun_reuses_every_table_and_calls_nothing(world):
    build(world)
    st = store(world)
    before = st.rows()
    st.close()
    again = fakes()
    res = build(world, models=again)
    assert all(r["built"] == 0 and r["reused"] == 2 and r["failed"] == 0
               for r in res.values())
    assert again["embed"].calls == [] and again["splade"].calls == []
    assert again["cross_encoder"].pairs == [] and again["image"].images == []
    st = store(world)
    assert st.rows() == before                     # tables and their times unchanged
    st.close()


def test_a_changed_lecture_is_rebuilt_and_only_it(world):
    build(world, ["bm25"])
    recs = world["records"]
    recs[1]["units"][0]["text"] = "a different first unit about gradient descent"
    _write_world(world["tmp"], recs)
    res = build(world, ["bm25"])
    assert res["bm25"]["built"] == 1 and res["bm25"]["reused"] == 1


def test_other_number_settings_stop_the_build(world):
    build(world, ["bm25"])
    ov = {**world["kw"]["overrides"], "bm25": {"k1": 1.2}}
    with pytest.raises(SystemExit, match="another settings"):
        build(world, ["bm25"], overrides=ov)
    with pytest.raises(SystemExit, match="another"):
        build(world, ["bm25"], model_device="cuda")     # the device is in the model keys


def test_transport_settings_do_not_stop_the_build(world):
    build(world, ["bm25"])
    ov = {**world["kw"]["overrides"], "embed": {"requests_per_minute": 5}}
    assert build(world, ["bm25"], overrides=ov)["bm25"]["reused"] == 2


def test_a_missing_input_is_never_built_on_the_side(world):
    """A run is offline for every model it was not given: hybrid without cached vectors
    fails (SimilarityUnavailable) instead of calling the endpoint, and nothing is stored."""
    res = build(world, ["hybrid"], models={"splade": FakeSplade()})
    assert res["hybrid"]["built"] == 0 and res["hybrid"]["failed"] == 2
    st = store(world)
    assert st.rows("hybrid") == []
    st.close()


def test_load_time_is_apart_from_compute(world):
    class Slow(FakeSplade):
        def __init__(self, t):
            super().__init__()
            self.t = t
            self.loads = 0

        def _load(self):
            self.loads += 1
            self.t.now += 30.0

        def encode(self, texts):
            self.t.now += 1.0
            return super().encode(texts)

    t = FakeTime()
    model = Slow(t)
    res = build(world, ["splade"], models={"splade": model}, clock=t.clock)
    assert model.loads == 1 and res["splade"]["load_s"] == 30.0
    st = store(world)
    assert [r["cost_s"] for r in st.rows("splade")] == [2.0, 2.0]   # units + slides
    st.close()
    res = build(world, ["splade"], models={"splade": Slow(t)}, clock=t.clock)
    assert res["splade"]["load_s"] == 0.0                # nothing to build: no load


# --------------------------------------------------------------------------- #
# The cost rule on a fake embedding transport
# --------------------------------------------------------------------------- #

class FakeTime:
    """A clock that only moves when the transport works or the client sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def http_error(code):
    return urllib.error.HTTPError("https://x/v1", code, "err", {}, io.BytesIO(b"busy"))


def vectors(texts) -> bytes:
    rows = []
    for i, t in enumerate(texts):
        v = [0.0] * DIM
        for w in _words(t):
            v[_hash(w)] += 1.0
        rows.append({"index": i, "embedding": v})
    return json.dumps({"data": rows[::-1]}).encode()     # out of order on purpose


def client(t: FakeTime, script, *, retries=3):
    """An EmbeddingClient whose transport plays `script` (seconds the attempt takes, and
    an exception to raise or None to answer); once the script is used up, every request
    takes 1.5 s and succeeds."""
    script = list(script)

    def send(req, timeout):
        secs, err = script.pop(0) if script else (1.5, None)
        t.now += secs
        if err is not None:
            raise err
        return vectors(json.loads(req.data)["input"])

    return M.EmbeddingClient(base_url="https://x/v1", model="m", key="k",
                             max_retries=retries, requests_per_minute=0,
                             transport=send, sleep=t.sleep, clock=t.clock)


def test_only_the_successful_attempt_is_timed():
    t = FakeTime()
    c = client(t, [(7.0, http_error(503)), (4.0, urllib.error.URLError("down")),
                   (2.0, None)])
    out = c.embed(["neural networks", "gradient descent"])
    assert out.shape == (2, DIM) and out[0, _hash("neural")] == 1.0     # index order kept
    s = c.stats()
    assert s["requests"] == 1 and s["request_seconds"] == 2.0 and s["texts_sent"] == 2
    assert s["failed_attempts"] == 2 and s["failed_seconds"] == 11.0
    assert s["errors"] == {"HTTP 503": 1, "URLError": 1}
    assert len(t.sleeps) == 2 and s["wait_seconds"] == pytest.approx(sum(t.sleeps))
    assert t.now == pytest.approx(7 + 4 + 2 + sum(t.sleeps))    # time passed, not counted


def test_a_failed_request_is_never_cost():
    t = FakeTime()
    c = client(t, [(3.0, http_error(429))] * 3, retries=2)
    with pytest.raises(M.EmbeddingCallFailed):
        c.embed(["x"])
    s = c.stats()
    assert s["requests"] == 0 and s["request_seconds"] == 0.0 and s["failed_calls"] == 1
    assert s["failed_attempts"] == 3 and s["failed_seconds"] == 9.0
    t = FakeTime()
    c = client(t, [(1.0, http_error(401))])
    with pytest.raises(M.EmbeddingCallFailed):
        c.embed(["x"])
    assert c.stats()["failed_calls"] == 1 and c.stats()["request_seconds"] == 0.0


def test_the_throttle_wait_is_not_cost():
    t = FakeTime()
    c = M.EmbeddingClient(base_url="https://x/v1", model="m", key="k",
                          requests_per_minute=600,
                          transport=lambda req, to: (setattr(t, "now", t.now + 1.0)
                                                     or vectors(json.loads(req.data)["input"])),
                          sleep=t.sleep, clock=t.clock)
    c.batch = 1
    c.embed(["a b", "c d", "e f"])
    s = c.stats()
    assert s["requests"] == 3 and s["request_seconds"] == 3.0
    assert t.sleeps and s["wait_seconds"] == pytest.approx(sum(t.sleeps))


def test_the_build_records_embed_cost_by_the_rule(world):
    """Through the build: per lecture, cost = the successful requests' durations (the
    local work takes no time on the fake clock); the backoff and the failed attempt are
    recorded apart and are not in the cost."""
    t = FakeTime()
    c = client(t, [(4.0, http_error(429))])       # the very first attempt fails
    res = build(world, ["embed"], models={"embed": c}, clock=t.clock)
    assert res["embed"]["built"] == 2
    st = store(world)
    rows = {r["lecture_id"]: r for r in st.rows("embed")}
    st.close()
    first, second = rows["L1"], rows["L2"]
    assert first["requests"] == 2                                  # units + slides
    assert first["request_s"] == pytest.approx(3.0)
    assert first["cost_s"] == pytest.approx(3.0)
    assert first["failed_attempts"] == 1 and first["failed_s"] == pytest.approx(4.0)
    assert first["wait_s"] == pytest.approx(t.sleeps[0])
    assert first["wall_s"] == pytest.approx(3.0 + 4.0 + t.sleeps[0])
    assert json.loads(first["errors"]) == {"HTTP 429": 1}
    # L2 has the same slides as L1: their vectors come from the cache, so one request
    assert second["requests"] == 1 and second["cost_s"] == pytest.approx(1.5)
    assert second["failed_attempts"] == 0 and second["wait_s"] == 0.0


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #

def test_true_slide_ranks_share_ties():
    S = np.array([[0.9, 0.1, 0.5], [0.0, 0.0, 0.0], [0.2, 0.2, 0.7]])
    assert B.true_slide_ranks(S, [0, 1, 1]).tolist() == [1.0, 2.0, 2.5]


def test_sanity_passes_on_a_good_build_and_catches_broken_tables(world):
    build(world)
    rows = {r["measure"]: r for r in B.sanity("tuning", **world["kw"])}
    assert set(rows) == set(B.MEASURES) and all(r["ok"] for r in rows.values())
    for m in ("bm25", "embed", "hybrid", "cross_encoder"):        # the easy lecture
        assert rows[m]["median_true_rank"] == 1.0 and rows[m]["top3_share"] == 1.0
    assert rows["bm25"]["units"] == 10
    assert rows["bm25"]["chance_top3_share"] == pytest.approx(0.75)
    assert (world["kw"]["out_root"] / "tuning" / "sanity.csv").exists()

    st = store(world)
    S = st.get("bm25", "toy", "L1")
    key_l2 = st.key_of("bm25", "toy", "L2")
    st.put("bm25", "toy", "L1", key_l2, S, 1, {"wall_s": 0, "cost_s": 0})   # another lecture's
    S2 = st.get("embed", "toy", "L2")
    S2[0, 0] = np.nan
    st.put("embed", "toy", "L2", st.key_of("embed", "toy", "L2"), S2, 1,
           {"wall_s": 0, "cost_s": 0})
    st.put("splade", "toy", "L2", st.key_of("splade", "toy", "L2"), S2.T, 1,
           {"wall_s": 0, "cost_s": 0})
    st.conn.execute("DELETE FROM tables WHERE measure='image' AND lecture_id='L1'")
    st.close()
    rows = {r["measure"]: r for r in B.sanity("tuning", **world["kw"])}
    assert not rows["bm25"]["ok"] and rows["bm25"]["order_ok"] == 1
    assert not rows["embed"]["ok"] and rows["embed"]["finite_ok"] == 1
    assert not rows["splade"]["ok"] and rows["splade"]["shape_ok"] == 1
    assert not rows["image"]["ok"] and rows["image"]["tables"] == 1
    assert rows["hybrid"]["ok"]


def test_sanity_uses_the_device_the_set_was_built_with(world):
    build(world, ["splade"], model_device="cuda")       # fakes: the device is only a key
    rows = {r["measure"]: r for r in B.sanity("tuning", measures=["splade"], **world["kw"])}
    assert rows["splade"]["ok"]
    with pytest.raises(SystemExit):
        B.sanity("tuning", model_device="cpu", **world["kw"])


def test_times_adds_the_measures_a_blend_is_built_from(world):
    build(world)
    out = {r["measure"]: r for r in B.times("tuning", out_root=world["kw"]["out_root"])}
    assert out["hybrid"]["deps"] == "embed bm25"
    assert out["hybrid"]["cost_with_deps_s"] == pytest.approx(
        out["hybrid"]["compute_s"] + out["embed"]["compute_s"] + out["bm25"]["compute_s"])
    ce = out["cross_encoder"]
    assert ce["cost_with_deps_s"] == pytest.approx(
        ce["load_s"] + ce["compute_s"] + out["hybrid"]["cost_with_deps_s"])
    assert all(r["lectures"] == 2 for r in out.values())
    assert (world["kw"]["out_root"] / "tuning" / "times.csv").exists()


def test_the_weights_must_be_the_model_the_config_names(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "f.bin").write_bytes(b"w")
    cfg = {"dir": str(tmp_path), "splade": {"repo": "naver/splade-v3-distilbert",
                                            "subdir": "s", "files": {"f.bin": "0" * 64}}}
    s = B.load_settings()
    with pytest.raises(SystemExit, match="not the pinned file"):
        B.verify_weights(cfg, "splade", s)
    cfg["splade"]["files"]["f.bin"] = B.sha256_file(tmp_path / "s" / "f.bin")
    assert B.verify_weights(cfg, "splade", s) == tmp_path / "s"
    cfg["splade"]["repo"] = "other/model"
    with pytest.raises(SystemExit, match="model is"):
        B.verify_weights(cfg, "splade", s)


def test_the_pinned_weights_name_the_configured_models():
    cfg = B.load_weights_config()
    s = B.load_settings()
    for m in B.MODEL_MEASURES:
        assert cfg[m]["repo"] == s[m]["model"]
        assert len(cfg[m]["revision"]) == 40
        assert all(len(h) == 64 for h in cfg[m]["files"].values())


def test_torch_native_jit_is_off_before_torch_loads():
    """Fixed in step 10: with torch's native Triton ops on, every image matrix failed on
    the GPU (no C compiler in the image). similarity.models switches them off at import."""
    import os

    assert os.environ.get("TORCH_DISABLE_NATIVE_JIT") == "1"
    src = (B.REPO_ROOT / "similarity" / "models.py").read_text()
    assert src.index('os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")') < \
        src.index("import torch")


"""Step 9: the two code preparations for the tuning (docs/TUNING_DESIGN.md 2.6 and 6.3).

  * consensus members with their own inputs: ensemble, score_fusion and
    llm_rerank_kbest give each member its own matrix, cfg and settings (and ensemble's
    LLM voter its own LLM, so its own model); without them they run as before;
  * the cost of an LLM call: only the successful attempt's duration counts; the
    throttle and backoff sleeps, failed attempts and failed calls never do, and a cached
    answer brings its stored cost.

No test calls the API: `OpenAICompatClient` gets a fake transport, a fake sleep and a fake
clock, so every duration is exact.
"""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace

import numpy as np
import pytest

from methods import MEMBERS, METHODS, DPConfig, Member, run_method
from methods.consensus import FUSION_MEMBERS, VOTERS, score_fusion_posterior
from methods.llm import rerank_candidates
from methods.llm_client import (LLM, LLMCallFailed, LLMSettings, OpenAICompatClient,
                                ResponseCache, request_key)
from methods.monotone import nw_posterior
from methods.soft_order import cohesion_posterior, softdtw_posterior

from test_llm_methods import GoldClient, ScriptedClient, lecture

CFG2 = replace(DPConfig(), forward_penalty=0.3, stay_bonus=0.2)


def texts(n, m):
    return [f"slide {j}" for j in range(m)], [f"unit {i}" for i in range(n)]


def other_matrix(S, seed=7):
    return np.random.default_rng(seed).normal(size=S.shape)


# --------------------------------------------------------------------------- #
# Consensus members with their own inputs
# --------------------------------------------------------------------------- #

def test_members_are_registry_methods_with_the_same_settings():
    """A member's Member carries the tuned settings of the registry method of that name,
    so every member must run that method's own function."""
    assert set(MEMBERS) == {"ensemble", "score_fusion", "llm_rerank_kbest"}
    for names in MEMBERS.values():
        assert all(n in METHODS for n in names)
    for name, fn in VOTERS.items():
        assert METHODS[name].fn is fn
    for name, (decoder, _) in FUSION_MEMBERS.items():
        assert METHODS[name].fn is decoder
    for method in MEMBERS:
        assert "member_inputs" not in METHODS[method].settings
    assert METHODS["score_fusion"].settings == {"nw_gamma": 0.3}


@pytest.mark.parametrize("empty", [None, {}, "all-none"])
def test_without_member_inputs_nothing_changes(empty):
    S = np.random.default_rng(2).normal(size=(17, 6))
    slides, units = texts(17, 6)
    if empty == "all-none":
        inputs = {name: Member() for name in MEMBERS["ensemble"]}
    else:
        inputs = empty
    # ensemble: as the one-matrix vote (the tests in test_llm_methods.py pin it)
    base = run_method("ensemble", S, cfg=CFG2, slide_texts=slides, unit_texts=units,
                      llm=LLM(model="m", client=GoldClient(run_method("dtw", S))))
    out = run_method("ensemble", S, cfg=CFG2, slide_texts=slides, unit_texts=units,
                     llm=LLM(model="m", client=GoldClient(run_method("dtw", S))),
                     member_inputs=inputs)
    assert out == base
    # score_fusion: the plain sum of the three posteriors at their defaults
    F = (softdtw_posterior(S, cfg=CFG2) + cohesion_posterior(S, cfg=CFG2)
         + nw_posterior(S, cfg=CFG2))
    fusion_in = {n: Member() for n in MEMBERS["score_fusion"]} if empty == "all-none" else inputs
    fused = score_fusion_posterior(S, cfg=CFG2, member_inputs=fusion_in)
    assert np.array_equal(fused, F)
    assert run_method("score_fusion", S, cfg=CFG2) == [int(j) for j in F.argmax(axis=1)]
    # llm_rerank_kbest: the five methods at their defaults on the one matrix
    rerank_in = {n: Member() for n in MEMBERS["llm_rerank_kbest"]} if empty == "all-none" else inputs
    assert rerank_candidates(S, CFG2, rerank_in) == rerank_candidates(S, CFG2)


def test_ensemble_voters_get_their_own_matrix_cfg_and_settings():
    """dp alone against an LLM saying slide 0 everywhere: every disagreement is a 1-1
    tie that goes to dp, so the result is dp's answer - on ITS matrix, cfg and settings."""
    S = np.random.default_rng(2).normal(size=(17, 6))
    S2 = other_matrix(S)
    slides, units = texts(17, 6)
    mine = Member(sim=S2, cfg=CFG2, settings={"no_slide_score": 1.0})
    out = run_method("ensemble", S, slide_texts=slides, unit_texts=units,
                     llm=LLM(model="m", client=GoldClient([0] * 17)), members=("dp",),
                     smooth=False, member_inputs={"dp": mine})
    assert out == run_method("dp", S2, cfg=CFG2, no_slide_score=1.0)
    assert out != run_method("dp", S)


def test_ensemble_llm_voter_gets_its_own_model_settings_and_matrix():
    """The LLM voter asks with ITS LLM (model) and settings; ensemble's own LLM is not
    called. An unusable answer falls back to the argmax of the voter's own matrix, which
    here agrees with dtw and so breaks dp-dtw ties for dtw."""
    S = np.random.default_rng(2).normal(size=(17, 6))
    slides, units = texts(17, 6)
    dp, dtw = run_method("dp", S), run_method("dtw", S)
    assert dp != dtw
    onehot = np.eye(6)[dtw]
    voter_llm = LLM(model="voter-model", client=ScriptedClient("no idea"))
    unused = LLM(model="ensemble-model", client=ScriptedClient(AssertionError("called")))
    diag: dict = {}
    out = run_method("ensemble", S, slide_texts=slides, unit_texts=units, llm=unused,
                     members=("dp", "dtw"), smooth=False, diagnostics=diag,
                     member_inputs={"llm_single_call": Member(
                         sim=onehot, llm=voter_llm, settings={"row_chars": 3})})
    assert out == dtw and diag["parse_failed"] is True
    sent = voter_llm.client.payloads
    assert len(sent) == 1 and sent[0]["model"] == "voter-model"
    assert "unit 0" not in sent[0]["messages"][1]["content"]       # cut to 3 characters
    assert unused.client.payloads == [] and unused.calls == 0


def test_score_fusion_members_get_their_own_inputs_and_nw_gamma_is_its_own():
    S = np.random.default_rng(3).normal(size=(12, 5))
    S2 = other_matrix(S)
    inputs = {"softdtw": Member(sim=S2, settings={"gamma": 0.5, "mode": "path"}),
              "cohesion_dp": Member(cfg=CFG2, settings={"window": 2, "strength": 0.5}),
              "needleman_wunsch": Member(settings={"gap_slide": 0.3})}
    F = score_fusion_posterior(S, nw_gamma=1.0, member_inputs=inputs)
    expect = (softdtw_posterior(S2, gamma=0.5)
              + cohesion_posterior(S, cfg=CFG2, window=2, strength=0.5)
              + nw_posterior(S, gap_slide=0.3, gamma=1.0))
    assert np.array_equal(F, expect)
    assert run_method("score_fusion", S, nw_gamma=1.0, member_inputs=inputs) == \
        [int(j) for j in expect.argmax(axis=1)]


def test_rerank_candidates_get_their_own_inputs():
    S = np.random.default_rng(2).normal(size=(17, 6))
    S2 = other_matrix(S)
    inputs = {"dtw": Member(sim=S2), "pelt": Member(cfg=CFG2, settings={"change_penalty": 3.0})}
    cands, names = rerank_candidates(S, None, inputs)
    assert cands[0] == run_method("dtw", S2)
    pelt = run_method("pelt", S, cfg=CFG2, change_penalty=3.0)
    assert pelt == cands[[i for i, n in enumerate(names) if "pelt" in n.split("+")][0]]
    slides, units = texts(17, 6)
    out = run_method("llm_rerank_kbest", S, slide_texts=slides, unit_texts=units,
                     llm=LLM(model="m", client=ScriptedClient('{"choice": 1}')),
                     member_inputs=inputs)
    assert out == run_method("dtw", S2)


def test_member_inputs_are_checked():
    S = np.random.default_rng(2).normal(size=(9, 4))
    with pytest.raises(ValueError, match="no member"):
        score_fusion_posterior(S, member_inputs={"dtw": Member()})
    with pytest.raises(ValueError, match="no setting"):     # the hard NW has no gamma
        score_fusion_posterior(S, member_inputs={"needleman_wunsch": Member(settings={"gamma": 1})})
    with pytest.raises(ValueError, match="shape"):
        rerank_candidates(S, None, {"dp": Member(sim=S[:, :3])})
    with pytest.raises(ValueError, match="takes no llm"):
        rerank_candidates(S, None, {"dp": Member(llm=LLM(model="m"))})
    with pytest.raises(ValueError, match="no members"):
        run_method("dtw", S, member_inputs={"dtw": Member()})
    slides, units = texts(9, 4)
    with pytest.raises(ValueError, match="no member"):      # not among `members`
        run_method("ensemble", S, slide_texts=slides, unit_texts=units,
                   llm=LLM(model="m", client=GoldClient([0] * 9)), members=("dp",),
                   member_inputs={"pelt": Member()})
    with pytest.raises(ValueError, match="takes no cfg"):
        run_method("ensemble", S, slide_texts=slides, unit_texts=units,
                   llm=LLM(model="m", client=GoldClient([0] * 9)),
                   member_inputs={"llm_single_call": Member(cfg=CFG2)})


# --------------------------------------------------------------------------- #
# The cost of an LLM call
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


def answer(content='{"starts": [0]}', pt=11, ct=3) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": pt, "completion_tokens": ct}}).encode()


def http_error(code):
    return urllib.error.HTTPError("https://x/v1", code, "err", {}, io.BytesIO(b"busy"))


def client(script, t: FakeTime, *, retries=3, rpm=0.0):
    """A OpenAICompatClient whose transport plays `script`: (seconds the attempt takes, then
    the body to return or the exception to raise)."""
    script = list(script)

    def send(req, timeout):
        secs, out = script.pop(0)
        t.now += secs
        if isinstance(out, Exception):
            raise out
        return out

    s = LLMSettings(base_url="https://x/v1", model="m", max_retries=retries,
                    requests_per_minute=rpm)
    return OpenAICompatClient(s, "key", transport=send, sleep=t.sleep, clock=t.clock)


def test_a_retry_then_a_success_counts_only_the_last_attempt():
    t = FakeTime()
    c = client([(7.0, http_error(503)), (4.0, urllib.error.URLError("down")),
                (2.0, answer())], t)
    r = c.chat({"model": "m"})
    assert r["duration_s"] == 2.0 and r["attempts"] == 3 and r["failed_attempts"] == 2
    assert r["errors"] == ["HTTP 503", "URLError"]
    assert len(t.sleeps) == 2 and r["wait_s"] == pytest.approx(sum(t.sleeps))
    assert t.now == pytest.approx(7 + 4 + 2 + sum(t.sleeps))     # time passed, not counted


def test_the_throttle_and_backoff_sleeps_are_not_counted(tmp_path):
    t = FakeTime()
    llm = LLM(model="m", client=client([(2.0, http_error(429)), (1.5, answer()),
                                        (2.5, answer(pt=5, ct=1))], t, rpm=60.0),
              cache=ResponseCache(tmp_path / "c.sqlite3"))
    a = llm.ask("x", "s", "first", max_tokens=8)
    b = llm.ask("x", "s", "second", max_tokens=8)                  # throttled: 1 s apart
    assert any(0 < s <= 1.0 for s in t.sleeps)                     # the throttle slept
    assert (a["duration_s"], a["attempts"]) == (1.5, 2)
    assert (b["duration_s"], b["attempts"]) == (2.5, 1)
    st = llm.stats()
    assert st["calls"] == 2 and st["call_seconds"] == 4.0 and st["used_seconds"] == 4.0
    assert st["failed_attempts"] == 1 and st["errors"] == {"HTTP 429": 1}
    assert st["prompt_tokens"] == 16 and st["failed_calls"] == 0
    stored = llm.cache.get(request_key(llm.payload("s", "first", max_tokens=8)))
    assert stored["duration_s"] == 1.5 and stored["attempts"] == 2


@pytest.mark.parametrize("script,attempts,errors", [
    ([(3.0, http_error(503))] * 3, 3, {"HTTP 503": 3}),           # retries used up
    ([(1.0, http_error(503)), (3.0, http_error(400))], 2, {"HTTP 503": 1, "HTTP 400": 1}),
])
def test_a_failed_call_is_never_cost(tmp_path, script, attempts, errors):
    t = FakeTime()
    cache = ResponseCache(tmp_path / "c.sqlite3")
    llm = LLM(model="m", client=client(script, t, retries=2), cache=cache)
    with pytest.raises(LLMCallFailed) as e:
        llm.ask("x", "s", "u", max_tokens=8)
    assert e.value.attempts == attempts
    st = llm.stats()
    assert st["failed_calls"] == 1 and st["errors"] == errors
    for k in ("calls", "call_seconds", "used_calls", "used_seconds", "prompt_tokens",
              "used_prompt_tokens", "failed_attempts"):
        assert st[k] == 0, k
    assert not cache.path.exists() or cache.get(request_key(llm.payload("s", "u", max_tokens=8))) is None


def test_a_cache_hit_reports_the_stored_cost(tmp_path):
    """The cost belongs to the answer: a method answered from the cache reports the
    duration and tokens of the answer it used, and nothing paid in this run."""
    S, gold, slides, units = lecture(n_per=3, m=4)
    starts = json.dumps({"starts": [3 * j for j in range(4)]})
    cache = ResponseCache(tmp_path / "c.sqlite3")
    t = FakeTime()
    first: dict = {}
    out = run_method("llm_monotone", S, slide_texts=slides, unit_texts=units, diagnostics=first,
                     llm=LLM(model="m", cache=cache,
                             client=client([(5.0, http_error(502)), (2.25, answer(starts, 40, 9))], t)))
    assert out == list(gold)
    assert first["calls"] == 1 and first["call_seconds"] == 2.25
    assert first["used_seconds"] == 2.25 and first["failed_attempts"] == 1
    assert first["errors"] == {"HTTP 502": 1}
    replay: dict = {}
    run_method("llm_monotone", S, slide_texts=slides, unit_texts=units, diagnostics=replay,
               llm=LLM(model="m", client=None, cache=cache))
    assert replay["calls"] == 0 and replay["call_seconds"] == 0 and replay["cache_hits"] == 1
    assert replay["prompt_tokens"] == 0 and replay["errors"] == {}
    assert (replay["used_calls"], replay["used_seconds"], replay["used_prompt_tokens"],
            replay["used_completion_tokens"]) == (1, 2.25, 40, 9)
    assert replay["used_untimed"] == 0

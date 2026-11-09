"""Step 3: the six LLM methods, ensemble's LLM voter, the response cache and the client
configuration.

No test calls the API. `GoldClient` stands in for the model: it reads each request the
way a model would and answers from the true labels, so a correct method recovers them;
`ScriptedClient` returns fixed answers (garbage, truncated JSON, a wrong choice) to test
the fallbacks. Every request a method sends is recorded and checked.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from methods import run_method
from methods.llm import (RERANK_METHODS, HierConfig, hier_config, parse_gaps, parse_starts,
                         rerank_candidates, rescale_counts)
from methods.llm_client import (CONFIG_PATH, LLM, LLMCallFailed, LLMUnavailable,
                                OpenAICompatClient, ResponseCache, api_key, load_settings,
                                request_key)


class GoldClient:
    """A fake model that knows the true slide of every unit."""

    def __init__(self, gold):
        self.gold = [int(g) for g in gold]
        self.payloads: list[dict] = []

    def chat(self, payload: dict) -> dict:
        self.payloads.append(payload)
        user = payload["messages"][1]["content"]
        g = self.gold
        m = int(re.search(r"The deck has (\d+) slides", user).group(1)) if "The deck has" in user else None
        if "Assign every transcript row" in user:
            blocks = [{"from": i, "to": i, "slide": s} for i, s in enumerate(g)]
            ans = {"blocks": blocks}
        elif "Say where each slide STARTS" in user:
            ans = {"starts": [next((i for i, s in enumerate(g) if s >= j), len(g)) for j in range(m)]}
        elif "Say how LONG" in user:
            ans = {"gaps": [sum(1 for s in g if s == j) for j in range(m)]}
        elif "candidate alignments" in user:
            best, choice = -1, 1
            for k, line in re.findall(r"CANDIDATE (\d+): (.*)", user):
                lab = []
                for slide, a, b in re.findall(r"slide (\d+): rows (\d+)-(\d+)", line):
                    lab += [int(slide)] * (int(b) - int(a) + 1)
                agree = sum(x == y for x, y in zip(lab, g))
                if agree > best:
                    best, choice = agree, int(k)
            ans = {"choice": choice}
        elif "TRANSCRIPT PACKETS IN CHRONOLOGICAL ORDER" in user:
            left = user.split("RIGHT PAGE RANGE")[0]
            left_end = max(int(p) for p in re.findall(r"page_index=(\d+)", left))
            last = None
            for pos, a, b in re.findall(r"\[local_packet_pos=(\d+)\].*?\(sections (\d+)-(\d+)\)", user):
                units = g[int(a):int(b) + 1]
                if sum(s <= left_end for s in units) * 2 >= len(units):
                    last = int(pos)
            ans = {"transition_after_local_packet_pos": last}
        elif "TRANSCRIPT SECTIONS:" in user:
            pages = [int(p) for p in re.findall(r"PAGE (\d+):", user)]
            ans = {"assignments": [{"section_idx": int(k), "page_index": min(max(g[int(k)], pages[0]), pages[-1])}
                                   for k in re.findall(r"\[section_idx=(\d+)\]", user)]}
        elif "PREVIOUS PAGE INDEX" in user:
            a = int(re.search(r"PREVIOUS PAGE INDEX:\n(\d+)", user).group(1))
            ans = {"assignments": [{"section_idx": int(k), "belongs_to": "prev" if g[int(k)] <= a else "next"}
                                   for k in re.findall(r"\[section_idx=(\d+)\]", user)]}
        else:
            raise AssertionError("unexpected prompt")
        return {"content": json.dumps(ans), "finish_reason": "stop",
                "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 10}}


class ScriptedClient:
    def __init__(self, *answers, finish_reason="stop"):
        self.answers = list(answers)
        self.finish_reason = finish_reason
        self.payloads: list[dict] = []

    def chat(self, payload):
        self.payloads.append(payload)
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a
        return {"content": a, "finish_reason": self.finish_reason, "usage": {}}


def gold_llm(gold, cache=None) -> LLM:
    return LLM(model="fake-model", client=GoldClient(gold), cache=cache)


def lecture(n_per=6, m=6, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    gold = np.repeat(np.arange(m), n_per)
    S = rng.normal(0, noise, (len(gold), m))
    S[np.arange(len(gold)), gold] += 1.0
    return S, gold, [f"slide text {j}" for j in range(m)], [f"unit text {i}" for i in range(len(gold))]


def run_llm(name, S, slides, units, llm, **kw):
    return run_method(name, S, slide_texts=slides, unit_texts=units, llm=llm, **kw)


LLM_METHODS = ["llm_single_call", "llm_monotone", "hierarchical", "llm_rerank_kbest",
               "fusion", "llm_monotone_constrained", "ensemble"]


@pytest.mark.parametrize("name", LLM_METHODS)
def test_a_model_that_knows_the_answer_is_followed(name):
    S, gold, slides, units = lecture(n_per=7, m=9)
    llm = gold_llm(gold)
    diag: dict = {}
    out = run_method(name, S, slide_texts=slides, unit_texts=units, llm=llm, diagnostics=diag)
    assert np.mean(np.asarray(out) == gold) >= 0.95
    assert diag["calls"] == len(llm.client.payloads) >= 1
    for p in llm.client.payloads:
        assert p["model"] == "fake-model" and p["temperature"] == 0.0
        assert p["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("name", LLM_METHODS)
def test_texts_must_match_the_matrix(name):
    S, gold, slides, units = lecture()
    with pytest.raises(ValueError, match="texts"):
        run_llm(name, S, slides[:-1], units, gold_llm(gold))


@pytest.mark.parametrize("name", ["llm_single_call", "llm_monotone", "llm_monotone_constrained"])
@pytest.mark.parametrize("answer", ["no idea", '{"wrong": 1}', '{"blocks": [{"from": 0, "to"'])
def test_unusable_answers_fall_back_to_the_matrix_argmax(name, answer):
    S, gold, slides, units = lecture()
    diag: dict = {}
    out = run_method(name, S, slide_texts=slides, unit_texts=units,
                     llm=LLM(model="m", client=ScriptedClient(answer)), diagnostics=diag)
    assert out == [int(j) for j in S.argmax(axis=1)] and diag["parse_failed"] is True


def test_rerank_candidates_come_from_five_different_methods():
    """DECIDED: the candidates are dtw, dp, pelt, needleman_wunsch and cohesion_dp, in
    that order, duplicates removed (not dtw's 5 best paths, which are near-copies)."""
    S = np.random.default_rng(2).normal(size=(17, 6))
    cands, names = rerank_candidates(S, None)
    expected = []
    for m in RERANK_METHODS:
        lab = run_method(m, S)
        if lab not in expected:
            expected.append(lab)
    assert cands == expected and names[0].startswith("dtw") and len(cands) >= 3
    S, gold, slides, units = lecture(noise=0.05)                  # all five agree
    llm = LLM(model="m", client=ScriptedClient('{"choice": 1}'))
    diag: dict = {}
    run_method("llm_rerank_kbest", S, slide_texts=slides, unit_texts=units, llm=llm,
               diagnostics=diag)
    user = llm.client.payloads[0]["messages"][1]["content"]
    assert "Below are 1 candidate" in user and "a different alignment method" in user
    assert diag["candidates"] == ["+".join(RERANK_METHODS)]


def test_rerank_returns_the_chosen_candidate_and_dtw_when_unusable():
    S = np.random.default_rng(2).normal(size=(17, 6))
    slides, units = [str(j) for j in range(6)], [str(i) for i in range(17)]
    cands, _ = rerank_candidates(S, None)
    assert len(cands) >= 3
    out = run_llm("llm_rerank_kbest", S, slides, units, LLM(model="m", client=ScriptedClient('{"choice": 3}')))
    assert out == cands[2]
    out = run_llm("llm_rerank_kbest", S, slides, units, LLM(model="m", client=ScriptedClient("??")))
    assert out == run_method("dtw", S)
    with pytest.raises(ValueError, match="no setting"):
        run_llm("llm_rerank_kbest", S, slides, units, LLM(model="m", client=ScriptedClient("??")), k=5)


def test_monotone_starts_are_repaired_into_order():
    assert parse_starts('{"starts": [0, 5, 2, 9]}', 10, 4) == ([0] * 5 + [2] * 4 + [3], False)


def test_constrained_counts_are_rescaled_to_the_transcript():
    """DECIDED: counts are rescaled to add up to n, so counts that overrun the
    transcript do not leave the last slides unreached (unscaled, the labels here would
    stop at slide 1)."""
    labels, failed, raw = parse_gaps('{"gaps": [6, 6, 6, 6]}', 10, 4)
    assert not failed and raw == 24 and labels == [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
    S, gold, slides, units = lecture(n_per=5, m=4)
    diag: dict = {}
    run_method("llm_monotone_constrained", S, slide_texts=slides, unit_texts=units,
               llm=LLM(model="m", client=ScriptedClient('{"gaps": [1, 1, 1, 1]}')),
               diagnostics=diag)
    assert diag["raw_sum"] == 4 and diag["sum_off"] == 4 - 20


def test_rescaling_rule():
    assert rescale_counts([6, 6, 6, 6], 10) == [3, 3, 2, 2]     # largest remainder, ties low
    assert rescale_counts([1, 0, 3], 8) == [2, 0, 6]
    assert rescale_counts([0, 0, 0], 7) == [3, 2, 2]             # all zero: even split
    for c in ([5, 0, 2, 9], [1], [3, 3, 3]):
        assert sum(rescale_counts(c, 17)) == 17


def test_constrained_request_carries_the_schema():
    S, gold, slides, units = lecture(m=5)
    llm = gold_llm(gold)
    run_llm("llm_monotone_constrained", S, slides, units, llm)
    rf = llm.client.payloads[0]["response_format"]
    items = rf["json_schema"]["schema"]["properties"]["gaps"]
    assert rf["type"] == "json_schema" and items["minItems"] == items["maxItems"] == 5


#: Templates that still show a number in their answer shape, left as they are on purpose:
#: RERANK_TEMPLATE's {"choice": 1} (one choice among k, not a list to copy).
NUMBER_EXAMPLE_KEPT = {"RERANK_TEMPLATE"}


def _numbers_in_answer_shapes(text: str) -> list[str]:
    """The numeric values in a template's JSON answer shapes: `"key": 12` and the items
    of a [...] list, except a list's leading 0 after "starts" (the first start must be 0)."""
    text = text.replace("{{", "{").replace("}}", "}")
    found = re.findall(r'"\w+"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    for key, body in re.findall(r'(?:"(\w+)"\s*:\s*)?\[([^\[\]]*)\]', text):
        items = [x.strip() for x in body.split(",")]
        if key == "starts" and items[:1] == ["0"]:
            items = items[1:]
        found += [x for x in items if re.fullmatch(r"-?\d+(?:\.\d+)?", x)]
    return found


def test_answer_shapes_in_the_prompts_show_no_numbers_to_copy():
    """With numbers in the shown answer shape the model copied them (12 of 60
    llm_monotone answers began 0, 12, 30), so the shapes use letters."""
    import methods.llm as L
    templates = {k: v for k, v in vars(L).items()
                 if k.endswith(("TEMPLATE", "PROMPT")) and isinstance(v, str)}
    assert {"SINGLE_CALL_TEMPLATE", "MONOTONE_TEMPLATE", "MONOTONE_CONSTRAINED_TEMPLATE",
            "RERANK_TEMPLATE"} <= set(templates)
    for name, text in templates.items():
        if name not in NUMBER_EXAMPLE_KEPT:
            assert _numbers_in_answer_shapes(text) == [], name
    # the check finds numeric answer shapes and lets the required leading 0 of "starts" through
    assert _numbers_in_answer_shapes('{{"starts": [0, 12, 30, ...]}}') == ["12", "30"]
    assert _numbers_in_answer_shapes('{{"gaps": [12, 18, 0, 7, ...]}}') == ["12", "18", "0", "7"]
    assert _numbers_in_answer_shapes('{"blocks": [{"from": 0, "to": 11, "slide": 0}]}') \
        == ["0", "11", "0"]
    assert _numbers_in_answer_shapes('{"starts": [0, s1, s2, ...]}') == []
    assert _numbers_in_answer_shapes(L.RERANK_TEMPLATE) == ["1"]


# --------------------------------------------------------------------------- #
# Ensemble's LLM voter
# --------------------------------------------------------------------------- #

def test_ensemble_counts_the_llm_vote_equally():
    """DECIDED: llm_single_call is a sixth voter with an equal vote. With members dp and
    dtw and an LLM that agrees with dtw, dtw wins every unit where dp differs (a vote
    without the LLM would give the tie to dp)."""
    S = np.random.default_rng(2).normal(size=(17, 6))
    dp, dtw = run_method("dp", S), run_method("dtw", S)
    assert dp != dtw
    llm = LLM(model="m", client=GoldClient(dtw))
    out = run_method("ensemble", S, slide_texts=[str(j) for j in range(6)],
                     unit_texts=[str(i) for i in range(17)], llm=llm,
                     members=("dp", "dtw"), smooth=False)
    assert out == dtw and len(llm.client.payloads) == 1


def test_ensemble_tie_rule_prefers_the_first_member():
    """One member (dp) against an LLM saying slide 0 everywhere: every disagreement is a
    1-1 tie, and the tie goes to dp."""
    S = np.random.default_rng(2).normal(size=(17, 6))
    llm = LLM(model="m", client=GoldClient([0] * 17))
    out = run_method("ensemble", S, slide_texts=[str(j) for j in range(6)],
                     unit_texts=[str(i) for i in range(17)], llm=llm, members=("dp",),
                     smooth=False)
    assert out == run_method("dp", S)


def test_ensemble_has_no_weights_and_reuses_the_cached_llm_answer(tmp_path):
    S, gold, slides, units = lecture()
    with pytest.raises(ValueError, match="no setting"):
        run_llm("ensemble", S, slides, units, gold_llm(gold), member_weight=2)
    cache = ResponseCache(tmp_path / "c.sqlite3")
    run_llm("llm_single_call", S, slides, units, gold_llm(gold, cache))
    replay = LLM(model="fake-model", client=None, cache=cache)
    run_llm("ensemble", S, slides, units, replay)
    assert replay.calls == 0 and replay.cache_hits == 1


# --------------------------------------------------------------------------- #
# Hierarchical
# --------------------------------------------------------------------------- #

def test_hierarchical_leaf_budget_fits_the_answer():
    """The leaf budget lists every unit: a compact entry is ~18-22 tokens and a leaf
    can hold 8 packets x 20 units = 160 units (64 + 16 per unit, capped at 1200, would
    cut long leaves off)."""
    cfg = HierConfig()
    for n in (10, 71, 160):
        assert cfg.unit_tokens(n) >= 64 + 30 * n
    assert HierConfig(ask_for_reason=True).unit_tokens(50) >= 50 * 100


def test_hierarchical_sizes_from_the_lecture_and_takes_overrides():
    assert hier_config(40, 400).packet_size == 10
    assert hier_config(40, 4000).packet_size == 20
    assert hier_config(40, 400, packet_size=4).packet_size == 4


def test_hierarchical_spreads_a_leaf_it_cannot_parse():
    S, gold, slides, units = lecture(n_per=2, m=3)
    diag: dict = {}
    out = run_method("hierarchical", S, slide_texts=slides, unit_texts=units,
                     llm=LLM(model="m", client=ScriptedClient("garbage")), diagnostics=diag)
    assert out == sorted(out) and diag["parse_failed"] >= 1 and diag["repairs"] >= 1


def test_hierarchical_counts_truncated_answers():
    S, gold, slides, units = lecture(n_per=2, m=3)
    diag: dict = {}
    run_method("hierarchical", S, slide_texts=slides, unit_texts=units,
               llm=LLM(model="m", client=ScriptedClient('{"assignments": [', finish_reason="length")),
               diagnostics=diag)
    assert diag["truncated"] >= 1


# --------------------------------------------------------------------------- #
# The cache and the client
# --------------------------------------------------------------------------- #

def test_a_second_run_is_answered_from_the_cache(tmp_path):
    S, gold, slides, units = lecture(n_per=5, m=8)
    cache = ResponseCache(tmp_path / "c.sqlite3")
    first = run_llm("fusion", S, slides, units, gold_llm(gold, cache))
    replay = LLM(model="fake-model", client=None, cache=cache)       # offline
    diag: dict = {}
    assert run_method("fusion", S, slide_texts=slides, unit_texts=units, llm=replay,
                      diagnostics=diag) == first
    assert diag["calls"] == 0 and diag["cache_hits"] >= 1


def test_any_change_to_the_request_is_a_cache_miss(tmp_path):
    """max_tokens and the text caps are in the key, so changing either asks again
    instead of replaying an earlier answer."""
    S, gold, slides, units = lecture()
    cache = ResponseCache(tmp_path / "c.sqlite3")
    llm = gold_llm(gold, cache)
    for kw in ({}, {}, {"max_tokens": 999}, {"row_chars": 5}, {"slide_chars": 7}):
        run_llm("llm_single_call", S, slides, units, llm, **kw)
    assert llm.calls == 4 and llm.cache_hits == 1
    p = llm.payload("s", "u", max_tokens=10)
    assert request_key(p) != request_key({**p, "max_tokens": 11})
    assert request_key(p) != request_key({**p, "model": "other"})


def test_offline_without_cache_and_failed_calls(tmp_path):
    S, gold, slides, units = lecture()
    with pytest.raises(LLMUnavailable):
        run_llm("llm_monotone", S, slides, units, LLM(model="m", client=None))
    cache = ResponseCache(tmp_path / "c.sqlite3")
    llm = LLM(model="m", client=ScriptedClient(LLMCallFailed("HTTP 503")), cache=cache)
    with pytest.raises(LLMCallFailed):
        run_llm("llm_monotone", S, slides, units, llm)
    assert not cache.path.exists() or cache.get(request_key({})) is None
    assert llm.calls == 0


def test_config_and_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_BASE", "https://example.org/v1")
    s = load_settings(CONFIG_PATH)
    assert s.base_url == "https://example.org/v1" and s.model and s.temperature == 0.0
    monkeypatch.delenv("LLM_API_BASE")
    assert load_settings(CONFIG_PATH).base_url == ""      # an offline replay needs none
    with pytest.raises(LLMUnavailable):
        OpenAICompatClient(load_settings(CONFIG_PATH), "key")
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    assert api_key(tmp_path / "missing.toml") == "from-env"
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(LLMUnavailable):
        api_key(tmp_path / "missing.toml")
    (tmp_path / "cred.toml").write_text('api_key = "from-file"\n')
    assert api_key(tmp_path / "cred.toml") == "from-file"

"""Step 17: the paper's tables and figures (scripts/paper_assets.py, scripts/paper_names.py).

Nothing here reads the results: the display names are checked against the registry and
config/categories.toml, the costs and the tables are made from made-up lectures of the
real 31 methods (with the real tuned settings, which are in git), the worked example's
lecture is chosen among made-up lectures, and nothing is drawn.
"""

from __future__ import annotations

import re
import tomllib

import pytest

import analysis as A
import categorise as C
import final_report as F
import paper_assets as P
import paper_names as N
import tune as T
from methods import METHODS

AXES = A.AXES


# --------------------------------------------------------------------------- #
# The names
# --------------------------------------------------------------------------- #

def possible_levels() -> set[tuple[str, str]]:
    """Every (axis, level) config/categories.toml can give, "other" included."""
    with open(C.CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    fos = C.read_fos(C.REPO_ROOT / cfg["discipline"]["fos_list"])
    out = {("text_source", lv) for lv in cfg["text_source"]["levels"]}
    out |= {(a, lv) for a in C.TERCILE_AXES for lv in cfg["terciles"][a]["levels"]}
    out |= {("tables", lv) for lv in C.TABLE_LEVELS}
    out |= {("discipline", lv) for lv in C.discipline_levels(
        fos, cfg["discipline"]["areas"], cfg["discipline"]["own_level"]) + [C.OTHER]}
    return out


def test_every_method_has_one_distinct_name():
    assert set(N.METHOD_NAMES) == set(METHODS)
    assert len(set(N.METHOD_NAMES.values())) == len(METHODS)


def test_every_level_attribute_and_measure_has_a_name():
    assert possible_levels() <= set(N.LEVEL_NAMES)
    assert set(AXES) <= set(N.AXIS_NAMES)
    assert set(T.MEASURES) <= set(N.MEASURE_NAMES)


# --------------------------------------------------------------------------- #
# Made-up results of the 31 methods
# --------------------------------------------------------------------------- #

CORPORA = list(N.CORPUS_NAMES)
#: One test lecture per corpus (lecture i takes level i of every axis, cycling, so each
#: level of every axis has a lecture) and one tuning lecture per corpus.
TEST = [(c, f"t{i}") for i, c in enumerate(CORPORA)]
TUNING = [(c, f"u{i}") for i, c in enumerate(CORPORA)]
ORDER = {"text_source": ["pdf_text_layer", "corpus_corrected_ocr", "tesseract_ocr"],
         "length": ["short", "medium", "long"], "images": ["low", "medium", "high"],
         "tables": ["none", "some"], "noise": ["low", "medium", "high"],
         "text_density": ["low", "medium", "high"],
         "discipline": ["Computer and information sciences", "Natural sciences (other)",
                        "Engineering and technology", "Medical and health sciences",
                        "Social sciences"]}
#: Model loads: an LLM method with none, the others with one (spread over the lectures).
LOAD = {m: (0.0 if F.kind(m) == "LLM" else 2.5) for m in METHODS}


def lecture_row(m: int, i: int, key) -> dict:
    """Method number m on lecture number i: different numbers per method and lecture."""
    n = 10 + i
    return {"corpus": key[0], "lecture_id": key[1], "status": "ok",
            "n_correct": str((m + 3 * i) % n), "n_units": str(n),
            "compute_s": repr(0.001 * (m + 1) * (i + 1)), "build_cost_s": repr(5.0 + i),
            "used_calls": str(1 + i % 2), "used_seconds": repr(2.0 * m + i),
            "used_prompt_tokens": str(100 * i), "used_completion_tokens": str(10 * m)}


def made_up() -> dict:
    methods = sorted(METHODS)
    kinds = {m: F.kind(m) for m in methods}
    rows = {m: [lecture_row(j, i, k) for i, k in enumerate(TEST)]
            for j, m in enumerate(methods)}
    levels = {a: {k: ORDER[a][i % len(ORDER[a])] for i, k in enumerate(TEST)} for a in AXES}
    cats = {k: {"corpus": k[0], "lecture_id": k[1],
                **{a: ORDER[a][i % len(ORDER[a])] for a in AXES}}
            for i, k in enumerate(TEST + TUNING)}
    accs = {m: {(r["corpus"], r["lecture_id"]): A.lecture_accuracy(r) for r in rs}
            for m, rs in rows.items()}
    q1 = A.q1_rows(rows, kinds, LOAD)
    long = [r for a in AXES for r in A.q2_rows(a, accs, kinds, levels[a], ORDER[a])]
    units = sum(int(r["n_units"]) for r in rows[methods[0]])
    q4 = [{"method": m, **{f"{c}_{s}": (repr(1.5 * j) if s == "pct" else str(units))
                           for c, _, _ in P.Q.COLUMNS for s in ("pct", "units")}}
          for j, m in enumerate(methods)]
    cuts = {"cut_points": {a: {"q1": 12.0, "q2": 19.0, "levels": ORDER[a]}
                           for a in C.TERCILE_AXES}}
    return {"methods": methods, "kinds": kinds, "q1": q1, "order": ORDER,
            "q2": P.q2_cells(long), "rows": rows, "cats": cats, "levels": levels,
            "load_s": LOAD, "q4": q4,
            "selection": [{"corpus": k[0], "lecture_id": k[1], "set": "test"} for k in TEST]
            + [{"corpus": k[0], "lecture_id": k[1], "set": "tuning"} for k in TUNING],
            "attributes": {k: {"n_slides": str(8 + i)} for i, k in enumerate(TEST)},
            "cut_points": cuts, "tuned": T.load_tuned(),
            "costs": P.category_costs(rows, kinds, LOAD, levels)}


# --------------------------------------------------------------------------- #
# The costs
# --------------------------------------------------------------------------- #

def test_cost_of_all_documents_equals_q1_total():
    d = made_up()
    for r in d["q1"]:
        assert d["costs"][r["method"]][P.ALL] == pytest.approx(P.q1_total(r), abs=1e-12)
    assert P.check_costs(d["q1"], d["costs"]) == []
    assert P.check(d) == []


def test_cost_counts_what_the_kind_has_and_spreads_the_load():
    d = made_up()
    lec = TEST[2]
    level = d["levels"]["length"][lec]
    same = [k for k in TEST if d["levels"]["length"][k] == level]
    for m in ("dtw", "llm_monotone", "fusion"):
        rows = {(r["corpus"], r["lecture_id"]): r for r in d["rows"][m]}

        def by_hand(r):
            s = float(r["build_cost_s"]) + LOAD[m] / len(TEST)
            if F.kind(m) != "LLM":
                s += float(r["compute_s"])
            if F.kind(m) != "non-LLM":
                s += float(r["used_seconds"])
            return s

        want = sum(by_hand(rows[k]) for k in same) / len(same)
        assert d["costs"][m][("length", level)] == pytest.approx(want)


def test_the_check_finds_a_cost_that_is_not_q1s():
    d = made_up()
    d["costs"] = P.category_costs(d["rows"], d["kinds"], {}, d["levels"])  # loads left out
    problems = P.check_costs(d["q1"], d["costs"])
    assert len(problems) == sum(1 for m in METHODS if LOAD[m])


# --------------------------------------------------------------------------- #
# The tables
# --------------------------------------------------------------------------- #

def balanced(text: str) -> bool:
    depth = 0
    for ch in re.sub(r"\\[{}%&_#$]", "", text):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


def test_tables_are_sane_latex():
    tables = P.tables(made_up())
    assert set(tables) == {"tab_corpora", "tab_categories", "tab_methods", "tab_q1", "tab_q3",
                           "tab_q4", "tab_q2_full"}
    for name, text in tables.items():
        assert balanced(text), name
        assert text.count(r"\label{") == 1 and text.count(r"\caption{") == 1, name
        envs = re.findall(r"\\(begin|end)\{(\w+)\}", text)
        for env in {e for _, e in envs}:
            assert envs.count(("begin", env)) == envs.count(("end", env)), (name, env)
        assert r"\toprule" in text and r"\bottomrule" in text, name


def test_tables_use_the_display_names():
    tables = P.tables(made_up())
    for name in ("tab_q1", "tab_q2_full", "tab_q4", "tab_methods"):
        for m in METHODS:
            assert P.tex(N.method_name(m)) in tables[name], (name, m)
    for name in ("tab_q1", "tab_q2_full", "tab_q3", "tab_q4", "tab_methods"):
        assert r"\_" not in tables[name], name      # no id or setting like gap_segment left
    assert "Corrected OCR (corpus)" in tables["tab_q3"]
    assert "Computer science" in tables["tab_q3"]


def test_q1_numbers_follow_the_format():
    tables = P.tables(made_up())
    assert P.seconds(22.72) == "22.7" and P.seconds(0.06) == "0.1" and P.seconds(0) == "0.0"
    assert P.seconds(0.0097) == "$<$0.1" and P.seconds(0.0097, latex=False) == "<0.1"
    assert P.seconds(None) == "--"
    assert P.compute_seconds(0.211) == "0.211" and P.compute_seconds(0.0004) == "$<$0.001"
    dtw = next(line for line in tables["tab_q1"].splitlines()
               if re.match(r"\d+ & DTW &", line))
    cells = [c.strip(" \\") for c in dtw.split("&")]
    assert re.fullmatch(r"\d+\.\d", cells[3])        # accuracy, 1 decimal
    assert re.fullmatch(r"\d+\.\d{3}", cells[4])     # compute, 3 decimals
    assert re.fullmatch(r"\d+\.\d", cells[5])        # tables, 1 decimal
    assert cells[6] == cells[7] == "--"              # no LLM cost for a non-LLM method
    assert (P.head_cell("Compute", "(s)") in tables["tab_q1"]
            and P.head_cell("Total", "(s)") in tables["tab_q1"])
    assert "unrounded" in tables["tab_q1"] and "share a rank" not in tables["tab_q1"]


def test_m3av_is_written_with_its_superscript():
    tables = P.tables(made_up())
    assert P.tex(N.CORPUS_NAMES["m3av_a"]) == "M$^3$AV A"
    assert "M3AV" not in tables["tab_corpora"] and "M$^3$AV B" in tables["tab_corpora"]


def test_q3_groups_are_bold_labels_without_inner_rules():
    tables = P.tables(made_up())
    q3 = tables["tab_q3"]
    assert q3.count(r"\midrule") == 1                 # under the header only
    for axis in AXES:
        assert rf"\textbf{{{P.tex(N.AXIS_NAMES[axis])}}}" in q3
    # the caption speaks of a slash only when a tie put one in the table
    assert ("slash" in q3) == (" / " in q3)


def test_q2_full_is_one_table_of_all_levels():
    d = made_up()
    text = P.tables(d)["tab_q2_full"]
    n_levels = sum(len(ORDER[a]) for a in AXES)
    assert n_levels == 22
    assert r"@{\extracolsep{\fill}}l" + "r" * n_levels + "@{}" in text
    assert text.count(r"\begin{tabular*}") == 1
    assert r"\setlength{\tabcolsep}{2pt}" in text and r"\footnotesize" not in text
    assert text.count(r"\rotatebox{90}") == n_levels
    assert r"Corrected\\OCR\end{tabular}}" in text and "Corrected OCR (corpus)" not in text
    rows = [line for line in text.splitlines() if line.endswith(r"\\")
            and line.split(" & ")[0] in {P.tex(N.method_name(m)) for m in METHODS}]
    assert len(rows) == len(METHODS)
    for line in rows:
        cells = [c.strip(" \\") for c in line.split(" & ")]
        assert len(cells) == 1 + n_levels
        assert all(re.fullmatch(r"\d+\.\d", c) for c in cells[1:])


def test_method_table_continues_with_a_marked_head():
    text = P.tables(made_up())["tab_methods"]
    assert "(continued)" in text.split(r"\endfirsthead")[1].split(r"\endhead")[0]
    assert r">{\raggedright\arraybackslash}p{" in text and r"\raggedright " not in text


# --------------------------------------------------------------------------- #
# The scatter's label placement
# --------------------------------------------------------------------------- #

def test_spread_labels_keeps_the_gap_and_the_bounds():
    targets = [5.0, 5.1, 5.2, 20.0, 5.05, 0.0]
    pos = P.spread_labels(targets, 2.0, 0.0, 30.0)
    s = sorted(pos)
    assert all(b - a >= 2.0 - 1e-9 for a, b in zip(s, s[1:]))
    assert all(0.0 <= p <= 30.0 for p in pos)
    assert pos[3] == 20.0                                       # alone: not moved
    assert [targets[i] for i in sorted(range(6), key=lambda i: pos[i])] == sorted(targets)
    assert P.spread_labels([1.0, 10.0], 2.0, 0.0, 30.0) == [1.0, 10.0]


# --------------------------------------------------------------------------- #
# The worked example's lecture
# --------------------------------------------------------------------------- #

def example_inputs():
    """Six made-up lectures: four pass the first filter (one of them on its edges), one
    only the wider one, one neither; per method a mean and per lecture an accuracy."""
    size = {("a", "edge"): (15, 200), ("a", "x"): (20, 150), ("b", "y"): (25, 100),
            ("b", "z"): (18, 120), ("c", "wide"): (28, 250), ("c", "out"): (40, 500)}
    mean = {m: 50.0 + i for i, m in enumerate(P.EXAMPLE_METHODS)}
    off = {("a", "edge"): 9.0, ("a", "x"): 2.0, ("b", "y"): 1.0, ("b", "z"): 1.0,
           ("c", "wide"): 0.0, ("c", "out"): 0.0}
    acc = {m: {k: mean[m] + (o if i % 2 else -o) for k, o in off.items()}
           for i, m in enumerate(P.EXAMPLE_METHODS)}
    return acc, mean, size


def test_example_picks_the_most_typical_lecture_of_the_first_filter():
    acc, mean, size = example_inputs()
    key, filt, score = P.select_example(acc, mean, size)
    # ("b", "y") and ("b", "z") tie at 4 x 1.0 points: the smaller key wins; ("c", "wide")
    # is closer still but only passes the wider filter
    assert (key, filt) == (("b", "y"), 0) and score == pytest.approx(4.0)


def test_example_choice_does_not_depend_on_the_order_of_the_lectures():
    acc, mean, size = example_inputs()
    first = P.select_example(acc, mean, size)
    for _ in range(5):
        keys = list(size)
        keys.reverse()
        assert P.select_example(acc, mean, {k: size[k] for k in keys}) == first
    assert P.select_example(acc, mean, size) == first


def test_example_widens_the_filter_only_when_nothing_passes_the_first():
    acc, mean, size = example_inputs()
    narrow = {k: v for k, v in size.items() if k[0] == "c"}
    key, filt, _ = P.select_example(acc, mean, narrow)
    assert (key, filt) == (("c", "wide"), 1)
    with pytest.raises(ValueError):
        P.select_example(acc, mean, {("c", "out"): size[("c", "out")]})


# --------------------------------------------------------------------------- #
# The numbers the text quotes and the appendix tables
# --------------------------------------------------------------------------- #

def write_gz(path, header, rows):
    import csv
    import gzip
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


#: Two made-up lectures: "a" goes back once (3 -> 2), "b" never; units and labels.
TRUTH = {("mavils", "a"): [1, 1, 2, 3, 2, 3], ("mcd", "b"): [1, 2, 2, 3]}
LABELS = {("mavils", "a"): [2, 1, 2, 0, 2, 1], ("mcd", "b"): [2, 2, 1, 1]}


def unit_files(tmp_path, preds):
    """Prediction files of made-up methods (method -> per lecture its predicted slides)
    and the labels file, as step 16 and step 15 write them."""
    for m, by in preds.items():
        write_gz(tmp_path / f"{m}.csv.gz",
                 ["corpus", "lecture_id", "unit_index", "true_slide", "pred_slide"],
                 [[c, lec, i, t, by[(c, lec)][i]] for (c, lec), ts in TRUTH.items()
                  for i, t in enumerate(ts)])
    write_gz(tmp_path / "labels.csv.gz", ["corpus", "lecture_id", "unit_index", "true_slide",
                                          "label"],
             [[c, lec, i, TRUTH[(c, lec)][i], lab] for (c, lec), ls in LABELS.items()
              for i, lab in enumerate(ls)])
    return tmp_path, tmp_path / "labels.csv.gz"


def test_back_jumps_count_lectures_steps_and_changes():
    bj = P.back_jumps(TRUTH)
    assert (bj["lectures"], bj["back_lectures"]) == (2, 1)
    assert (bj["steps"], bj["back"], bj["changes"]) == (8, 1, 6)
    assert bj["corpus"] == {"mavils": [1, 1], "mcd": [1, 0]}


def test_unit_data_counts_wrong_and_wrong_core_units(tmp_path):
    right = {k: list(v) for k, v in TRUTH.items()}
    off = {("mavils", "a"): [1, 2, 2, 3, 3, 3], ("mcd", "b"): [2, 2, 2, 3]}
    # off is wrong on a1 (label 1), a4 (label 2) and b0 (label 2)
    d, labels = unit_files(tmp_path, {"m1": right, "m2": off})
    ud = P.unit_data(["m1", "m2"], d, labels)
    assert ud["truth"] == TRUTH and ud["units"] == 10 and ud["core"] == 5
    assert ud["labels"] == {2: 5, 1: 4, 0: 1}
    assert ud["wrong"] == {"m1": 0, "m2": 3} and ud["wrong_core"] == {"m1": 0, "m2": 2}
    q4 = {"q4": [{"method": "m2", "all_pct": "20.0"}, {"method": "m1", "all_pct": "0.0"}]}
    rows = P.q4_expected(q4, ud)
    assert [r["method"] for r in rows] == ["m2", "m1"]          # Q4's order kept
    assert rows[0]["error"] == pytest.approx(30.0)
    assert rows[0]["expected"] == pytest.approx(0.5 * 30.0)     # core share 5 / 10
    assert rows[0]["ratio"] == pytest.approx(20.0 / 15.0)


def test_unit_data_refuses_methods_with_other_true_slides(tmp_path):
    d, labels = unit_files(tmp_path, {"m1": TRUTH})
    write_gz(tmp_path / "m2.csv.gz",
             ["corpus", "lecture_id", "unit_index", "true_slide", "pred_slide"],
             [["mavils", "a", 0, 1, 1]])
    with pytest.raises(ValueError):
        P.unit_data(["m1", "m2"], d, labels)


def test_check_units_finds_a_mismatch_and_a_back_jump_in_an_in_order_corpus():
    ud = {"truth": {("mcd", "b"): [1, 3, 2]}, "units": 10, "wrong": {"m": 3},
          "wrong_core": {"m": 2}}
    d = {"rows": {"m": [{"n_units": "10", "n_correct": "8"}]},
         "q4": [{"method": "m", "all_pct": "20.0"}]}
    problems = P.check_units(d, ud)
    assert any("wrong units" in p for p in problems)            # 3 against step 13's 2
    assert any("goes back" in p for p in problems)
    assert not any("core units" in p for p in problems)          # 2 / 10 = 20 %


def test_llm_fallbacks_sum_the_unusable_and_the_cut_off_answers():
    def rows(pf, tr, calls):
        return [{"parse_failed": str(p), "truncated": str(t), "used_calls": str(c)}
                for p, t, c in zip(pf, tr, calls)]
    d = {"rows": {m: rows([0, 0], [0, 0], [1, 1]) for m in P.FALLBACK_METHODS}}
    d["rows"]["llm_single_call"] = rows([1, 0, 1], [1, 0, 0], [1, 1, 1])
    d["rows"]["hierarchical"] = rows([2, 0], [0, 0], [11, 9])
    fb = P.llm_fallbacks(d)
    assert fb["llm_single_call"] == {"lectures": 3, "answers": 3, "fallbacks": 2,
                                     "truncated": 1, "fallback_lectures": 2}
    assert fb["hierarchical"]["fallbacks"] == 2 and fb["hierarchical"]["answers"] == 20
    assert fb["hierarchical"]["fallback_lectures"] == 1


def test_tuning_table_and_its_check():
    td = [{"method": m, "best": 70.0 + i % 3, "chosen": 70.0 + i % 3 - 0.2,
           "test": 67.0 + i % 3} for i, m in enumerate(sorted(METHODS))]
    assert P.check_tuning(td) == []
    bad = [dict(td[0], chosen=td[0]["best"] - 0.7), dict(td[1], chosen=td[1]["best"] + 0.1)]
    assert len(P.check_tuning(bad)) == 2
    assert len(P.check_tuning([dict(td[0], combos=0)])) == 1     # runs short of the grid
    text = P.tab_tuning(td, 60, 140)
    assert balanced(text) and r"\label{tab:tuning}" in text
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    assert len(body) == len(td)                                  # one method per row
    assert all(len(line.split(" & ")) == 5 for line in body)
    assert "$-$2.8" in text                                      # test - chosen, signed


def test_q4_expected_table_has_one_method_per_row_with_the_ratio():
    rows = [{"method": m, "observed": 6.9, "expected": 12.6, "ratio": 6.9 / 12.6}
            for m in sorted(METHODS)]
    text = P.tab_q4_expected(rows, 44.3)
    assert balanced(text) and "44.3" in text and "0.55" in text
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    assert len(body) == len(rows) and body[-1].count(" & ") == 3


def test_prompts_are_the_code_s_text_with_single_braces():
    assert P.prompt_text('{n} rows: {{"a": {last}}}') == '{n} rows: {"a": {last}}'
    from methods import llm as L
    text = P.prompts_tex()
    assert text.count(r"\begin{lstlisting}") == text.count(r"\end{lstlisting}") == 9
    for t in (L.SYSTEM_PROMPT, L.SINGLE_CALL_TEMPLATE, L.MONOTONE_TEMPLATE,
              L.MONOTONE_CONSTRAINED_TEMPLATE, L.RERANK_TEMPLATE, L.HIER_SPLIT_SYSTEM_PROMPT,
              L.HIER_LEAF_SYSTEM_PROMPT):
        assert P.prompt_text(t) in text
    assert "{{" not in text and "{n_slides}" in text and "{spp}" in text


def test_numbers_define_each_macro_once():
    td = [{"method": m, "best": 70.0, "chosen": 70.0, "test": 70.0 - i}
          for i, m in enumerate(sorted(METHODS))]
    q4e = [{"method": m, "observed": 5.0, "error": 30.0, "expected": 13.3, "ratio": 0.38}
           for m in sorted(METHODS)]
    fb = {m: {"lectures": 140, "answers": 140, "fallbacks": 0, "truncated": 0,
              "fallback_lectures": 0} for m in P.FALLBACK_METHODS}
    ud = {"core": 443, "units": 1000}
    text = P.numbers_tex(P.back_jumps(TRUTH), q4e, fb, td, ud, sorted(METHODS)[:14])
    names = re.findall(r"\\newcommand\{\\(\w+)\}", text)
    assert len(names) == len(set(names)) > 20
    assert r"\newcommand{\FallbackOthers}{none of them\xspace}" in text
    assert r"\newcommand{\CorePct}{44.3\xspace}" in text


def test_numbers_take_further_macros_once():
    td = [{"method": m, "best": 70.0, "chosen": 70.0, "test": 70.0 - i}
          for i, m in enumerate(sorted(METHODS))]
    q4e = [{"method": m, "observed": 5.0, "error": 30.0, "expected": 13.3, "ratio": 0.38}
           for m in sorted(METHODS)]
    fb = {m: {"lectures": 140, "answers": 140, "fallbacks": 0, "truncated": 0,
              "fallback_lectures": 0} for m in P.FALLBACK_METHODS}
    ud = {"core": 443, "units": 1000}
    d = made_up()
    dtw = next(r for r in d["q1"] if r["method"] == "dtw")
    more = P.text_numbers(d, td, dtw["build_cost_s_per_lecture"])
    text = P.numbers_tex(P.back_jumps(TRUTH), q4e, fb, td, ud, sorted(METHODS)[:14], more)
    names = re.findall(r"\\newcommand\{\\(\w+)\}", text)
    assert len(names) == len(set(names)) and set(more) <= set(names)
    assert r"\newcommand{\DefaultFwd}{0.05\xspace}" in text
    assert r"\newcommand{\LongUnitsPct}{100\xspace}" in text    # made-up: one count everywhere
    assert "TuneLengthMix" not in more                          # no runs read, no reweighting


def test_over_five_counts_a_drop_that_rounds_to_five():
    td = [{"method": m, "best": 70.0, "chosen": 70.0, "test": 70.0} for m in sorted(METHODS)]
    td[0]["test"], td[1]["test"] = 64.96, 65.06                 # -5.04 and -4.94 points
    fb = {m: {"lectures": 1, "answers": 1, "fallbacks": 0, "truncated": 0,
              "fallback_lectures": 0} for m in P.FALLBACK_METHODS}
    q4e = [{"method": m, "observed": 5.0, "error": 30.0, "expected": 13.3, "ratio": 0.38}
           for m in sorted(METHODS)]
    text = P.numbers_tex(P.back_jumps(TRUTH), q4e, fb, td, {"core": 1, "units": 2},
                         sorted(METHODS)[:14])
    assert rf"\newcommand{{\TuneOverFive}}{{1\xspace}}" in text


def test_macro_names_are_letters_and_unique():
    assert P.camel("LLM-prior DP") == "LLMPriorDP"
    assert P.camel("LLM re-rank (5 methods)") == "LLMReRank"
    assert P.camel("LLM monotone (constrained)") == "LLMMonotoneConstrained"
    assert P.camel("pdf_text_layer") == "PdfTextLayer"
    assert (P.level_macro(("discipline", "Natural sciences (other)"))
            == "DisciplineNaturalSciencesOther")
    names = [P.method_macro(m) for m in sorted(METHODS)]
    assert len(set(names)) == len(names) and all(re.fullmatch("[A-Za-z]+", n) for n in names)
    vals = {"A": "1"}
    with pytest.raises(ValueError):
        P.put(vals, "A", "2")


def test_method_numbers_give_every_method_and_level_once():
    d = made_up()
    md = {"best": {m: {"bm25": 30.0, "image": 40.25} for m in P.MEASURE_METHODS}}
    vals = P.method_numbers(d, md)
    assert all(re.fullmatch("[A-Za-z]+", k) for k in vals)
    q1 = {r["method"]: r for r in d["q1"]}
    assert vals["AccGotoh"] == f"{q1['gotoh']['accuracy']:.1f}"
    assert vals["TuneBestTextTiling"] == "40.2"
    for k, c in d["q2"].items():
        assert vals[f"Highest{P.level_macro(k)}"] == f"{max(c['acc'].values()):.1f}"
    assert "AccScoreFusionLengthShort" in vals and "CostLLMPriorDPLengthLong" in vals


def test_check_numbers_count_the_label_and_discipline_checks(tmp_path):
    import csv

    def write(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([header, *rows])
    write_gz(tmp_path / "labels.csv.gz", ["corpus", "lecture_id", "unit_index", "label"],
             [["mcd", "a", i, x] for i, x in enumerate([2, 1, 1, 0])]
             + [["mcd", "b", i, x] for i, x in enumerate([2, 2])])
    head = ["corpus", "lecture_id", "unit_index", "label"]
    write(tmp_path / "second" / "s.csv", head, [["mcd", "a", 0, 1], ["mcd", "a", 1, 1],
                                               ["mcd", "a", 2, 1], ["mcd", "a", 3, 2]])
    write(tmp_path / "key.csv", ["item"] + head, [[1, "mcd", "a", 1, 1], [2, "mcd", "b", 0, 2]])
    write(tmp_path / "manual.csv", ["item", "corpus", "lecture_id", "unit_index", "label_check"],
          [[1, "mcd", "a", 1, 2], [2, "mcd", "b", 0, 2]])
    write(tmp_path / "raw" / "b01.csv", head,
          [["mcd", "a", i, x] for i, x in enumerate([2, 1, 1, 0])])
    write(tmp_path / "raw" / "b02.csv", head, [["mcd", "b", i, x] for i, x in enumerate([2, 2])])
    write(tmp_path / "disc.csv", ["corpus", "lecture_id", "field_code", "field"],
          [["mcd", "a", "1.2", "CS"], ["mcd", "b", "3.3", "Health"]])
    check = ["corpus", "lecture_id", "assigned_v3", "verdict"]
    write(tmp_path / "check.csv", check, [["mcd", "a", "1.2 CS", "yes"],
                                          ["mcd", "b", "3.3 Health", "near"]])
    vals = P.check_numbers(tmp_path / "second", tmp_path / "key.csv", tmp_path / "manual.csv",
                           tmp_path / "raw", tmp_path / "disc.csv", tmp_path / "check.csv",
                           tmp_path / "labels.csv.gz")
    assert (vals["SecondLectures"], vals["SecondUnits"]) == ("1", "4")
    assert vals["SecondExactPct"] == "50.0"
    assert (vals["SecondWithinPct"], vals["SecondDisagree"]) == ("75.0", "2")
    assert vals["SecondCoreSupporting"] == "1"          # 2 against 1 counts, 0 against 2 not
    assert (vals["BlindUnits"], vals["BlindExact"]) == ("2", "1")
    assert vals["BlindExactPct"] == "50"
    assert (vals["BlindWithin"], vals["BlindDisagree"]) == ("2", "1")
    assert vals["BlindCoreOverSupporting"] == "1"                # checker core, agents supporting
    assert (vals["BatchCoreMin"], vals["BatchCoreMax"]) == ("25", "100")
    assert (vals["DisciplineChecked"], vals["DisciplineRight"], vals["DisciplineNear"],
            vals["DisciplineWrong"]) == ("2", "1", "1", "0")
    write(tmp_path / "check.csv", check, [["mcd", "a", "3.3 Health", "yes"]])
    with pytest.raises(ValueError):                              # checked another label
        P.check_numbers(tmp_path / "second", tmp_path / "key.csv", tmp_path / "manual.csv",
                        tmp_path / "raw", tmp_path / "disc.csv", tmp_path / "check.csv",
                        tmp_path / "labels.csv.gz")


def test_example_numbers_count_the_core_units_on_the_wrong_slide():
    ex = {"filter": 0, "n_slides": 3, "n_units": 4, "truth": [1, 2, 2, 3],
          "labels": [2, 2, 1, 2], "methods": {"llm_single_call": {"pred": [1, 1, 1, 1]}}}
    vals = P.example_numbers(ex)
    assert (vals["ExampleCore"], vals["ExampleSingleCoreWrong"]) == ("3", "2")
    assert (vals["ExampleSlidesFrom"], vals["ExampleUnitsTo"]) == tuple(
        str(x) for x in (P.EXAMPLE_FILTERS[0][0][0], P.EXAMPLE_FILTERS[0][1][1]))


def test_reweight_gives_each_level_its_share():
    acc = {("c", "a"): 0.9, ("c", "b"): 0.7, ("c", "x"): 0.4}
    level = {("c", "a"): "short", ("c", "b"): "short", ("c", "x"): "long"}
    assert P.reweight(acc, level, {"short": 0.5, "long": 0.5}) == pytest.approx(0.6)
    assert P.reweight(acc, level, {"short": 0.25, "medium": 0.0, "long": 0.75}) == \
        pytest.approx(0.25 * 0.8 + 0.75 * 0.4)
    with pytest.raises(ValueError):
        P.reweight(acc, level, {"short": 0.5, "medium": 0.5})


def test_top_band_ends_at_the_first_large_step():
    q1 = [{"accuracy": a} for a in (72.2, 72.1, 71.4, 70.6, 68.0, 60.0)]
    assert P.top_band_end(q1, 1.0) == 4 and P.top_band_end(q1, 0.5) == 2
    assert P.top_band_end(q1, 10.0) == 6
    q1 = [{"accuracy": 80.0 - 0.5 * i} for i in range(P.TOP_BAND)] + [{"accuracy": 50.0}]
    assert P.check_top_band(q1) == []
    assert P.check_top_band(q1[:-2] + [{"accuracy": 50.0}]) != []


def test_q3_marks_a_tie_that_only_rounding_makes():
    cell = {"acc": {"a": 73.84, "b": 73.79, "c": 60.0, "d": 10.04, "e": 10.0},
            "rank": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}}
    assert P.display_tie(cell, lowest=False) and P.display_tie(cell, lowest=True)
    cell["acc"]["b"] = 73.7
    assert not P.display_tie(cell, lowest=False)


def test_dataset_table_averages_to_q1():
    d = made_up()
    da = P.dataset_accuracy(d)
    assert da["n"] == {c: 1 for c in CORPORA}
    assert P.check_datasets(da, d) == []
    text = P.tab_datasets(da, d)
    assert balanced(text) and r"\label{tab:datasets}" in text and "M$^3$AV A" in text
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    assert len(body) == len(METHODS)
    da["dtw"]["mcd"] += 1.0
    assert len(P.check_datasets(da, d)) == 1


# --------------------------------------------------------------------------- #
# The bootstrap over lectures and the distance of the errors
# --------------------------------------------------------------------------- #

def test_percentile_is_linear_between_the_sorted_values():
    assert P.percentile([4.0, 1.0, 3.0, 2.0], 50) == pytest.approx(2.5)
    assert P.percentile([0.0, 10.0], 2.5) == pytest.approx(0.25)
    assert P.percentile([7.0], 97.5) == 7.0
    assert P.interval([float(x) for x in range(101)]) == pytest.approx((2.5, 97.5))


def test_bootstrap_is_seeded_and_paired():
    acc = {"a": [10.0, 50.0, 90.0, 30.0], "b": [11.0, 51.0, 91.0, 31.0]}
    means = P.bootstrap_means(acc, draws=200, seed=7)
    assert means == P.bootstrap_means(acc, draws=200, seed=7)       # same seed, same draws
    assert means != P.bootstrap_means(acc, draws=200, seed=8)
    assert len(means["a"]) == 200 and min(means["a"]) >= 10.0 and max(means["a"]) <= 90.0
    # b is a everywhere plus one point: paired, the difference is 1 on every draw, though
    # each method's own interval is wide
    diffs = [y - x for x, y in zip(means["a"], means["b"])]
    assert all(d == pytest.approx(1.0) for d in diffs)
    lo, hi = P.interval(means["a"])
    assert hi - lo > 20


def test_bootstrap_refuses_methods_on_other_lectures():
    with pytest.raises(ValueError):
        P.bootstrap_means({"a": [1.0, 2.0], "b": [1.0]}, draws=3, seed=0)


def test_bootstrap_table_and_its_check():
    d = made_up()
    bd = P.bootstrap_data(d, draws=50, seed=1)
    assert P.check_bootstrap(bd, d) == []
    ranked = [r["method"] for r in sorted(d["q1"], key=lambda r: r["rank"])]
    assert [r["method"] for r in bd["rows"]] == ranked and bd["ref"] == ranked[0]
    assert bd["rows"][0]["diff"] is None
    assert all(r["ci"][0] <= r["ci"][1] for r in bd["rows"])
    text = P.tab_bootstrap(bd)
    assert balanced(text) and r"\label{tab:bootstrap}" in text
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    assert len(body) == len(METHODS) and all(line.count(" & ") == 5 for line in body)
    assert body[0].endswith(r"-- & -- \\")
    bd["acc"]["dtw"][0] += 1.0
    assert len(P.check_bootstrap(bd, d)) == 1


def test_bootstrap_numbers_count_the_top_band_intervals_that_hold_zero():
    d = made_up()
    bd = P.bootstrap_data(d, draws=50, seed=1)
    top = [r["method"] for r in sorted(d["q1"], key=lambda r: r["rank"])][:P.TOP_BAND]
    vals = P.bootstrap_numbers(bd, top)
    rows = {r["method"]: r for r in bd["rows"]}
    want = sum(rows[m]["diff"][0] <= 0 <= rows[m]["diff"][1] for m in top[1:])
    assert vals["TopBandOverlapZero"] == str(want) and vals["TopBandOthers"] == "13"
    assert vals["BootDraws"] == "50"
    assert all("-" not in vals[k] or "$-$" in vals[k] for k in vals if k.endswith(("Lo", "Hi")))


def test_paired_t_matches_the_hand_computed_p():
    # d = 1, 2, 3, 4: mean 2.5, sd sqrt(5/3), t = 2.5 / (sd / 2) = 3.8730, df 3; the
    # two-sided p is I_{3/(3+t^2)}(1.5, 0.5) = 0.0304663 (scipy.stats.ttest_1samp agrees)
    assert P.paired_t([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.030466291662170977, abs=1e-6)
    assert P.paired_t([0.5, -0.2, 0.3]) == pytest.approx(0.4380485130509837, abs=1e-6)
    assert P.paired_t([-1.0, -2.0, -3.0, -4.0]) == pytest.approx(P.paired_t([1.0, 2.0, 3.0, 4.0]))
    assert P.paired_t([0.0, 0.0, 0.0]) == 1.0 and P.paired_t([2.0, 2.0]) == 0.0
    assert P.reg_beta(1.0, 1.0, 0.3) == pytest.approx(0.3)             # the uniform CDF
    assert P.reg_beta(2.0, 3.0, 0.4) == pytest.approx(0.5248)            # 1 - P(Bin(4, .4) <= 1)


def test_holm_steps_down_and_stops_at_the_first_failure():
    # sorted: .001 <= .05/4, .01 <= .05/3, .03 > .05/2 stops, so .04 is not rejected
    # though it is under .05
    assert P.holm([0.04, 0.001, 0.03, 0.01]) == [False, True, False, True]
    assert P.holm([0.01, 0.02, 0.03]) == [True, True, True]               # .01<=.0167, .02<=.025, .03<=.05
    assert P.holm([0.2, 0.001]) == [False, True]
    assert P.holm([]) == []


def test_significance_table_and_its_check():
    d = made_up()
    sd = P.significance_data(d)
    assert P.check_significance(sd, d) == []
    ranked = sorted(d["q1"], key=lambda r: r["rank"])
    assert [r["method"] for r in sd["rows"]] == [r["method"] for r in ranked]
    assert sd["ref"] == ranked[0]["method"] and sd["rows"][0]["diff"] is None
    acc = {r["method"]: r["accuracy"] for r in ranked}
    for r in sd["rows"][1:]:
        assert r["diff"] == pytest.approx(acc[sd["ref"]] - acc[r["method"]])
        assert 0 <= r["p"] <= 1 and 0 <= r["higher"] <= sd["n"]
    text = P.tab_significance(sd)
    assert balanced(text) and r"\label{tab:significance}" in text
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    assert len(body) == len(METHODS) == 31 and all(line.count(" & ") == 5 for line in body)
    assert body[0].startswith(P.tex(P.N.method_name(sd["ref"])))
    assert body[0].endswith(r"-- & -- & -- & -- \\")
    assert all(line.endswith((r"yes \\", r"no \\")) for line in body[1:])
    vals = P.significance_numbers(sd, [r["method"] for r in ranked][:P.TOP_BAND])
    assert vals["SigTests"] == "30"
    assert int(vals["SigOutsideCount"]) == len(METHODS) - P.TOP_BAND
    sd["rows"][1]["diff"] += 0.1
    sd["rows"] = sd["rows"][:-1]
    assert len(P.check_significance(sd, d)) == 2


def test_unit_data_measures_how_far_the_wrong_units_are(tmp_path):
    right = {k: list(v) for k, v in TRUTH.items()}
    near = {("mavils", "a"): [1, 2, 2, 3, 3, 3], ("mcd", "b"): [2, 2, 2, 3]}
    # far: a0 two slides off, b1 one, b3 two
    far = {("mavils", "a"): [3, 1, 2, 3, 2, 3], ("mcd", "b"): [1, 3, 2, 1]}
    d, labels = unit_files(tmp_path, {"m1": right, "m2": near, "m3": far})
    ud = P.unit_data(["m1", "m2", "m3"], d, labels)
    assert ud["near"] == {"m1": 0, "m2": 3, "m3": 1}
    assert ud["distance"] == {"m1": 0, "m2": 3, "m3": 5}
    sv = P.severity(ud)
    assert sv["m2"] == {"wrong": pytest.approx(30.0), "near": pytest.approx(100.0),
                        "distance": pytest.approx(1.0)}
    assert sv["m3"]["near"] == pytest.approx(100 / 3)
    assert sv["m3"]["distance"] == pytest.approx(5 / 3)
    assert sv["m1"]["wrong"] == 0.0


def test_severity_table_and_numbers():
    d = made_up()
    ranked = [r["method"] for r in sorted(d["q1"], key=lambda r: r["rank"])]
    sv = {m: {"wrong": 20.0 + i, "near": 80.0 - i, "distance": 1.0 + i / 10}
          for i, m in enumerate(ranked)}
    text = P.tab_severity(sv, d)
    assert balanced(text) and r"\label{tab:severity}" in text
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    assert len(body) == len(METHODS) and all(line.count(" & ") == 3 for line in body)
    assert body[0].startswith(P.tex(N.method_name(ranked[0])) + " & 20.0 & 80.0 & 1.00")
    vals = P.severity_numbers(sv, ranked[:P.TOP_BAND])
    assert (vals["TopNearMin"], vals["TopNearMax"]) == ("67", "80")
    alone = [sv[m]["near"] for m in P.LLM_ALONE]
    assert vals["AloneNearMin"] == f"{min(alone):.0f}"
    assert vals["AloneNearMax"] == f"{max(alone):.0f}"


# --------------------------------------------------------------------------- #
# The preparation counts (Table B1)
# --------------------------------------------------------------------------- #

#: Made-up preparation output: per corpus (lectures kept, transcripts cut, units with a
#: true slide before and after the cap, units without one) and its dropped lectures. n1
#: and n2 are M3AV talks in neither half, so both halves' dropped files list them.
PREP_KEPT = {"mavils": (2, 1, 50, 40, 5), "m3av_a": (2, 1, 30, 25, 3),
             "m3av_b": (2, 0, 20, 20, 2), "mcd": (1, 0, 8, 8, 0),
             "slide_highlight": (1, 0, 6, 6, 0)}
PREP_DROPPED = {
    "mavils": [("deck3", "no_ground_truth")],
    "m3av_a": [("a3", "slide_without_image"), ("n1", "fewer_than_5_slides"),
               ("n2", "no_slides")],
    "m3av_b": [("n1", "fewer_than_5_slides"), ("n2", "no_slides")],
    "mcd": [("sp_9", "fewer_than_5_slides")],
    "slide_highlight": [],
}
PREP_SPLIT = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B"}


def write_prepared(root, dropped=PREP_DROPPED, split=PREP_SPLIT):
    """summary.csv, the dropped files and the split, as prepare.py and split_m3av.py
    write them; returns the split's path."""
    import csv
    import prepare as PR
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PR.SUMMARY_FIELDS)
        w.writeheader()
        for c, (kept, cut, before, after, without) in PREP_KEPT.items():
            reasons: dict[str, int] = {}
            for _, x in dropped[c]:
                reasons[x] = reasons.get(x, 0) + 1
            w.writerow({k: "" for k in PR.SUMMARY_FIELDS} | {
                "corpus": c, "lectures_available": kept + len(dropped[c]),
                "lectures_kept": kept, "lectures_dropped": len(dropped[c]),
                "drop_reasons": ";".join(f"{k}={v}" for k, v in sorted(reasons.items())),
                "units_before_cap": before, "units_after_cap": after, "lectures_cut": cut,
                "transcripts_cut": cut, "units_without_slide": without})
    for c, rows in dropped.items():
        with open(root / f"{c}_dropped.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["corpus", "lecture_id", "reason", "n_slides", "detail"])
            w.writerows([c, lec, x, "", ""] for lec, x in rows)
    path = root / "m3av_split.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lecture_id", "version", "stratum"])
        w.writerows([lec, v, "s"] for lec, v in split.items())
    return path


def prep_pool(kept=PREP_KEPT):
    """d as check_prepared reads it: the pool's (corpus, lecture) keys."""
    return {"cats": {(c, f"{c}_{i}"): {} for c, k in kept.items() for i in range(k[0])}}


def test_prepare_table_counts_the_talks_in_neither_half_once(tmp_path):
    split = write_prepared(tmp_path)
    pr = P.read_prepared(tmp_path)
    assert P.check_prepared(pr, prep_pool(), split) == []
    pc = P.prepare_counts(pr)
    assert list(pc["rows"]) == [*CORPORA[:3], P.M3AV_NEITHER, *CORPORA[3:]]
    assert pc["rows"]["m3av_a"]["lectures"] == 3 and pc["rows"]["m3av_b"]["lectures"] == 2
    assert pc["rows"]["m3av_a"]["dropped"] == {"slide_without_image": 1}
    assert pc["rows"]["m3av_b"]["dropped"] == {}
    assert pc["rows"][P.M3AV_NEITHER] == {
        "lectures": 2, "dropped": {"fewer_than_5_slides": 1, "no_slides": 1}, "kept": 0,
        "transcripts_cut": 0, "units_without_slide": 0, "units_cut": 0, "units_kept": 0}
    t = pc["total"]
    # 3 MaViLS + 3 + 2 in the halves + 2 in neither + 2 MCD + 1 Slide-Highlight
    assert t["lectures"] == 13 and t["kept"] == 8
    assert t["lectures"] == t["kept"] + sum(t["dropped"].values())
    assert t["dropped"]["fewer_than_5_slides"] == 2 and t["dropped"]["no_slides"] == 1
    assert (t["transcripts_cut"], t["units_cut"], t["units_kept"]) == (2, 15, 99)
    assert pc["reasons"] == ["no_ground_truth", "no_slides", "fewer_than_5_slides",
                             "slide_without_image"]
    assert P.prepare_numbers(pc) == {"PrepCut": "2", "PrepPool": "8"}
    text = P.tab_prepare(pc)
    assert balanced(text) and r"\label{tab:prepare}" in text
    assert "no labels" in text and "no labelled units" not in text
    body = text.split(r"\midrule")[1].strip().splitlines()
    assert len(body) == 6 and body[3].startswith(P.lines_cell(("M³AV,", "neither half")) + " & 2 & 0 & 1 & 1 & 0")
    assert text.split(r"\midrule")[2].strip().startswith(r"All & 13 & 1 & 1 & 2 & 1 & 8 & 2 &")
    # the same files give the same table
    assert P.tab_prepare(P.prepare_counts(P.read_prepared(tmp_path))) == text


def test_check_prepared_finds_what_does_not_add_up(tmp_path):
    split = write_prepared(tmp_path)
    pr = P.read_prepared(tmp_path)
    # a dropped file that lists other reasons than summary.csv, and one reason unknown
    pr["dropped"]["mcd"][0]["reason"] = "too_quiet"
    got = P.check_prepared(pr, prep_pool(), split)
    assert len(got) == 2 and "mcd_dropped.csv lists" in got[0] and "too_quiet" in got[1]
    # a pool that is not the kept lectures: PrepPool is not PoolLectures
    pr = P.read_prepared(tmp_path)
    got = P.check_prepared(pr, prep_pool(PREP_KEPT | {"mcd": (2,)}), split)
    assert len(got) == 2 and "mcd keeps 1" in got[0] and "PrepPool 8 is not PoolLectures 9" in got[1]
    # a talk in neither half that only one half lists, and one in both that is in a half
    bad = PREP_DROPPED | {"m3av_b": [("n1", "fewer_than_5_slides"), ("a3", "slide_without_image")]}
    split = write_prepared(tmp_path / "b", bad, PREP_SPLIT)
    got = P.check_prepared(P.read_prepared(tmp_path / "b"), None, split)
    assert got == ["prepare: M3AV talk a3 is dropped in both halves but is in half A of the split",
                   "prepare: M3AV talk n2 is in neither half but dropped in one only"]
    # a dropped file that is not there
    (tmp_path / "slide_highlight_dropped.csv").unlink()
    assert P.check_prepared(P.read_prepared(tmp_path), None, split) == [
        "prepare: slide_highlight_dropped.csv is missing"]


def test_without_summary_the_table_is_a_placeholder(tmp_path):
    with pytest.raises(P.PreparedMissing, match="summary.csv is missing"):
        P.read_prepared(tmp_path)
    text = P.tab_prepare_missing()
    assert balanced(text) and r"\label{tab:prepare}" in text and "not in the repository" in text
    assert text.count("&") == 0

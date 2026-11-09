"""Tests for step 5: the attribute definitions (box union and area percentages, the
picture filter, noise, text density), the discipline answer check and its one retry, the validation sample, the
word-list build and the schema of the attributes table.

Nothing loads the layout model or calls the LLM: the boxes are written by hand, the LLM
is a fake client, and every input is a tiny synthetic file.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile

import pytest

import assign_discipline as ad
import attributes as at
import build_wordlist as bw
import detect_layout as dl
from methods.llm_client import LLM

FOS = [{"code": "1.1", "field": "Mathematics", "major_field": "Natural sciences"},
       {"code": "1.2", "field": "Computer and information sciences",
        "major_field": "Natural sciences"},
       {"code": "5.1", "field": "Psychology", "major_field": "Social sciences"}]


# --------------------------------------------------------------------------- box union

def test_union_area_disjoint_boxes_add_up():
    assert at.union_area([[0, 0, 10, 10], [20, 20, 30, 25]], 100, 100) == 100 + 50


def test_union_area_overlap_counts_once():
    # Two 10x10 boxes overlapping in a 5x10 strip: 100 + 100 - 50.
    assert at.union_area([[0, 0, 10, 10], [5, 0, 15, 10]], 100, 100) == 150


def test_union_area_nested_and_duplicate_boxes():
    assert at.union_area([[0, 0, 20, 20], [5, 5, 10, 10], [0, 0, 20, 20]], 100, 100) == 400


def test_union_area_three_way_overlap():
    boxes = [[0, 0, 4, 4], [2, 2, 6, 6], [1, 3, 5, 7]]
    # Brute force on the unit grid (all corners are integers).
    cells = {(x, y) for x0, y0, x1, y1 in boxes for x in range(x0, x1) for y in range(y0, y1)}
    assert at.union_area(boxes, 10, 10) == len(cells)


def test_union_area_clips_to_the_image_and_drops_empty_boxes():
    assert at.union_area([[-5, -5, 5, 5]], 10, 10) == 25
    assert at.union_area([[8, 8, 20, 20]], 10, 10) == 4
    assert at.union_area([[3, 3, 3, 9], [20, 20, 30, 30]], 10, 10) == 0
    assert at.union_area([], 10, 10) == 0


def _slide(w, h, *boxes):
    return {"width": w, "height": h,
            "boxes": [{"class": c, "box": b, "score": 0.9} for c, b in boxes]}


def test_area_pct_pools_area_over_slides_and_separates_classes():
    slides = [_slide(10, 10, ("picture", [0, 0, 10, 5]), ("picture", [0, 0, 10, 5]),
                     ("table", [0, 5, 10, 10])),
              _slide(20, 10, ("text", [0, 0, 20, 10]))]
    # picture: 50 px of 100 + 200 px = 300 px; the duplicate box counts once.
    assert at.area_pct(slides, "picture") == pytest.approx(100 * 50 / 300)
    assert at.area_pct(slides, "table") == pytest.approx(100 * 50 / 300)
    assert at.area_pct(slides, "chart") == 0.0
    assert at.area_pct([_slide(10, 10, ("picture", [-1, -1, 11, 11]))], "picture") == 100.0


# --------------------------------------------------------------------------- picture filter

RULES = {"repeat_share": 0.5, "repeat_iou": 0.7, "tiny_pct": 1.0}


def _drops(slides):
    return at.picture_drops(slides, "picture", repeat_share=0.5, repeat_iou=0.7, tiny_pct=1.0)


def _lecture(n, pictures):
    """n slides of 1024x1024 px; pictures: slide index -> picture boxes on it.

    1024 is a power of two, so the normalised coordinates and the IoU at the boundary
    are exact in floating point.
    """
    return [_slide(1024, 1024, *(("picture", b) for b in pictures.get(i, [])))
            for i in range(n)]


def test_box_iou():
    assert at.box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert at.box_iou([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1 / 3)
    assert at.box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    assert at.box_iou([1, 1, 1, 1], [1, 1, 1, 1]) == 0.0


def test_repeat_rule_drops_a_box_on_half_the_slides_and_counts_its_own():
    logo = [900, 0, 1024, 124]
    # On 2 of 4 slides (the box's own slide counted): 50 % -> dropped on both.
    assert _drops(_lecture(4, {0: [logo], 2: [logo]})) == [["repeat"], [], ["repeat"], []]
    # On every slide, shifted a little (IoU well above 0.7): dropped everywhere.
    shifted = {i: [[900 + i, 0, 1024, 124]] for i in range(5)}
    assert _drops(_lecture(5, shifted)) == [["repeat"]] * 5


def test_repeat_rule_keeps_a_box_on_fewer_than_half_the_slides():
    fig = [100, 100, 600, 600]
    # 2 of 5 slides = 40 %: kept.
    assert _drops(_lecture(5, {0: [fig], 3: [fig]})) == [[None], [], [], [None], []]
    # Different places on every slide (no pair at IoU >= 0.7): all kept.
    moving = {i: [[100 * i, 0, 100 * i + 300, 300]] for i in range(5)}
    assert _drops(_lecture(5, moving)) == [[None]] * 5


def test_repeat_rule_counts_a_slide_once_and_only_picture_boxes():
    logo = [900, 0, 1024, 124]
    # Three copies on slide 0 are one slide of 5 (20 %, plus the copies' own slide): kept.
    assert _drops(_lecture(5, {0: [logo, logo, logo]})) == [[None] * 3, [], [], [], []]
    # A table in the same place on the other slides does not make the picture repeat.
    slides = _lecture(4, {0: [logo]})
    for s in slides[1:]:
        s["boxes"].append({"class": "table", "box": logo, "score": 0.9})
    assert _drops(slides) == [[None], [None], [None], [None]]


def test_repeat_rule_iou_boundary_and_normalised_coordinates():
    a = [0, 0, 640, 1024]       # normalised width 0.625
    b = [0, 0, 448, 1024]       # 0.4375 inside it: IoU = 0.4375 / 0.625 = 0.7 exactly
    c = [0, 0, 447, 1024]       # IoU just under 0.7
    assert at.box_iou([0, 0, 0.625, 1], [0, 0, 0.4375, 1]) == 0.7
    assert _drops(_lecture(4, {0: [a], 1: [b]})) == [["repeat"], ["repeat"], [], []]
    assert _drops(_lecture(4, {0: [a], 1: [c]})) == [[None], [None], [], []]
    # The same place on slides of different pixel sizes: compared in 0..1 coordinates.
    slides = [_slide(1000, 500, ("picture", [0, 0, 100, 50])),
              _slide(2000, 1000, ("picture", [0, 0, 200, 100])),
              _slide(1000, 500), _slide(1000, 500)]
    assert _drops(slides) == [["repeat"], ["repeat"], [], []]


def test_repeat_rule_share_boundary_with_an_odd_number_of_slides():
    fig = [100, 100, 600, 600]
    # 5 slides: 3 (60 %) is at least half, 2 (40 %) is not.
    assert _drops(_lecture(5, {0: [fig], 1: [fig], 2: [fig]}))[:3] == [["repeat"]] * 3
    assert _drops(_lecture(5, {0: [fig], 1: [fig]}))[:2] == [[None]] * 2


def test_tiny_rule_drops_boxes_under_one_percent_of_the_slide():
    # A 100x100 slide: 1 % is 100 px. 99 px is dropped, 100 px is kept.
    def one(box):
        return _drops([_slide(100, 100, ("picture", box))] + [_slide(100, 100)] * 4)[0]
    assert one([0, 0, 9, 11]) == ["tiny"]                  # 99 px
    assert one([0, 0, 10, 10]) == [None]                   # 100 px: exactly 1 %, kept
    assert one([0, 0, 50, 50]) == [None]
    assert one([-5, -5, 5, 15]) == ["tiny"]                # 5 x 15 = 75 px after clipping
    # Tables are never filtered, however small.
    assert _drops([_slide(100, 100, ("table", [0, 0, 1, 1]))]) == [[None]]


def test_repeat_comes_before_tiny():
    icon = [0, 0, 50, 50]                                   # 0.24 % of 1024x1024
    marks = _drops(_lecture(4, {0: [icon], 1: [icon], 3: [[500, 500, 510, 510]]}))
    assert marks == [["repeat"], ["repeat"], [], ["tiny"]]


def test_keep_pictures_removes_dropped_boxes_and_counts_them():
    logo, fig, dot = [900, 0, 1024, 124], [100, 100, 600, 600], [10, 10, 20, 20]
    slides = _lecture(4, {0: [logo, fig], 1: [logo, dot], 2: [logo]})
    slides[3]["boxes"].append({"class": "table", "box": [0, 0, 1024, 512], "score": 0.9})
    kept, n = at.keep_pictures(slides, "picture", RULES)
    assert n == {"repeat": 3, "tiny": 1}
    assert [[b["box"] for b in s["boxes"]] for s in kept] == [[fig], [], [], [[0, 0, 1024, 512]]]
    assert at.area_pct(kept, "picture") == pytest.approx(100 * 500 * 500 / (4 * 1024 * 1024))
    assert at.area_pct(kept, "table") == at.area_pct(slides, "table")


# --------------------------------------------------------------------------- noise, density

WORDS = frozenset({"the", "model", "learns", "data", "is", "in"})


def test_noise_tokens_letters_only_lowercased_min_length_two():
    assert at.noise_tokens("The MODEL, x=3.5; a GPU-based") == ["the", "model", "gpu", "based"]
    assert at.noise_tokens("e.g. 42 ∑ a b") == []          # single letters, digits, symbols
    assert at.noise_tokens("naïve") == ["na", "ve"]         # letters are A-Z/a-z only


def test_noise_pct_pools_all_slides():
    texts = ["The model learns", "data xqzt", "", "3 4 5"]
    # tokens: the model learns data xqzt -> 1 of 5 not in the list
    assert at.noise_counts(texts, WORDS) == (1, 5)
    assert at.noise_pct(texts, WORDS) == pytest.approx(20.0)
    assert at.noise_pct(["", "1 2"], WORDS) == 0.0


def test_text_density_counts_tokens_with_a_letter_or_digit():
    assert at.slide_words("Results: 42 % - • | x2") == 3      # Results: 42 x2
    assert at.text_density(["a b c", "", "d - e"]) == pytest.approx((3 + 0 + 2) / 3)
    assert at.text_density([]) == 0.0


def test_wordlist_loader_skips_the_header(tmp_path):
    p = tmp_path / "w.txt.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("# header\n# more\nalpha\nbeta\n\n")
    assert at.load_wordlist(p) == frozenset({"alpha", "beta"})


def test_build_wordlist_selects_sizes_parts_and_letters(tmp_path):
    files = {
        "scowl-2020.12.07/final/english-words.10": "the\nModel\ndon't\n",
        "scowl-2020.12.07/final/american-words.70": "color\n",
        "scowl-2020.12.07/final/british-words.80": "colour\n",          # size 80: out
        "scowl-2020.12.07/final/english-proper-names.50": "Alice\n",
        "scowl-2020.12.07/final/english-contractions.10": "isn't\n",    # part: out
        "scowl-2020.12.07/final/english-words.35": "na\xefve\n",        # not a-z: out
        "scowl-2020.12.07/final/australian-words.10": "arvo\n",         # spelling: out
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in files.items():
            data = text.encode("latin-1")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    words, used = bw.build(buf.getvalue())
    assert words == ["alice", "color", "model", "the"]
    assert "english-words.35" in used and "british-words.80" not in used
    out = tmp_path / "w.txt.gz"
    bw.write(words, used, out)
    assert at.load_wordlist(out) == frozenset(words)
    first = out.read_bytes()
    bw.write(words, used, out)
    assert out.read_bytes() == first                    # fixed gzip timestamp


# --------------------------------------------------------------------------- discipline

class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.payloads = []

    def chat(self, payload):
        self.payloads.append(payload)
        return {"content": self.answers.pop(0), "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _llm(answers):
    client = FakeClient(answers)
    return LLM(model="fake-model", client=client, cache=None), client


REC = {"lecture_id": "CHI-1", "title": None, "meta": {"venue_prefix": "CHI"},
       "slides": [{"index": i, "text": f"slide {i} text"} for i in range(1, 11)],
       "units": [{"text": " ".join(f"w{u}_{j}" for j in range(60))} for u in range(4)]}
SELECT = {"n_slides": 3, "n_spread": 3, "slide_chars": 100, "transcript_words": 150}


def test_parse_answer_by_code_by_name_and_invalid():
    assert ad.parse_answer('{"code": "1.2", "field": "x"}', FOS)[0]["code"] == "1.2"
    assert ad.parse_answer('```json\n{"code": " 5.1 "}\n```', FOS)[0]["field"] == "Psychology"
    assert ad.parse_answer('{"code": "9.9", "field": "mathematics"}', FOS)[0]["code"] == "1.1"
    row, problem = ad.parse_answer('{"code": "9.9", "field": "Astrology"}', FOS)
    assert row is None and "9.9" in problem
    assert ad.parse_answer("Computer science, I think.", FOS) == (None, "no JSON object")


def test_spread_positions_are_the_centres_of_equal_parts():
    assert ad.spread_positions(7, 3) == [1, 3, 5]
    assert ad.spread_positions(6, 3) == [1, 3, 5]
    assert ad.spread_positions(37, 3) == [6, 18, 30]
    assert ad.spread_positions(2, 3) == [0, 1]          # fewer left than asked: all of them
    assert ad.spread_positions(0, 3) == []
    assert ad.spread_positions(5, 0) == []


def test_select_slides_first_three_then_three_spread_over_the_rest():
    pick = [s["index"] for s in ad.select_slides(REC["slides"], 3, 3)]
    assert pick == [1, 2, 3, 5, 7, 9]                   # the rest is slides 4..10
    assert pick == [s["index"] for s in ad.select_slides(REC["slides"], 3, 3)]
    assert [s["index"] for s in ad.select_slides(REC["slides"][:4], 3, 3)] == [1, 2, 3, 4]
    assert [s["index"] for s in ad.select_slides(REC["slides"][:2], 3, 3)] == [1, 2]


def test_transcript_opening_takes_the_first_words_across_units():
    words = ad.transcript_opening(REC["units"], 150).split()
    assert len(words) == 150 and words[0] == "w0_0" and words[-1] == "w2_29"
    assert ad.transcript_opening([{"text": " a  b\n"}, {"text": None}, {"text": "c"}], 150) \
        == "a b c"
    assert ad.transcript_opening([], 150) == ""


def test_prompt_holds_venue_selected_slides_transcript_and_every_field():
    p = ad.build_prompt(REC, FOS, **{**SELECT, "slide_chars": 8})
    assert "Venue code (from the corpus): CHI" in p
    assert all(f"Slide {i}: slide {i}" in p for i in (1, 2, 3, 5, 7, 9))
    assert not any(f"Slide {i}:" in p for i in (4, 6, 8, 10))
    assert "Slide 1: slide 1 " not in p                 # cut to 8 chars
    assert "Transcript (first 150 words): w0_0 " in p and "w2_29\n" in p and "w2_30" not in p
    assert "not the application area it is applied to" in p
    assert "Label the discipline of the work itself (its methods and contribution)" in p
    assert all(f"{r['code']}: {r['field']}" in p for r in FOS)
    assert "Title" not in p
    assert "Title: A talk" in ad.build_prompt({**REC, "title": "A talk"}, FOS, **SELECT)
    assert "Transcript" not in ad.build_prompt({**REC, "units": []}, FOS, **SELECT)


VENUES = {"CHI": "the Conference on Human Factors in Computing Systems (CHI) 2021, paper "
                  "presentations (human–computer interaction research talks)"}


def test_prompt_describes_a_known_venue_and_sends_an_unknown_code_bare():
    p = ad.build_prompt(REC, FOS, **SELECT, venues=VENUES)
    assert f"Venue (from the corpus): {VENUES['CHI']}\n" in p
    assert "Venue code" not in p
    # A code with no entry, or no venues at all: the bare code, the request as before.
    ipp = {**REC, "meta": {"venue_prefix": "IPP"}}
    assert "Venue code (from the corpus): IPP\n" in ad.build_prompt(ipp, FOS, **SELECT,
                                                                     venues=VENUES)
    assert ad.build_prompt(ipp, FOS, **SELECT, venues=VENUES) == ad.build_prompt(ipp, FOS,
                                                                                 **SELECT)
    # No venue code (the other corpora): the venues change nothing.
    plain = {**REC, "meta": {}}
    assert ad.build_prompt(plain, FOS, **SELECT, venues=VENUES) == ad.build_prompt(plain, FOS,
                                                                                   **SELECT)
    assert "Venue" not in ad.build_prompt(plain, FOS, **SELECT, venues=VENUES)


def test_venues_in_the_repo_config_describe_the_five_m3av_codes():
    import tomllib
    with open(ad.CONFIG, "rb") as fh:
        venues = tomllib.load(fh)["discipline"]["venues"]
    assert set(venues) == {"CHI", "IPP", "MLS", "NIH", "Ubi"}
    assert all(v.strip() for v in venues.values())
    # Only the venue is described: no OECD field code is written into it. (Field names
    # are not checked: "Oxford Mathematics" is the venue's own name.)
    codes = {r["code"] for r in ad.load_fos(ad.REPO_ROOT / "config" / "oecd_fos2007.csv")}
    assert not any(codes & set(v.split()) for v in venues.values())


def test_assign_valid_first_answer():
    llm, client = _llm(['{"code": "1.2", "field": "Computer and information sciences"}'])
    r = ad.assign(llm, REC, FOS, max_tokens=50, **SELECT)
    assert (r["field_code"], r["field"], r["attempts"]) == ("1.2", FOS[1]["field"], 1)
    p = client.payloads[0]
    assert p["temperature"] == 0.0 and p["response_format"] == {"type": "json_object"}
    assert p["model"] == "fake-model"


def test_assign_retries_once_with_a_different_request():
    llm, client = _llm(['{"code": "7.7"}', '{"code": "5.1"}'])
    r = ad.assign(llm, REC, FOS, max_tokens=50, **SELECT)
    assert (r["field"], r["attempts"]) == ("Psychology", 2)
    assert len(client.payloads) == 2
    first, second = (p["messages"][1]["content"] for p in client.payloads)
    assert second.startswith(first) and "not valid" in second
    assert [a["content"] for a in r["answers"]] == ['{"code": "7.7"}', '{"code": "5.1"}']


def test_assign_two_invalid_answers_make_it_unassigned():
    llm, client = _llm(["no idea", '{"code": "0"}', '{"code": "1.1"}'])
    r = ad.assign(llm, REC, FOS, max_tokens=50, **SELECT)
    assert (r["field"], r["field_code"], r["attempts"]) == (ad.UNASSIGNED, "", 2)
    assert r["raw_answer"] == '{"code": "0"}'
    assert len(client.payloads) == 2                      # no third call


def test_fos_list_in_the_repo_is_the_42_fields():
    fos = ad.load_fos(ad.REPO_ROOT / "config" / "oecd_fos2007.csv")
    assert len(fos) == 42
    assert len({r["major_field"] for r in fos}) == 6
    assert {"1.2", "2.2", "6.4"} <= {r["code"] for r in fos}


def test_blocked_without_a_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    import methods.llm_client as lc
    # api_key's default path is bound at definition time: point it at a missing file.
    monkeypatch.setattr(lc.api_key, "__defaults__", (tmp_path / "none.toml",))
    monkeypatch.setattr(ad, "OUT_DIR", tmp_path / "out")
    assert ad.main([]) == ad.BLOCKED_EXIT
    assert "discipline blocked: no API key" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- validation sample

def test_quotas_proportional_with_a_minimum():
    sizes = {"mavils": 652, "m3av_a": 10313, "m3av_b": 10152, "mcd": 250,
             "slide_highlight": 150}
    q = dl.quotas(sizes, 100, 10)
    assert sum(q.values()) == 100
    assert q["mavils"] == q["mcd"] == q["slide_highlight"] == 10
    assert q["m3av_a"] == q["m3av_b"] == 35
    assert dl.quotas({"a": 5, "b": 1000}, 20, 10) == {"a": 5, "b": 15}


def _lines(corpus, n, n_table):
    out = []
    for i in range(n):
        boxes = [{"class": "table", "box": [0, 0, 1, 1], "score": 0.9}] if i < n_table else []
        out.append({"corpus": corpus, "lecture": f"L{i // 10}", "slide": i % 10 + 1,
                    "width": 10, "height": 10, "boxes": boxes})
    return out


def test_sample_is_seeded_stratified_and_oversamples_tables():
    lines = _lines("mavils", 200, 3) + _lines("m3av_a", 2000, 100) + _lines("mcd", 50, 0)
    kw = dict(n=40, seed=7, minimum=10, table_share=0.4, table_class="table")
    s = dl.sample_slides(lines, **kw)
    assert len(s) == 40
    by = {}
    for r in s:
        by.setdefault(r["corpus"], []).append(r["stratum"])
    assert {c: len(v) for c, v in by.items()} == {"mavils": 10, "m3av_a": 20, "mcd": 10}
    assert by["mavils"].count("table") == 3               # all it has
    assert by["m3av_a"].count("table") == 8               # ceil(0.4 * 20), > 5 % base rate
    assert by["mcd"].count("table") == 0
    assert len({(r["corpus"], r["lecture"], r["slide"]) for r in s}) == 40
    assert [(r["corpus"], r["lecture"], r["slide"]) for r in dl.sample_slides(lines, **kw)] == \
        [(r["corpus"], r["lecture"], r["slide"]) for r in s]


# --------------------------------------------------------------------------- the table

def _record(corpus, lid, slide_texts, unit_texts, source="tesseract_ocr", venue=None):
    return {"corpus": corpus, "lecture_id": lid, "title": None, "text_source": source,
            "meta": {"venue_prefix": venue} if venue else {},
            "slides": [{"index": i, "text": t, "image": f"{lid}/{i:02d}.png"}
                       for i, t in enumerate(slide_texts, 1)],
            "units": [{"index": i, "text": t, "true_slide": 1}
                      for i, t in enumerate(unit_texts)]}


def _write_inputs(tmp_path, discipline=True, layout_extra=None, drop_layout=False):
    prepared, out = tmp_path / "prepared", tmp_path / "attributes"
    prepared.mkdir()
    out.mkdir()
    recs = {"mavils": [_record("mavils", "lecA", ["The model learns", "data xqzt 42", ""],
                               ["so the model", "learns from data now"], "pdf_text_layer")],
            "m3av_a": [_record("m3av_a", "CHI-1", ["in the data"], ["hello there"],
                               "corpus_corrected_ocr", "CHI")]}
    for c in at.CORPORA:
        with gzip.open(prepared / f"{c}.jsonl.gz", "wt") as fh:
            for r in recs.get(c, []):
                fh.write(json.dumps(r) + "\n")
    logo = {"class": "picture", "box": [0, 9, 1, 10], "score": 0.9}   # on every slide
    lines = [{"corpus": "mavils", "lecture": "lecA", "slide": 1, "width": 10, "height": 10,
              "boxes": [{"class": "picture", "box": [0, 0, 10, 5], "score": 0.8},
                        {"class": "picture", "box": [0, 0, 5, 5], "score": 0.7}, logo]},
             {"corpus": "mavils", "lecture": "lecA", "slide": 2, "width": 10, "height": 10,
              "boxes": [{"class": "table", "box": [0, 0, 10, 10], "score": 0.9},
                        {"class": "picture", "box": [5, 5, 5.5, 5.5], "score": 0.9}, logo]},
             {"corpus": "mavils", "lecture": "lecA", "slide": 3, "width": 10, "height": 10,
              "boxes": [logo]},
             {"corpus": "m3av_a", "lecture": "CHI-1", "slide": 1, "width": 20, "height": 10,
              "boxes": []}]
    if drop_layout:
        lines = lines[1:]
    lines += layout_extra or []
    with gzip.open(out / "layout_boxes.jsonl.gz", "wt") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    if discipline:
        with open(out / "discipline.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["corpus", "lecture_id", "field_code", "field", "attempts", "raw_answer"])
            w.writerow(["mavils", "lecA", "1.2", "Computer and information sciences", 1, "{}"])
            w.writerow(["m3av_a", "CHI-1", "", "unassigned", 2, "x"])
    (tmp_path / "config").mkdir()
    with gzip.open(tmp_path / "config" / "w.txt.gz", "wt") as fh:
        fh.write("# list\n" + "\n".join(sorted(WORDS)) + "\n")
    cfg = {"layout": {"threshold": 0.6, "image_class": "picture", "table_class": "table"},
           "picture_filter": dict(RULES), "noise": {"wordlist": "config/w.txt.gz"}}
    return cfg, prepared, out


def test_table_schema_and_values(tmp_path):
    cfg, prepared, out = _write_inputs(tmp_path)
    rows, record = at.build(cfg, prepared=prepared, out_dir=out, repo_root=tmp_path)
    assert [list(r) for r in rows] == [at.COLUMNS] * 2
    a, b = rows
    assert (a["corpus"], a["lecture_id"], a["text_source"]) == ("mavils", "lecA", "pdf_text_layer")
    assert (a["n_slides"], a["n_units"], a["n_transcript_words"]) == (3, 2, 7)
    # 50 of 300 px, overlap once; the logo (every slide) and the 0.25 px box are dropped.
    assert a["image_pct"] == pytest.approx(100 * 50 / 300, abs=1e-4)   # rounded to 4
    assert a["table_pct"] == pytest.approx(100 * 100 / 300, abs=1e-4)
    assert a["noise"] == pytest.approx(20.0)                # xqzt of 5 tokens
    assert a["text_density"] == pytest.approx(2.0)          # (3 + 3 + 0) / 3
    assert a["discipline"] == "Computer and information sciences"
    assert (b["image_pct"], b["table_pct"], b["noise"], b["discipline"]) == \
        (0.0, 0.0, 0.0, "unassigned")
    assert record["discipline_status"] == "assigned"
    assert record["pictures_dropped"]["mavils"] == {"repeat": 3, "tiny": 1}
    assert record["pictures_dropped"]["m3av_a"] == {"repeat": 0, "tiny": 0}
    path = at.write(rows, record, out)
    with open(path, newline="") as fh:
        got = list(csv.DictReader(fh))
    assert list(got[0]) == at.COLUMNS and len(got) == 2
    assert json.loads((out / "document_attributes.json").read_text())["rows"] == 2


def test_table_blocked_discipline_is_empty(tmp_path):
    cfg, prepared, out = _write_inputs(tmp_path, discipline=False)
    rows, record = at.build(cfg, prepared=prepared, out_dir=out, repo_root=tmp_path)
    assert [r["discipline"] for r in rows] == [None, None]
    assert record["discipline_status"].startswith("blocked")
    at.write(rows, record, out)
    with open(out / "document_attributes.csv", newline="") as fh:
        assert [r["discipline"] for r in csv.DictReader(fh)] == ["", ""]


def test_table_stops_on_missing_layout_or_unknown_lecture(tmp_path):
    cfg, prepared, out = _write_inputs(tmp_path, drop_layout=True)
    with pytest.raises(ValueError, match="no layout"):
        at.build(cfg, prepared=prepared, out_dir=out, repo_root=tmp_path)
    extra = [{"corpus": "mcd", "lecture": "ghost", "slide": 1, "width": 1, "height": 1,
              "boxes": []}]
    (tmp_path / "b").mkdir()
    cfg, prepared, out = _write_inputs(tmp_path / "b", layout_extra=extra)
    with pytest.raises(ValueError, match="not kept"):
        at.build(cfg, prepared=prepared, out_dir=out, repo_root=tmp_path / "b")


def test_validate_rejects_bad_rows():
    row = {"corpus": "mcd", "lecture_id": "x", "text_source": "tesseract_ocr", "n_slides": 5,
           "n_units": 3, "n_transcript_words": 30, "image_pct": 10.0, "table_pct": 0.0,
           "noise": 5.0, "text_density": 12.0, "discipline": "Psychology"}
    at.validate([row], [("mcd", "x")], False)
    with pytest.raises(ValueError, match="outside"):
        at.validate([{**row, "image_pct": 100.5}], [("mcd", "x")], False)
    with pytest.raises(ValueError, match="missing"):
        at.validate([{**row, "noise": float("nan")}], [("mcd", "x")], False)
    with pytest.raises(ValueError, match="missing"):
        at.validate([{**row, "discipline": None}], [("mcd", "x")], False)
    at.validate([{**row, "discipline": None}], [("mcd", "x")], True)
    with pytest.raises(ValueError, match="more than once"):
        at.validate([row, row], [("mcd", "x")], False)
    with pytest.raises(ValueError, match="missing"):
        at.validate([row], [("mcd", "x"), ("mcd", "y")], False)

# Step 5: the lecture attributes

One table, `data/attributes/document_attributes.csv`, one row per kept lecture (the
1131 lectures of `data/prepared/`), one column per attribute. Step 6 builds the
categories from it. The attributes describe each lecture *as prepared*: the first 40
slides at most, and the transcript up to the cap (README, step 2).

## The columns

| column | in plain words |
|---|---|
| `corpus` | mavils, m3av_a, m3av_b, mcd or slide_highlight |
| `lecture_id` | the lecture's id in its corpus |
| `text_source` | where the slide text comes from: `pdf_text_layer` (MaViLS PDFs), `tesseract_ocr` (our OCR: MCD, Slide-Highlight, one scanned MaViLS deck), `corpus_corrected_ocr` (M3AV's hand-corrected OCR) |
| `n_slides` | kept slides (at most 40) |
| `n_units` | kept transcript units (sentences, speech segments, lines or subtitle cues, as the corpus gives them) |
| `n_transcript_words` | words in the kept units: whitespace-separated tokens, summed |
| `image_pct` | how much of the lecture's slide area is pictures, in % (below) |
| `table_pct` | the same for tables |
| `noise` | how much of the slide text is not English words, in % (below) |
| `text_density` | how many words a slide carries, on average (below) |
| `discipline` | the lecture's field of science, one of the 42 OECD FOS 2007 fields, or `unassigned`; empty while the discipline step is blocked |

`data/attributes/document_attributes.json` records the definitions, the hashes of every
input, the discipline status and, per corpus, the median, minimum and maximum of each
number.

### image_pct and table_pct

A layout-detection model finds the boxes of pictures and tables on every slide image
(next section). For one lecture:

    image_pct = 100 x (sum over its slides of the area covered by its kept picture boxes)
                    / (sum over its slides of the full image area, width x height)

"Area covered" is the area of the **union** of the boxes: where two picture boxes
overlap, the overlap counts once (boxes are clipped to the image first). Boxes of
different classes are not merged with each other: a table inside a picture counts in
both. `table_pct` is the same with the table boxes. A slide weighs by its pixel count;
within a lecture the slides almost always have one size, so each weighs about equally
(across lectures sizes differ: M3AV frames are mostly 1280x720, MaViLS renders range from
756x567 to 2134x1600, which does not matter for a ratio taken per lecture).

**M3AV's slide images are video frames**: the frame the corpus gives for each slide
segment, which may show the speaker, a picture-in-picture, the room or a cropped slide.
A speaker's video inset is a picture to the detector and counts in `image_pct`. In the
first run 91-92 % of the M3AV frames had at least one picture box, against 56 % of the
MaViLS pages; the validation sheet shows whether these are slide pictures or video
content. MaViLS
images are the PDF pages rendered at 150 dpi; MCD and Slide-Highlight images are the
corpora's own slide images.

**Which picture boxes are kept.** The check of the 100-slide sheet (below) found
the picture boxes wrong on 72 of the 100 slides: logos and watermarks (44), the speaker's
camera inset in M3AV frames (32 of the 33 frames that show one), small icons and glyphs
(13), and video-player or beamer control bars (6); only 4 real figures were missed, and
the table boxes were right on all 100. So two rules drop picture boxes before the union
(tables are not filtered): the **repeat** rule drops a picture box if a picture box in the
same place (IoU >= 0.7 in page coordinates normalised to 0..1) is on at least 50 % of the
lecture's slides, each slide counted once and the box's own slide included; logos,
insets and bars sit in one place on most slides, figures rarely do. The **tiny** rule
then drops a picture box smaller than 1 % of the slide area. The thresholds are in
`config/attributes.toml` (`[picture_filter]`); `document_attributes.json` records how
many boxes each rule dropped per corpus, and `data/attributes/validation_100_v2.pdf`
(`scripts/detect_layout.py sheet_v2`) shows the same 100 slides with the kept picture
boxes in blue and the dropped ones in grey, dashed, labelled "repeat" or "tiny".

### noise

The share of the slide text that a spell checker would not accept, which in practice
is mostly OCR damage, plus formulas, code, identifiers, acronyms and non-English text.

- **Token**: a maximal run of the letters A-Z / a-z in the slide text, lowercased, of
  length 2 or more. Digits, punctuation and symbols separate tokens and are never
  tokens; single letters are not tokens (variables and list markers would otherwise
  count either way depending on the list). A letter outside A-Z (é, ü, Greek) splits the
  word, and its pieces are usually not in the list.
- **Word list**: `config/wordlist_en_scowl70.txt.gz`, 131,341 words, built by
  `scripts/build_wordlist.py` from SCOWL 2020.12.07 (Kevin Atkinson,
  http://wordlist.aspell.net/, the source of aspell's and hunspell's English
  dictionaries; the tarball is pinned by sha256 and the licence notice is in the file's
  header): sizes 10-70 (70 is SCOWL's "large" size), the english, american and british
  spellings, the parts words, capitalised words, proper names and abbreviations,
  lowercased, a-z only. Names and abbreviations are in because slides are full of them
  (author names, "fig", "kg").
- **noise** = 100 x (tokens not in the list) / (all tokens), pooled over all the
  lecture's slides. A lecture with no token at all would get 0 (none of the 1131 has
  none).

### text_density

The mean number of slide-text words per slide: the whitespace-separated tokens holding
at least one letter or digit (so bare bullets, dashes and '|' from OCR do not count),
summed over the lecture's slides and divided by `n_slides`. Slides without text count
as 0 words.

### discipline

One field of the OECD Fields of Science and Technology (FOS 2007), second level: 42
fields in 6 major fields (`config/oecd_fos2007.csv`; source:
OECD 2007, *Revised Field of Science and Technology (FOS) Classification in the Frascati
Manual*, DSTI/EAS/STP/NESTI(2006)19/FINAL; see `config/oecd_fos2007.md`).

The corpora carry no field metadata, so the LLM (the endpoint of `LLM_API_BASE`, the default model of
`config/llm.toml`, temperature 0, thinking off) reads, per lecture: its title if the
corpus has one (none does at present), the M3AV venue code (CHI, IPP, MLS, NIH, Ubi)
the text of its first 3 slides plus 3 more spread evenly over the rest of the deck (at the
centres of three equal parts; each slide cut to 1200 characters), and the first 150 words
of its transcript. The first slides alone are often a title, a biography, a CME notice or
nearly empty. The counts are settings (`config/attributes.toml`, `[discipline]`), and the
selection is fixed arithmetic, so a rerun sends the same request. It gets the full list
of 42 fields as "code: name" and answers in JSON, `{"code": ..., "field": ...}`. The
prompt says: "Label the discipline of the work itself (its methods and contribution),
not the application area it is applied to." A check of 50 labels from the first version
(first 3 slides only, no such line) found 38 right, 9 adjacent and 3 wrong, mostly HCI
work set in health labelled as a health science and CHI fabrication and materials talks
labelled Nano-technology; that version's output is kept in
`data/attributes/discipline_v1.csv`, and the 50 verdicts in `data/attributes/discipline_check_50.csv`.
The same 50 lectures were checked again after each change: `discipline_check_50_v2.csv` (41 right,
8 adjacent, 1 wrong) and `discipline_check_50_v3.csv` (the final labels: 42 right, 7 adjacent, 1 wrong).
The venue code is sent as the venue's full name and a short description of the venue,
from the M3AV paper and README (`[discipline.venues]`, sources in its comments), with no
venue tied to a field; the previous output, with bare codes, is kept in
`data/attributes/discipline_v2.csv`.

- An answer is **valid** if its `code` is in the list (or, failing that, its `field`
  equals a field name, ignoring case).
- An invalid answer is **retried once**, with the same prompt plus what was wrong (a
  new request, so the cache does not replay the bad answer). A second invalid answer
  gives `unassigned`.
- Every answer is cached in `data/llm_cache.sqlite3` under the hash of the full request
  (method `discipline`): a rerun replays, any change to the prompt or model asks again.
- Output: `data/attributes/discipline.csv` (corpus, lecture_id, field_code, field,
  attempts, raw_answer), `discipline_raw.jsonl.gz` (every answer of every attempt),
  `discipline.json` (model, temperature, the system prompt, the prompt and
  retry templates, counts).

The endpoint's address is read from `LLM_API_BASE`, the API key from `LLM_API_KEY` or the
untracked `config/llm_credentials.toml` only. Without one the script prints `discipline blocked: no API key` and writes nothing;
the table is then written with an empty discipline column and
`document_attributes.json` says `"discipline_status": "blocked: no API key"`.

## The layout model

**Docling's heron layout detector** (`docling-project/docling-layout-heron` on Hugging
Face, commit `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`; every file pinned by sha256 in
`config/attributes.toml`; `model.safetensors` sha256 `00333a43...4c7929e0c`).

- What: an RT-DETRv2 object detector (ResNet backbone) trained on about 150,000 pages
  (canonical DocLayNet, DocLayNet-v2, WordScape) with 17 classes, among them `picture`
  and `table` (the others: caption, footnote, formula, list item, page header and
  footer, section header, text, title, code, form, checkboxes, key-value region,
  document index).
- Licence: Apache-2.0.
- Reported accuracy: 0.776 mAP@[.50:.95] on the canonical DocLayNet test set
  (Livathinos et al. 2025, "Advanced Layout Analysis Models for Docling",
  arXiv:2509.11720, Table 3).
- Why this one: it has the two classes needed, a permissive licence and a published
  evaluation, and it runs with `transformers` alone (the versions the `models` extra
  already pins). DocLayout-YOLO (DocStructBench weights) was the other candidate: its
  code is AGPL-3.0 and needs its own fork of ultralytics. Neither model was trained on
  slides; the validation sheet below is how we find out how well it transfers.
- Threshold: **0.6**, fixed (the value of the model card's own example). Boxes below it
  are not stored.
- Inference: the model's own image processor (resize to 640 x 640), batch 32, fp32 with
  TF32 off, on one A100. torch 2.13.0+cu126, torchvision 0.28.0+cu126, transformers
  5.17.0, Pillow 12.3.0, numpy 2.2.6, in one Apptainer image (`cluster/layout.def`,
  base `python:3.12.15-slim-bookworm` pinned by digest; the full `pip freeze` is copied
  into `layout_boxes.json`). The image keeps torch's tens of thousands of files off the
  project space, which is close to its inode quota.
- Output: `data/attributes/layout_boxes.jsonl.gz`, one line per slide (corpus, lecture,
  slide, width, height, and every box of every class at or above the threshold: class,
  [x0, y0, x1, y1] in pixels, score), and `layout_boxes.json` (model, revision, hashes,
  threshold, packages, GPU, counts per corpus).

## Validation of the boxes

The detector is checked by eye: `scripts/detect_layout.py sheet` writes

- `data/attributes/validation_100.pdf` (not in git; the sheet shows slide images of the corpora): 100 slides, one per page, each with its picture
  boxes (blue) and table boxes (red) drawn and labelled with class and score, and a
  numbered id (S001-S100) with corpus, lecture and slide;
- `data/attributes/validation_100.csv`: the template to fill in, columns `slide_id`,
  `user_images_ok` (y/n), `user_tables_ok` (y/n), `comment`;
- `data/attributes/validation_100_key.csv`: what each id is (corpus, lecture, slide,
  stratum, number of picture and table boxes);
- `data/attributes/validation_100_checked.csv`: the filled sheet (picture boxes wrong on
  72 of the 100 slides, table boxes right on all 100);
- `data/attributes/validation_100_v2_checked.csv`: the same 100 slides checked again
  after the picture filter (`validation_100_v2.pdf`): picture boxes right on 85.

The sample is seeded (20261011) and stratified: the 100 slides are shared among the
corpora in proportion to their slide counts, with at least 10 per corpus (the small
corpora get 10, the rest is shared in proportion); within each corpus, 40 % of its quota
(rounded up) is drawn from the slides with at least one detected table, or all of them
if there are fewer, and the rest from the other slides. Tables are rare, and a uniform
sample would show almost none. "ok" means: every picture (table) on the slide has a box,
and every box is a picture (table).

## Files and commands

    sbatch cluster/00_layout_image.sbatch    # once: the image envs/lsab/images/layout.sif
    sbatch cluster/05_detect_layout.sbatch   # GPU: weights, boxes, validation sheet
    sbatch cluster/06_attributes.sbatch      # CPU: discipline (LLM), then the table
    python scripts/detect_layout.py sheet_v2  # after 06: the v2 sheet

| file | what |
|---|---|
| `scripts/build_wordlist.py` | builds `config/wordlist_en_scowl70.txt.gz` from SCOWL (run once; the list is in git) |
| `scripts/detect_layout.py` | `fetch` the weights, `detect` the boxes, draw the validation `sheet` and, after the picture filter, `sheet_v2` |
| `scripts/assign_discipline.py` | one FOS field per lecture with the LLM |
| `scripts/attributes.py` | the definitions above and the table, with its checks |
| `config/attributes.toml` | the model, revision, hashes, threshold, sample, word list, prompt sizes |
| `tests/test_attributes.py` | box union and percentages, the picture filter, noise and density, the discipline check and retry, the sample, the word-list build, the table schema |

The table's checks: exactly one row per kept lecture, every slide of every lecture has a
layout line, no missing value (except the discipline while blocked), every percentage in
0..100.

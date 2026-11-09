# Lecture slide alignment benchmark

A benchmark of methods that align lecture transcripts to slides, on four public corpora
(MaViLS, M3AV, MCD, Slide-Highlight). M3AV is split into two independent datasets with
different lectures, m3av_a and m3av_b, both prepared like any corpus with M3AV's own
hand-corrected slide OCR. The pipeline is listed below, step by step.

## Licence and data

The code is under the MIT licence. The data this project created (the M3AV split, the
selected lectures, the lecture attributes and categories, the importance labels, the
tuning and final-run tables, every method's predicted slide per unit, the similarity
tables, the analysis tables and the LLM answer cache, all under `data/`) is under CC BY
4.0. The corpora themselves are not in the repository and keep their own licences: M3AV
CC BY-NC-SA 4.0; MaViLS as its repository states (Apache-2.0 for the code, most slide
decks from MIT OpenCourseWare under CC BY-NC-SA 4.0, the rest without stated terms); MCD
and Slide-Highlight no licence, so they may not be redistributed. The prepared M3AV
files will be provided as a separate archive under CC BY-NC-SA 4.0, linked from this
README. `LICENSE` has the details.

What a clone holds: the code, the configs and jobs, and the files under `data/` listed
above. They carry the corpora's lecture ids and numbers, not their text or slides, so
every table and figure of the paper can be rebuilt from a fresh clone (step 17).

What the scripts download: the four corpora, at pinned commits or revisions and checked
by hash (`scripts/download.py`, step 1), into `data/raw/`; from these, the OCR and the
prepared lectures (`data/ocr/`, `data/prepared/`, steps 2b and 2) are made again, and the
tracked split and selection files pick the same 200 lectures. The
scorers' model weights (`config/similarity_weights.toml`) and the container images are
fetched or built by the jobs of steps 2b, 5 and 10, outside the repository.

## The whole pipeline, in order

On the cluster, from the repository root (environment: see `ACTIVATE.md`); each line is
one SLURM job or one script, to be started when the one before has finished (logs in
`cluster/logs/`), with the file that describes it after the `#`. The Slurm account, the
proxy and the LLM endpoint come from the untracked `cluster/env.local` (copied from
`cluster/env.local.example`), sourced first:

    source cluster/env.local                   # SBATCH_ACCOUNT, CLUSTER_PROXY, LLM_API_BASE, LLM_API_KEY
    mkdir -p cluster/logs

    sbatch cluster/00_tesseract_image.sbatch   # step 2b: the Tesseract image, once; README, Step 2b
    sbatch cluster/00_layout_image.sbatch      # step 5:  the layout-detection image, once; docs/ATTRIBUTES.md
    sbatch cluster/01_download.sbatch          # step 1:  data/raw/; README, Step 1
    sbatch cluster/02_split_m3av.sbatch        # step 2b: data/splits/m3av_split.csv; README, Step 2b
    sbatch cluster/03_ocr.sbatch               # step 2b: data/ocr/ (array, one task per OCR corpus); README, Step 2b
    sbatch cluster/04_prepare.sbatch           # step 2:  data/prepared/; README, Step 2
    sbatch cluster/05_detect_layout.sbatch     # step 5:  layout boxes + validation sheet (GPU); docs/ATTRIBUTES.md
    sbatch cluster/06_attributes.sbatch        # step 5:  discipline (LLM) + attributes table; docs/ATTRIBUTES.md
    python scripts/categorise.py               # step 6:  data/categories/; docs/CATEGORIES.md
    python scripts/select_lectures.py          # step 7:  data/selection/ (60 tuning, 140 test); docs/SELECTION.md
                                               # step 8:  the tuning design, nothing to run; docs/TUNING_DESIGN.md
    sbatch cluster/10a_similarity_image.sbatch                         # step 10: the scorers' image and weights, once; docs/SIMILARITY.md
    sbatch --export=ALL,SET=tuning cluster/10b_similarity_text.sbatch  # step 10: bm25, embed, hybrid; docs/SIMILARITY.md
    sbatch --export=ALL,SET=tuning cluster/10c_similarity_models.sbatch  # step 10: splade, image, cross_encoder (GPU); docs/SIMILARITY.md
    sbatch --export=ALL,SET=tuning cluster/10d_similarity_blends.sbatch  # step 10: the blends, sanity.csv, times.csv; docs/SIMILARITY.md
    sbatch cluster/11a_tune_trial.sbatch                               # step 11: the trial, 2 lectures; docs/TUNING.md
    python scripts/tune.py plan --name matrix --tasks 20               # step 11: the plan of (a); docs/TUNING.md
    sbatch --array=0-19%20 --export=ALL,PLAN=matrix cluster/11b_tune_matrix.sbatch   # step 11 (a): non-LLM methods; docs/TUNING.md
    sbatch --export=ALL,STAGE=llm cluster/11c_tune_llm.sbatch                        # step 11 (b): LLM methods; docs/TUNING.md
    sbatch cluster/11d_tune_select.sbatch                              # step 11: merge, select; docs/TUNING.md
    sbatch --export=ALL,STAGE=consensus cluster/11c_tune_llm.sbatch    # step 11 (c): combined methods; docs/TUNING.md
    sbatch cluster/11d_tune_select.sbatch                              # step 11: merge, config/tuned_settings.toml; docs/TUNING.md
    sbatch --export=ALL,SET=test cluster/10b_similarity_text.sbatch    # step 12: the test tables, as step 10; docs/SIMILARITY.md
    sbatch --export=ALL,SET=test cluster/10c_similarity_models.sbatch  # step 12; docs/SIMILARITY.md
    sbatch --export=ALL,SET=test cluster/10d_similarity_blends.sbatch  # step 12; docs/SIMILARITY.md
    python scripts/tune.py plan --set test --settings config/tuned_settings.toml --name matrix --tasks 4   # step 13; docs/FINAL_RUN.md
    sbatch --array=0-3 --export=ALL,PLAN=matrix,SET=test,SETTINGS=config/tuned_settings.toml cluster/11b_tune_matrix.sbatch      # step 13; docs/FINAL_RUN.md
    sbatch --export=ALL,STAGE=llm,SET=test,SETTINGS=config/tuned_settings.toml cluster/11c_tune_llm.sbatch                        # step 13; docs/FINAL_RUN.md
    sbatch --export=ALL,STAGE=consensus,SET=test,SETTINGS=config/tuned_settings.toml cluster/11c_tune_llm.sbatch                  # step 13; docs/FINAL_RUN.md
    sbatch cluster/13_final_report.sbatch      # step 13: data/final/ (merge, tables, checks); docs/FINAL_RUN.md
    python scripts/analysis.py                 # step 14: data/analysis/ (Q1, Q2); docs/ANALYSIS.md
                                               # step 15: the importance labels, in git; docs/IMPORTANCE.md, docs/IMPORTANCE_CASES.md
    sbatch cluster/16_q4.sbatch                # step 16: data/final/predictions/, Q4; docs/ANALYSIS.md, "Step 16"
    python scripts/paper_assets.py             # step 17: results/tables/, results/figures/; docs/ANALYSIS.md, "Step 17"

The three jobs of step 10 (and of step 12) run one after the other: submit 10c with
`--dependency=afterok:<10b>` and 10d with `--dependency=afterok:<10c>`
(`docs/SIMILARITY.md`, "Building the tables"). The steps without a job (6, 7, 14, 17,
18) are light and run on the login node.

Tests (synthetic data only, no network, no Tesseract):

    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests

## Step 1: download the corpora

`scripts/download.py` downloads each corpus from its official source, at a pinned
version, into `data/raw/<corpus>/` (not in git), verifies it, and writes
`data/raw/<corpus>/SOURCE.json` (URLs, date, version, licence, files with size and
sha256, what was kept and what was skipped). Videos and audio are skipped. Rerunning
skips corpora that are already complete.

One corpus by hand (compute nodes need the proxy):

    source cluster/env.local && export https_proxy=$CLUSTER_PROXY http_proxy=$CLUSTER_PROXY
    python scripts/download.py --corpus mavils      # mavils | m3av | mcd | slide_highlight | all

## Step 2b: the M3AV split and the Tesseract OCR

**The Tesseract image.** `cluster/tesseract.def` defines an Apptainer image: Ubuntu 24.04
(`noble-20260810`, pinned by digest) with Ubuntu's Tesseract packages pinned to exact
versions: `tesseract-ocr` 5.3.4-1build5 (Tesseract 5.3.4, Leptonica 1.82.0) and
`tesseract-ocr-eng` 1:4.1.0-2, the English model of tessdata_fast 4.1.0
(`eng.traineddata`, sha256 `7d4322bd...170b2`). `cluster/00_tesseract_image.sbatch`
builds it once into `$PROJECT_ROOT/envs/lsab-tesseract/tesseract.sif` (one file: the
project space is close to its inode quota).

**The split.** `scripts/split_m3av.py` takes the M3AV lectures that step 2 keeps (the same
loader and rules) and splits them into two independent datasets, 50/50 at random with a fixed seed (20261011), within
strata of topic x length so both halves have the same mix:

- topic: the M3AV source of the talk, the prefix of its id (CHI, IPP, MLS, NIH, Ubi);
  M3AV ships no other topic or field metadata;
- length: the tercile of the deck's slide count (before the cap).

It writes `data/splits/m3av_split.csv` (`lecture_id, version A|B, stratum`) and
`data/splits/m3av_split.json` (seed, cut points, balance of the halves). These two
files are a definition: they are not in `.gitignore`.

**The OCR.** `scripts/ocr.py` OCRs, and only these:

| OCR corpus | images |
|---|---|
| `mavils` | the MaViLS deck without a PDF text layer (`image_processing.pdf`), its pages rendered at 150 dpi exactly as step 2 renders them |
| `mcd` | every MCD slide image |
| `slide_highlight` | every Slide-Highlight slide image |

M3AV is not OCRed: m3av_a and m3av_b both use the corpus's own slide text.

The settings are fixed: `--oem 1` (LSTM), `--psm 3` (automatic page segmentation,
Tesseract's default), `-l eng`, TSV output, one thread per process. The text is rebuilt
line by line from the TSV; the mean confidence is the mean of the word confidences
(0-100). Each image also gets a blank flag (at most 0.1 % of its pixels more than 32
grey levels away from the most common one). Output in `data/ocr/` (not in git):
`<name>.jsonl.gz` (one line per image: lecture, slide, text, mean_conf, n_words,
blank_image, width, height) and `<name>.json` (engine, version, packages, language data
and its sha256, settings, image sha256, inputs, counts).

By hand (a compute node; `module load apptainer/1.4.3` first):

    python scripts/split_m3av.py
    python scripts/ocr.py --corpus mcd --workers 32     # mavils | mcd | slide_highlight | all

## Step 2: prepare the lectures (40-slide cap)

`scripts/prepare.py` reads the archives in `data/raw/` without unpacking them, the split
and the OCR, and writes, per prepared corpus (mavils, m3av_a, m3av_b, mcd,
slide_highlight), into `data/prepared/` (not in git, except `summary.csv` and the
`<corpus>_dropped.csv` files, which hold ids, reasons and counts only and are the source of
the paper's Table B1):

- `<corpus>.jsonl.gz`: one JSON record per kept lecture: the slides (index, text, image
  member, the corpus ids they come from, the OCR confidence and blank flag where the
  text is our OCR), the transcript units (text, times where the corpus has them,
  position in the corpus's unit list) with the true slide of each, the text source, the
  unit type, and the counts before and after the cap;
- `<corpus>_images.zip`: slide images, `<lecture>/<slide>.png|jpg`;
- `<corpus>_dropped.csv`: the lectures not kept, with the reason (the M3AV talks the
  rules drop are in no split version and are listed in both M3AV dropped files);
- `summary.csv`: per-corpus counts (below).

What `summary.csv` and the dropped files count. `summary.csv` has one row per prepared
corpus (mavils, m3av_a, m3av_b, mcd, slide_highlight; a run of one corpus replaces only
its row). Its columns: `corpus`; `text_source` (`source=decks` per text source of the kept
lectures); `unit_type`; `ground_truth`; `images`; `lectures_available` (every lecture the
loader yields, kept or dropped), `lectures_kept`, `lectures_dropped` (available = kept +
dropped) and `drop_reasons` (`reason=lectures`, `;`-separated, the same counts as the
dropped file); then sums over the kept lectures only: `slides_before_cap`,
`slides_after_cap`, `units_before_cap` (units with a true slide and text, whole lecture),
`units_after_cap` (the kept units), `lectures_cut` (deck over 40 slides or transcript
cut), `transcripts_cut` (transcript cut at the first unit of a slide after 40; the paper's
`\PrepCut`), `units_without_slide` (units the ground truth gives no slide, whole lecture),
`units_empty_text`, `slides_without_text`, `images_written`, `slides_ocr`,
`slides_ocr_empty_text` and `ocr_mean_conf` (mean word confidence over the OCR'd slides,
empty without OCR). `<corpus>_dropped.csv` has one row per dropped lecture: `corpus`,
`lecture_id`, `reason` (`no_ground_truth`: a MaViLS deck without its label file;
`no_slides`; `fewer_than_5_slides`; `no_labelled_units`; `slide_without_image`), `n_slides`
(the deck's slides, empty if unknown) and `detail`. The M3AV talks in neither split
version (the ones the rules drop before the split) are loaded and dropped in both m3av_a and
m3av_b, so each half's `lectures_available`, `lectures_dropped` and dropped file include
them and a sum over the two rows counts them twice. `scripts/paper_assets.py` (Table B1)
counts them once: the talks listed in both M3AV dropped files form their own row, and are
subtracted from each half's row; it checks that they are in neither version of the split
and that every other dropped M3AV lecture is in its own half (reason
`slide_without_image`).

The rules:

- **Cap:** slides 1-40 are kept. The transcript is kept from its start up to, not
  including, the first unit whose true slide is > 40; everything after it is cut, even
  if the lecture later returns to slides <= 40.
- **Drop:** a lecture whose deck has fewer than 5 slides is dropped.
- **Drop:** a lecture with a kept slide (1-40) that has no image is dropped (reason
  `slide_without_image`), so every kept lecture can be scored on its slide images. This
  drops 9 M3AV lectures (4 in m3av_a, 5 in m3av_b). The rule is applied after the split,
  so `data/splits/m3av_split.csv` does not change; these lectures are listed in their
  version's dropped file.
- Units the ground truth gives no slide (MaViLS label -1, speech outside every
  annotated slide segment) and units with empty text are not kept; the record counts
  them. The run stops if a kept unit has no true slide in 1..n, a lecture has more than
  40 slides or no slides, the unit order changes, the images do not match the slides,
  a slide whose text is our OCR has no OCR or has empty OCR text on an image that is not
  blank, a kept M3AV lecture is in no split version, or a lecture is in both m3av_a
  and m3av_b.

Per corpus:

| corpus | slide | slide text (text source) | unit | ground truth |
|---|---|---|---|---|
| MaViLS | PDF page; images rendered at 150 dpi | PDF text layer, PyMuPDF (`pdf_text_layer`); for the scanned `image_processing.pdf`, our Tesseract OCR of the renders (`tesseract_ocr`) | sentence (xlsx row, start time) | human label per sentence in the main xlsx (20 of 24 files; the 4 control re-labelings are not used) |
| M3AV A | video segments with identical text and matching image (32x24 thumbnails) are one slide, numbered by first appearance | the corpus's hand-corrected OCR (`corpus_corrected_ocr`) | speech segment (<= 10 s, written form, times) | segment holding most of the unit's words |
| M3AV B | as M3AV A (different lectures) | as M3AV A (`corpus_corrected_ocr`) | as M3AV A | as M3AV A |
| MCD | `sp_X_k.png`, renumbered 1..n in k order | our Tesseract OCR (`tesseract_ocr`) | transcript line (no times) | file pairing `sp_X_k.png` <-> `sp_X_k.txt` |
| Slide-Highlight | `PART_k_id.png` | our Tesseract OCR (`tesseract_ocr`) | subtitle cue (times relative to the slide segment) | file pairing `PART_k_id.png` <-> `PART_k_id.srt` |

One corpus by hand:

    python scripts/prepare.py --corpus mavils       # mavils | m3av_a | m3av_b | mcd | slide_highlight | all

## Step 3: the alignment methods

`methods/` holds the 31 alignment methods (wbs dropped), each checked and tested one by
one: what each does, its settings and the design decisions are in `docs/METHODS.md`. Every method maps a similarity matrix
(units x slides) to one 0-based slide per unit (`run_method(name, sim)`); the six LLM
methods also take the texts and an `LLM` (model and transport in `config/llm.toml`; the
endpoint's address from `LLM_API_BASE`, the API key from `LLM_API_KEY` or the untracked
`config/llm_credentials.toml`, both set in `cluster/env.local`; every
answer is cached in `data/llm_cache.sqlite3`, keyed by the full request). Tests
(`tests/test_methods.py`, `tests/test_llm_methods.py`) use made-up matrices and a fake
LLM; nothing calls the API.

## Step 4: the similarity scorers

`similarity/` holds the 8 scorers that build the matrix the methods read: bm25, embed,
splade, cross_encoder, image (ColQwen2 on the slide images), and the blends hybrid,
hybrid_splade and hybrid_image. What each does, its settings and the design decisions are
in `docs/SIMILARITY.md`. `build_matrix(name, lecture, ctx)` returns one lecture's
(units x slides) matrix in the form `run_method` takes. The settings, models and
endpoint are in `config/similarity.toml`. The embedding endpoint uses the same API key as
the LLM methods. Model-built matrices and embedding vectors are cached in
`data/similarity_cache.sqlite3` (not in git), keyed by every setting that changes the
numbers. splade, cross_encoder and image need the `models` extra, which is installed in
the Apptainer image of `cluster/similarity.def` (built by `cluster/10a_similarity_image.sbatch`),
not in the virtual environment. The tests (`tests/test_similarity.py`) use made-up texts and fake models.
Nothing loads a model or calls the endpoint.

## Step 5: the lecture attributes

`scripts/attributes.py` writes `data/attributes/document_attributes.csv`: one
row per kept lecture, with corpus, lecture_id, text_source, n_slides, n_units,
n_transcript_words, image_pct, table_pct, noise, text_density and discipline. Every
definition, the model choices and the validation procedure are in `docs/ATTRIBUTES.md`;
in short:

- **image_pct / table_pct**: the share of the lecture's slide area (full image pixels,
  summed over its slides) covered by the union of the picture / table boxes that a
  layout-detection model finds (`scripts/detect_layout.py`: Docling's heron detector,
  Apache-2.0, pinned commit and sha256 in `config/attributes.toml`, threshold 0.6, run
  on a GPU from the Apptainer image `cluster/layout.def`). M3AV's slide images are video
  frames. Every box is kept in `data/attributes/layout_boxes.jsonl.gz` for auditing.
- **noise**: the share of slide-text tokens (letter runs, lowercased, length >= 2) not in
  the fixed English word list `config/wordlist_en_scowl70.txt.gz` (SCOWL 2020.12.07,
  built by `scripts/build_wordlist.py`).
- **text_density**: mean slide-text words per slide.
- **discipline**: one OECD FOS 2007 field (`config/oecd_fos2007.csv`), chosen by the
  LLM from the title, the M3AV venue code and the first 3 slides
  (`scripts/assign_discipline.py`; answers cached like the LLM methods'). Without an API
  key the step reports "discipline blocked: no API key" and the column stays empty.

**The check of the boxes**: `data/attributes/validation_100.pdf` (not in git: `scripts/detect_layout.py sheet` draws it) shows 100 slides
(seeded, stratified by corpus, at least 10 per corpus, slides with tables over-sampled)
with their boxes; `data/attributes/validation_100_checked.csv` holds the verdicts
(slide_id, user_images_ok, user_tables_ok, comment).

By hand (compute nodes; the image and the weights must exist):

    python scripts/detect_layout.py fetch                    # proxy needed
    apptainer exec --nv --containall --pwd $(pwd -P) -B $(pwd -P) -B <models_lsab> \
        <envs/lsab/images/layout.sif> python scripts/detect_layout.py detect
    python scripts/detect_layout.py sheet
    python scripts/assign_discipline.py                      # proxy and API key needed
    python scripts/attributes.py

## Step 6: the categories

`scripts/categorise.py` (settings in `config/categories.toml`, rules in `docs/CATEGORIES.md`)
gives every lecture one level per axis in `data/categories/`:
`python scripts/categorise.py`.

## Step 7: the lecture selection

`scripts/select_lectures.py` (settings in `config/selection.toml`, rules in `docs/SELECTION.md`)
chooses a tuning set (60 lectures) and a disjoint test set (140) with at least 5 and 10
lectures per level of every axis, into `data/selection/`:
`python scripts/select_lectures.py`.

Step 8, the tuning design (a written proposal; nothing run): `docs/TUNING_DESIGN.md`.

Step 10, the similarity tables: `scripts/build_similarity.py` builds all 8 measures' tables
for one lecture set into `data/similarity/<set>/tables.sqlite3`, timed per
lecture, with a sanity check; three jobs, `cluster/10b`, `10c` (GPU) and `10d`, after the
image and weights of `cluster/10a` (see `docs/SIMILARITY.md`, "Building the tables").

Step 11, the tuning run: `scripts/tune.py` runs every combination of each method's grid
(`config/tuning_grid.toml`) on the 60 tuning lectures into `data/tuning/runs/` (in git),
and `scripts/select_settings.py` chooses one per method into
`config/tuned_settings.toml` and `data/tuning/selection_report.csv` (see `docs/TUNING.md`).

Step 13, the final run: every method with its tuned setting on the 140 test lectures,
through `scripts/tune.py --set test --settings config/tuned_settings.toml`, into
`data/final/`; `scripts/final_report.py` writes one table per method
(accuracy and cost per lecture), the summary and the checks (see `docs/FINAL_RUN.md`).

Step 14, the analysis: `scripts/analysis.py` reads the final run and the categories and writes
Q1 (the methods by mean test accuracy, with cost) and Q2 (every method within each category
level) as CSV and markdown into `data/analysis/`, with its checks (see `docs/ANALYSIS.md`):
`python scripts/analysis.py`. The paper's Q3, one highest and one lowest method at every
level of every attribute, is read from these Q2 rows by `scripts/paper_assets.py`
(`results/tables/tab_q3.tex`, step 17). `data/analysis/q3_best_worst.csv` and `.md` are a
different view of the same rows: per method, the level where its rank is best and the one
where it is worst; the paper does not use them.

Step 15, the importance labels: every unit of the 140 test lectures labelled 0/1/2 (filler /
supporting / core) against its true slide by Claude agents following `docs/IMPORTANCE.md`;
committed as data; checked by a second labelling of 10 lectures and a blind 100-unit check by
another Claude model.
`scripts/importance_pack.py` writes the packs the agents read (only the slides and the units
with their true slide) and their batches into `data/importance/packs/` (not in git);
`scripts/importance_merge.py` checks the agents' CSVs (`data/importance/raw/`) and writes
`data/importance/labels.csv.gz` and `labels_meta.json`; `--agreement A B` compares two
labellings into `data/importance/agreement.csv`. `scripts/importance_sample.py` draws the
blind 100-unit check sample (a lecture at random, then a unit in it; seed 20261012) into
`data/importance/local/check_sample.md` (no labels) and `data/importance/check_sample_key.csv`
(the labels). The sheet quotes the corpora's text, so it is not distributed: `local/` is kept
out of git, and the same seed draws it again from the prepared lectures.
The recurring cases met during labelling, and the agreement numbers: `docs/IMPORTANCE_CASES.md`.

Step 16, Q4 (important units placed on the wrong slide): `cluster/16_q4.sbatch` replays
the final run offline, LLM answers from the cache only, into `data/final/predictions/` (one
predicted slide per unit, `scripts/final_predictions.py`), then `scripts/analysis_q4.py`
writes `data/analysis/q4_wrong_important.csv`/`.md`, `q4_heatmap.pdf`/`.png` and
`q4_checks.txt`: per method and column (all documents; length short, medium, long; images
low, medium, high; tables none, some) the important units (label 2) placed on the wrong
slide, as a percentage of all units of the column's documents. Claude assigned the
importance labels; our scripts did the evaluation. The labelling is not part of any
method's cost (see `docs/ANALYSIS.md`, "Step 16").

Step 17, the paper's tables and figures: `python scripts/paper_assets.py` writes `results/tables/*.tex` and `results/figures/*.pdf` from the results, under the names of `scripts/paper_names.py` (in git; see `docs/ANALYSIS.md`, "Step 17"), and the numbers the text quotes into `results/tables/numbers.tex`. Every input is a file in git (the example figure takes its lecture's true slides from `data/final/predictions/`), so it runs on a fresh clone.

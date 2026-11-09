# The 8 similarity scorers (step 4)

A scorer turns one lecture into the **similarity matrix** the methods read (docs/METHODS.md).

- **Input:** the unit texts and the slide texts of one lecture, in `index` order. `image`
  and `hybrid_image` also read the slide images from `data/prepared/<corpus>_images.zip`.
- **Output:** a float matrix with one row per unit and one column per slide (column `k` is
  the slide with `index` `k + 1`). A larger value means more similar, and every value is
  finite. This is what `methods.run_method(name, sim)` takes. The methods z-score each
  row before they use it (`DPConfig.standardize`), so a scorer's row scale and offset do
  not matter to them. What matters is the order of the slides within a row and how far
  apart their scores are.
- **Code:** `similarity/` (`core.py` holds the numpy scoring rules, `models.py` the
  models and the endpoint, `cache.py` the cache, and `registry.py` the 8 scorers and
  `build_matrix`).

      from similarity import Context, build_matrix, lecture_from_record, images_zip
      ctx = Context.from_config()                    # offline=True: read the cache only
      lec = lecture_from_record(record, images_zip=images_zip("mcd"))
      sim = build_matrix("hybrid", lec, ctx)

- **Settings:** `config/similarity.toml`, one section per scorer. A missing or unknown
  setting is an error. The embedding endpoint is the LLM methods' (address
  `LLM_API_BASE`, key `LLM_API_KEY` or the untracked `config/llm_credentials.toml`), which
  is never in git.
- **Cache:** `data/similarity_cache.sqlite3` (one file, not in git). It holds the matrices
  of the model scorers (splade, cross_encoder, image) and one vector per text for embed.
  A matrix is stored under the sha256 of the scorer, **every setting that changes its
  numbers** (its own settings and those of every scorer it is built from), the unit and
  slide texts, the sha256 of every slide image it reads, and for bm25 the sha256 of the
  stop-word list and the stemmer's version (when they are on). An embedding vector is
  stored under (model, the exact text sent). Settings that only change speed or transport
  (the endpoint's batch size, timeout, retries, rate and URL) are left out of the key.
  `CODE_VERSION` in `registry.py` is in the key too. bm25 and the blends are cheap and
  are recomputed each time.
- **Tests:** `tests/test_similarity.py`. Fake models stand in for the real ones, and no
  test loads a model or calls the endpoint. For every scorer the tests check the shape
  and orientation, that the right slide wins on an easy lecture, empty texts, an empty
  lecture, determinism, that a matrix read from the cache equals the one computed, and
  that every setting is in the cache key. Some tests are about a single scorer: the
  blend math, the scoring rules worked out by hand (BM25, SPLADE, MaxSim, the top-k
  merge) and missing images. Each design note and each decision below has a test.
- **Packages:** bm25, embed and the blends need numpy only, and bm25's stemming needs
  `snowballstemmer` (pinned, pure Python, imported only when stemming is on). The
  embedding client uses the standard library. splade, cross_encoder and image need the `models` extra in
  `pyproject.toml` (torch, transformers, colpali-engine, pillow), at pinned versions. The
  extra is not in the virtual environment: step 10 runs it from the
  Apptainer image `cluster/similarity.def`. These packages are imported only when a
  matrix is missing from the cache.
- **Where they run:** the device of each local model is a setting (`device = "cpu"` by
  default, or `"cuda"` on a GPU node). The device is in the cache key, because a GPU and
  a CPU give slightly different last digits.

## Building the tables (step 10)

`scripts/build_similarity.py` builds the tables of all 8 measures for one lecture set
(`tuning`, 60 lectures, or `test`, 140; data/selection/selected_lectures.csv) with
`build_matrix` and the settings of config/similarity.toml as they are. Three jobs, one
after another, because the shared cache is a SQLite file in WAL mode that two nodes must
not write at once:

    sbatch cluster/10a_similarity_image.sbatch          # once: the image and the weights
    sbatch --export=ALL,SET=tuning cluster/10b_similarity_text.sbatch          # CPU: bm25, embed, hybrid
    sbatch --export=ALL,SET=tuning --dependency=afterok:<10b> cluster/10c_similarity_models.sbatch   # GPU: splade, image, cross_encoder
    sbatch --export=ALL,SET=tuning --dependency=afterok:<10c> cluster/10d_similarity_blends.sbatch   # CPU: hybrid_splade, hybrid_image, checks

- **Models.** splade, cross_encoder and image run from the Apptainer image
  `cluster/similarity.def` (the `models` extra; one file in $PROJECT_ROOT/models_lsab/images/,
  because torch in the virtual environment would cost tens of thousands of inodes). Their
  weights are pinned to one Hugging Face commit each, every file checked by sha256
  (`config/similarity_weights.toml`), and downloaded once into $PROJECT_ROOT/models_lsab/
  (`python scripts/build_similarity.py fetch`). The settings keep the Hugging Face names,
  so the keys do not depend on where the files lie. For ColQwen2 the pin is the last
  commit in colpali-engine's own format; the later commits only wrap the same weights
  for Sentence Transformers.
- **Device.** The jobs run the three models on a GPU: `--model-device cuda` sets their
  `device` to cuda (config/similarity.toml says cpu). The device is in their keys and in
  those of the blends built from them, so every job of a set passes the same value; the
  set stores it.
- **Where the tables live.** `data/similarity/<set>/tables.sqlite3`, one
  file per set: one row per measure and lecture with the matrix (float64, zlib-compressed,
  read back bit-identical), its key, and its times; the runs (host, hardware, device,
  model load time, package versions); the set's lectures in order; and the settings that
  decide the numbers, fixed at the first build. `Store.get(measure, corpus, lecture_id)`
  reads a table back. The model matrices and the embedding vectors are also in the shared
  cache, as `build_matrix` always does.
- **Rerunning.** A table is stored under its `matrix_key`, so a rerun reuses every table
  whose key still matches and builds only the rest. A build with other number-deciding
  settings than the set's stops; delete the set's folder to start again.
- **What is timed.** A measure's build time is its cost, reported apart from the methods
  (docs/TUNING_DESIGN.md 1.1 and 6). Per lecture: `wall_s`, the whole `build_matrix`
  call, and `cost_s`. For a local model, its load is timed once per run (`load_s`), apart
  from the compute. For embed the cost rule of the LLM calls applies: only successful
  requests count, each by the duration of its successful attempt; the rate-limit and
  backoff sleeps and the failed attempts are recorded apart (`wait_s`, `failed_s`,
  `failed_attempts`, `errors`) and are not cost, so embed's `cost_s` is its successful
  requests plus the local work. `EmbeddingClient` times each attempt as `OpenAICompatClient`
  does. Each run records the hardware (CPU model and cores, GPU name). A blend's own time
  reads its inputs from the cache; `python scripts/build_similarity.py times --set <set>`
  writes `data/similarity/<set>/times.csv` with each measure's load, compute, endpoint
  counts, hardware, and its cost with the measures it is built from (`cost_with_deps_s`).
- **Checks.** `python scripts/build_similarity.py sanity --set <set>` checks that every
  table is there, has the prepared lecture's shape (units x slides), has no NaN or inf,
  and has the key recomputed from the prepared data (the lecture, and the order of its
  units and slides). As a sign that nothing is broken or misaligned, not a result, it
  gives per measure the median rank of the true slide in each unit's row (ties share
  their mean rank) and the share of units whose true slide is in the top 3, next to the
  share a random row would give. It writes `data/similarity/<set>/sanity.csv`.
- **Tests.** `tests/test_build_similarity.py`, on a made-up set with the fake models of
  `tests/test_similarity.py` and the real embedding client on a fake transport and clock:
  the stored tables equal `build_matrix`, the storage round trip, a rerun that reuses
  everything and calls nothing, a changed lecture rebuilt alone, the settings lock, the
  load time apart, the cost rule for embed, and the checks catching broken tables.

## Summary

| scorer | model | runs on | design note |
|---|---|---|---|
| bm25 | none | CPU (numpy) | decided: 2-letter tokens kept, English stop words removed, stemming a setting (off) |
| embed | `qwen3-embedding-4b` | OpenAI-compatible embedding endpoint | the vector cache key holds the model; decided: query instruction on the units |
| splade | `naver/splade-v3-distilbert` | local torch (cpu/cuda) | - |
| cross_encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` | local torch (cpu/cuda) | the cache key holds the settings of the matrix it reranks |
| image | `vidore/colqwen2-v1.0-merged` | local torch + colpali-engine (cpu/cuda) | the cache key holds the model; decided: lectures with a slide without an image are dropped in step 2 |
| hybrid | from embed and bm25 | CPU (numpy) | - |
| hybrid_splade | from embed and splade | CPU (numpy) | - |
| hybrid_image | from embed, bm25 and image | CPU (numpy) | decided: two weights, inner and outer |

Matrix transforms applied after a table is built (csls, dual_softmax, qb_norm,
column_center, probcons) are not scorers and are not part of the benchmark. The
embedding model is a setting of embed, and the tables are built by step 10's
`scripts/build_similarity.py` (below).

---

## bm25

**How it works.** BM25 (Robertson and Zaragoza 2009). Each unit is a query and each
slide is a document. The document collection is **the lecture's own slides**, so the
inverse document frequency (IDF) is computed per lecture. It measures how specific a
word is to a few slides of this deck, and a word that is on every slide counts for
little. For each word of the unit that is on slide j, the score adds
`idf · f · (k1 + 1) / (f + k1 · (1 − b + b · len_j / avg_len))`, where f is the word's
count on the slide. The IDF is `log(1 + (N − n + 0.5) / (n + 0.5))`, which is never
negative. A word that appears twice in the unit counts twice. Each row is then divided by
its maximum, so values are in [0, 1].
**Tokens.** Lower-cased runs of letters and digits. Tokens shorter than `min_token_len`
are dropped, then the English stop words (`remove_stopwords`), then, if `stem` is on,
each token is stemmed. The stop-word list is `similarity/stopwords_en.txt`: the NLTK
English list (179 words, derived from the Snowball list; the source is cited in the
file). The stemmer is the Snowball English stemmer (Porter2) of `snowballstemmer`.
**Settings.**
- `k1` = 1.5: how quickly repeats of a word on a slide stop adding to the score. A lower
  value makes a slide that mentions a word once almost as good as one that repeats it.
- `b` = 0.75: how much long slides are penalised. 0 means no penalty. 1 means the full
  penalty, which favours short, focused slides.
- `min_token_len` = 2: shorter tokens are dropped. 2 keeps "AI", "ML", "3D" and
  two-digit numbers.
- `remove_stopwords` = true: function words ("the", "of", "and") do not count, so they
  cannot make a slide look similar.
- `stem` = false: when true, word forms match ("networks" and "network", "learning" and
  "learned"), at the cost of a few false matches. It is a tuning setting (step 8).

**Empty texts.** A slide with no text, or one with only short words, scores 0 for every
unit. A unit that shares no word with any slide gets a row of zeros, which the methods'
z-scoring turns into a row that prefers no slide.
**Decided: 2-letter tokens kept, English stop words removed, stemming an on/off
setting (off by default); all three are in the cache key.** Reason: acronyms such as
"AI" and "ML" are often the key terms of a slide, a fixed published stop list removes the
function words that a 3-letter cut only partly removes, and stemming is left to tuning.
Tests: `test_decided_bm25_keeps_two_letter_tokens`, `test_decided_bm25_removes_stop_words`
and `test_decided_bm25_stemming_is_a_setting_off_by_default`.

## embed

**How it works.** Each unit and each slide is embedded by a dense text-embedding model.
The score is the cosine of the two vectors, in [−1, 1].
**Model.** `qwen3-embedding-4b` (Qwen3-Embedding-4B, Zhang et al. 2025; 2560 dimensions),
served by the OpenAI-compatible endpoint (`/v1/embeddings`). The endpoint does not
version its weights, so the cache is what makes reruns reproducible.
**Settings.**
- `model`: the embedding model, a tuning setting (step 8).
- `max_chars` = 6000: each text is cut to this many characters before embedding (the
  MaViLS truncation). Only very long slides are affected.
- `query_instruction` = "Given a sentence from a lecture transcript, retrieve the lecture
  slide it refers to": the task sentence. Each unit is sent as
  `Instruct: <task>\nQuery:<unit text>`, the format of the Qwen3-Embedding model card
  (no space after "Query:"). Slides are sent plain. "" turns it off. The sentence is part
  of the text sent, so it is in the vector key.
- `batch` = 512, `batch_chars` = 1,000,000, `timeout_s`, `max_retries`,
  `requests_per_minute` = 20: transport only. The server embeds every text on its own
  (measured: the same vector at 16 to 512 texts per request), so the batch size does not
  change the vectors. The endpoint limits
  requests per account, so parallel jobs must share the rate.

**Empty texts.** An empty text is sent as a single space. So a slide
without text gets the vector of " " and some cosine with every unit. Only 11 MaViLS
slides have no text, and the OCR corpora have none.
**Design note (cache key).** Each vector is keyed by the model and the exact text sent,
which includes the truncation and any instruction, so the same cache used with another
embedding model never returns the first model's vectors. Test:
`test_fixed_embed_vectors_are_keyed_by_model`.
**Decided: the query instruction on the unit side only.** Reason: Qwen3-Embedding is
trained to receive retrieval queries with a task instruction and documents without one,
and its authors report that queries without one do slightly worse. Test:
`test_decided_embed_instruction_on_units_only`.

## splade

**How it works.** SPLADE (Formal et al. 2021) is a learned sparse retrieval model. Its
masked-language-model head gives every token of a text a weight for every vocabulary
word. The weights are `log(1 + relu(logit))`, max-pooled over the text's tokens and with
padding masked. So each text becomes a sparse vector of weighted words, which includes
related words the text never used. The score of a unit and a slide is the dot product of
their vectors, and each row is divided by its maximum, as in bm25.
**Model.** `naver/splade-v3-distilbert` (SPLADE-v3, Lassance et al. 2024; 67M
parameters; licence cc-by-nc-sa-4.0, research use). It runs locally on the configured
device.
**Settings.**
- `max_chars` = 6000: the same cut as embed.
- `max_length` = 256: tokens per text. Longer texts lose their end.
- `min_weight` = 1e-4: words at or below this weight are dropped (numerical dust).
- `batch` = 32, `device` = cpu: these change only the last digits. They are in the key.

**Empty texts.** These are sent as " " (the embed rule). If a unit's dot product is 0 for
every slide, the unit gets a row of zeros.
The scoring and the pooling match SPLADE.

## cross_encoder

**How it works.** This is two-stage reranking (Nogueira and Cho 2019). For each unit,
the `top_k` best slides of a first-stage matrix (by default `hybrid`) are rescored by a
cross-encoder. The cross-encoder reads the unit and the slide together and gives one
relevance logit. The k new scores are mapped linearly onto the band the k old values
held: the cross-encoder's best slide gets the row's old maximum, and its worst gets the
old k-th value. The other slides are untouched. So only the order inside the top k
changes. The row's range stays the same, so the methods' z-scores and positional prior
keep their meaning. The size of the gaps between the cross-encoder's own scores is not
kept. Ties in the first stage go to the lower slide.
**Model.** `cross-encoder/ms-marco-MiniLM-L6-v2` (MiniLM trained on MS MARCO passage
ranking; 22M parameters; Apache-2.0). It runs locally on the configured device.
**Settings.**
- `first_stage` = hybrid: the matrix that gets reranked. It is one of bm25, embed,
  hybrid, splade or hybrid_splade.
- `top_k` = 5: how many slides per unit are reranked. A larger value lets the
  cross-encoder rescue a correct slide that the first stage ranked lower. A value of 1
  changes nothing. The cost is n_units × top_k pairs.
- `max_length` = 256: tokens per pair, cut from the longer side first.
- `batch` = 64, `device` = cpu: these change only the last digits. They are in the key.

**Empty texts.** A pair with an empty slide is scored like any other pair. If the
cross-encoder gives all k slides the same score, the row is left as it was.
**Design note (cache key).** The key holds the settings of the whole first stage, so a
new embedding model, BM25 setting or blend weight in the hybrid matrix it reranks never
returns a matrix reranked from another first stage. Test:
`test_fixed_cross_encoder_key_includes_its_first_stage`.

## image

**How it works.** ColPali-style late interaction (Faysse et al. 2025). The slide is read
as a picture: there is no OCR and no slide text. ColQwen2 turns each slide image into
several hundred patch vectors and each unit text into token vectors (128 dimensions,
L2-normalised). The score is MaxSim: for each token of the unit, take its best-matching
patch, then sum over the tokens. The sum grows with the unit's length, so rows of
longer units have larger numbers. The methods' row z-scoring removes that. Padding
positions are dropped and never take part.
**Model.** `vidore/colqwen2-v1.0-merged` (ColQwen2, built on Qwen2-VL-2B, with the
adapter already merged in; Apache-2.0). It runs locally through colpali-engine on the
configured device, in float32.
**Settings.**
- `image_batch` = 1, `query_batch` = 8, `query_block` = 64: batching. Queries are sorted
  by length inside fixed blocks of 64 to reduce padding. These change only the last
  digits and are in the key.
- `dtype` = float32, `device` = cpu: in the key.

**Design note (cache key).** The key holds the model, every setting and the sha256 of
every image. Test: `test_fixed_image_key_includes_the_model`.
**Decided: a lecture with a kept slide without an image is dropped in step 2.**
Reason: every lecture is then scored by every scorer, and the 9 such lectures (one M3AV
slide each: CHI-0A6E7, CHI-77CFF, IPP-77743 and Ubi-BF690 in m3av_a; MLS-792FC,
NIH-157F0, NIH-22AB3, NIH-85CEA and NIH-AF1B2 in m3av_b) are too few to need a fill
rule. `scripts/prepare.py` drops them with the reason `slide_without_image` (in
`<corpus>_dropped.csv`); the M3AV split file is unchanged. The scorer still raises
`MissingSlideImage` on a slide without an image, as a safety net; there is no fill
rule. Tests: `test_lecture_with_a_kept_slide_without_image_is_dropped`
and `test_m3av_lecture_without_an_image_is_dropped_and_the_split_holds`
(tests/test_prepare.py) and
`test_missing_image_is_always_an_error`.

## hybrid

**How it works.** A blend of embed and bm25. Each matrix is z-scored row by row, and the
result is `(1 − weight) · z(embed) + weight · z(bm25)`. Z-scoring first matters: cosines
sit in a narrow band and BM25 has many zeros, so a raw mix would let one of them dominate
for reasons that have nothing to do with the right slide. A row with no spread (a unit
no scorer can tell apart) becomes zeros.
**Settings.** `weight` = 0.4, the share of bm25. 0 gives embed alone and 1 gives bm25
alone (both z-scored). A higher weight trusts exact word overlap more, which helps where
the slide text and the speech share terms, and hurts where the OCR is noisy or the speaker
paraphrases.

## hybrid_splade

**How it works.** The same blend as hybrid, with splade in bm25's place:
`(1 − weight) · z(embed) + weight · z(splade)`. At the same weight it differs from hybrid
in the sparse part only.
**Settings.** `weight` = 0.4, the share of splade.

## hybrid_image

**How it works.** A text blend of embed and bm25, blended with the image matrix:
text = `(1 − inner_weight) · z(embed) + inner_weight · z(bm25)`, then
`(1 − outer_weight) · z(text) + outer_weight · z(image)`. At the defaults the image has a
0.4 share and the text blend 0.6, which is itself 0.6 embed and 0.4 bm25. It is meant
for slides whose text says little (pictures, diagrams, formulas) but whose image does.
**Settings.** `inner_weight` = 0.4, the share of bm25 in the text blend; `outer_weight` =
0.4, the share of image. Both are this scorer's own settings and both are in the key;
`[hybrid].weight` does not affect it.
**Decided: two separate weights, inner and outer, both 0.4.** Reason: the text blend and
the image blend mix different things, so tuning should be able to move them apart, and
neither should follow the hybrid scorer's weight. Test:
`test_decided_hybrid_image_has_two_weights`.

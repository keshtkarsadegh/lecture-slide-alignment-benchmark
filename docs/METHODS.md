# The 31 alignment methods (step 3)

Each method gets one lecture and returns **one slide per transcript unit**.

- **Input:** a similarity matrix `sim` with one row per unit and one column per slide (a
  larger value means more similar). The six LLM methods also get the slide texts and the
  unit texts in order, plus an `LLM` object (the model, the response cache and, when
  calls are allowed, the client).
- **Output:** a list with one entry per unit: the 0-based column of its slide. Column
  `k` is the prepared slide with `index` `k + 1`. `-1` means "no slide". Only non-default
  settings can produce it (dp's `no_slide_score`, `drop_fill` = none in drop_dtw and
  ot_unbalanced), and since every unit in the data has a true slide, a `-1` is always
  scored as wrong.
- **Code:** `methods/` (`common.py`, `monotone.py`, `soft_order.py`, `boundaries.py`,
  `transport.py`, `consensus.py`, `llm.py`, `llm_client.py`, `registry.py`).
  `run_method(name, sim, ...)` runs a method by name, and `METHODS[name].settings` lists
  its settings with their defaults. `lecture_texts(record)` and `gold_labels(record)`
  read a record of `data/prepared/<corpus>.jsonl.gz`.
- **Tests:** `tests/test_methods.py`, `tests/test_llm_methods.py` and (step 9: members'
  own inputs, the cost of a call) `tests/test_tuning_prep.py`. No test calls the API.

Every method was checked in step 3 against its source and, where the lattice is small
enough, against exhaustive search. The **design note** of a method says where the code
does something a reader might not expect, and why; a note marked **decided** records a
choice between reasonable options, with the reason. Each note has a test. `wbs` was
dropped, so there are 31 methods.

## Summary

| group | method | design note |
|---|---|---|
| Independent labelling | argmax | decided: no threshold setting |
| Monotone path | dtw | - |
| | multiscale_dtw | the fine pass is DTW in a band, not a greedy ratchet |
| | needleman_wunsch | a "stay" cannot follow a skipped slide; decided: unassigned units filled from neighbours |
| | gotoh | the same as needleman_wunsch |
| | drop_dtw | the drop score is a percentile per row |
| | anchored_dp | - |
| | semimarkov | decided: fixed log-normal duration shape, so `dur_weight` has an effect |
| | adaptive_prior | the curve is passed to the shared code |
| Path with going back | dp | - |
| | rank_dp | tied scores share their average rank |
| | cohesion_dp | the cohesion signal leaves the positional prior out; Hearst's depth score |
| | jump_dtw | - |
| | forward_backward | - |
| | softdtw | - |
| Boundary-first | pelt | - |
| | bocpd | decided: labels traced back from the best final hypothesis (monotone) |
| | wbs2 | the statistic leaves the prior out; breadth-first recursion |
| | texttiling | the cohesion leaves the prior out; Hearst's depth score |
| | texttiling_dp | the same as texttiling |
| Transport | opw | - |
| | ot_unbalanced | - |
| Consensus | ensemble | decided: no weights or extra voters; llm_single_call is a fixed sixth voter |
| | score_fusion | the soft Needleman-Wunsch member uses needleman_wunsch's lattice |
| | mbr | int32 back-pointers, so any k |
| LLM reads the text | llm_single_call | the cache key holds the whole request |
| | llm_monotone | the cache key |
| | hierarchical | the answer budget fits long leaves; the cache key |
| LLM + similarity | llm_rerank_kbest | decided: candidates are the answers of 5 different methods |
| | fusion | through hierarchical |
| | llm_monotone_constrained | decided: the counts are rescaled to add up to the transcript |

**Dropped by decision.** `wbs`: each random interval proposed only one split, so it
rarely found the boundaries a deck needs; `wbs2` is the recursive version and stays.

**Grouping.** `llm_monotone_constrained` is listed under "LLM + similarity", but it reads
the matrix only for its fallback. `no_slide_score` is a setting of `dp` alone, the only
method that reads it.

## The shared parts

**Input check.** Every method rejects a matrix that is not 2-D, that has NaN or inf in
it, or that has units but no slides. It returns `[]` for a lecture with no units.

**Shared DP settings (`DPConfig`).** All the path methods use these, and `run_method(...,
cfg=DPConfig(...))` sets them:

- `forward_penalty` = 0.05: the cost of each slide moved forward.
- `backward_penalty` = 0.5: the cost of each slide moved back. It is only read by
  methods that can go back (dp, rank_dp, cohesion_dp, jump_dtw, forward_backward).
- `stay_bonus` = 0.2: the reward for staying on the same slide from one unit to the
  next.
- `standardize` = true: each row is z-scored first. Raw similarities sit in a narrow
  band, and the costs above only have a meaning on a known scale.
- `pos_weight` = 8: the weight of the positional prior. A unit at relative position p is
  expected near slide p·(m−1), and the score is lowered by
  `pos_weight · |slide − expected| / m`. With 0 there is no prior.

**Prepared emissions.** These are the z-scored rows minus the positional prior. All the
path methods add these numbers up.

**Cohesion (TextTiling's signal).** The cohesion at a gap is the cosine between the mean
rows of the `window` units before the gap and the `window` units after it. The depth of
a gap is how far its cohesion lies below the nearest peak on its left plus the nearest
peak on its right (Hearst 1997). texttiling, texttiling_dp and cohesion_dp use it.

---

## Independent labelling

### argmax
**How it works.** Each unit gets its most similar slide. No order and no prior are
used. It is the baseline that the ordered methods are read against. Ties go to the
lower slide.
**Settings.** None.
**Decided: no `no_slide_threshold` setting.** Reason: it could only
turn labels into -1, which is always scored wrong, so it could only lower accuracy.

## Monotone path

### dtw
**How it works.** Dynamic time warping (Sakoe and Chiba 1978) finds the best path
through the matrix that never goes back. From one unit to the next the path stays on its
slide (+stay_bonus), moves one slide forward (−forward_penalty) or moves two slides
forward (−2·forward_penalty). The path may start on any slide, paying forward_penalty
for each slide skipped before it, and may end on any slide.
**Settings.** Only the shared ones.
**Design note.** Neither end of the path is anchored.

### multiscale_dtw
**How it works.** This is FastDTW (Salvador and Chan 2007). The rows are averaged in
pairs, again and again, until at most `min_size` remain. DTW solves that small matrix.
Its path is then projected back up one level at a time, and at each level DTW is solved
again, allowed only within `radius` slides of the projected path.
**Settings.** `min_size` = 32: below this many units, plain dtw is used. `radius` = 4:
the width of the band (larger is closer to dtw).
**Design note.** The fine pass is the DTW recursion inside the band. Giving each row the
best slide in the band that is not below the previous row's slide would be a greedy
ratchet, not DTW: one noisy row that jumps ahead drags every later row with it. On a
clean 400 × 40 matrix such a ratchet scores 6 %; the DTW refinement scores 99 %, the same
as dtw.

### needleman_wunsch
**How it works.** Global sequence alignment (Needleman and Wunsch 1970) of the units
against the slides, in a many-to-one form. Each unit is either the first unit on a
slide, or another unit on the slide it is already on (+stay_bonus), or a gap (no slide,
−gap_segment). A slide that no unit is on costs gap_slide. The path covers every slide,
so slides skipped at the end are paid for too. Moving to the next slide costs nothing.
**Settings.** `gap_segment` = 0.4: the cost of leaving a unit unassigned (it is then filled, see below).
`gap_slide` = 0.1: the cost of each skipped slide.
**Design note.** Each cell remembers whether its slide already has a unit, and only then
may a "stay" follow. With one score per cell a "stay" could follow a skipped slide
("skip slide j, then stay on slide j"), and the first unit on a slide would collect the
stay bonus whenever stay_bonus > gap_slide, which is true at the defaults. On 30 small
random cases, enumerated exhaustively, the method always finds the optimum (one score per
cell misses it 4 times).
**Decided: units the alignment leaves unassigned are filled from their neighbours**
(the previous unit's slide; a leading run takes the first assigned slide - drop_dtw's
"carry" rule). Reason: every unit has a true slide, so -1 could only be wrong. The raw
alignment, with its -1s, is `needleman_wunsch_alignment`.

### gotoh
**How it works.** needleman_wunsch with affine slide gaps (Gotoh 1982). A run of L
skipped slides costs gap_open + (L−1)·gap_extend, so skipping a whole block, such as an
appendix, is cheap. Unit gaps still cost the same each time.
**Settings.** `gap_segment` = 0.4, `gap_open` = 0.4, `gap_extend` = 0.05. When
gap_open = gap_extend = gap_slide, the result is needleman_wunsch (a test checks this).
**Design note.** The same two-state cells as needleman_wunsch (one score per cell misses
the optimum in 1 of the 30 cases). **Decided:** unassigned units are filled from their neighbours, as in
needleman_wunsch, for the same reason (raw alignment: `gotoh_alignment`).

### drop_dtw
**How it works.** Drop-DTW (Dvornik et al. 2021) is DTW in which a unit may be dropped.
A unit is dropped when no slide it can reach beats its drop score, which is the
`drop_percentile` percentile of its own row. A dropped unit leaves the path where it is.
The path may advance any number of slides at once, paying forward_penalty +
(k−1)·slide_drop.
**Settings.** `drop_percentile` = 60: higher means more drops. `slide_drop` (default
forward_penalty): the cost of each slide passed over. `drop_fill` = carry: a dropped
unit takes the slide the path is on. The other values are argmax (the unit's own best
slide) and none (-1).
**Design note.** The paper takes one percentile over the whole matrix; this code takes
it per row.

### anchored_dp
**How it works.** Bertalign's two passes (Liu and Zhu 2023). First, anchors are fixed:
the most confident units, one per slide, whose best slide beats their second best by at
least the `margin_q` quantile of all such margins, and which DTW can connect. Then DTW
runs with every anchor fixed to its slide, and each unit between two anchors limited to
the slides between theirs. If no path fits, plain dtw is used.
**Settings.** `margin_q` = 0.75: higher means fewer, surer anchors; 1.0 means no anchors,
which is plain dtw.

### semimarkov
**How it works.** Semi-Markov segmentation (Sarawagi and Cohen 2004). The transcript is
cut into runs of consecutive units, and each run gets one slide, in strictly increasing
order. A run of L units scores the sum of its emissions + stay_bonus·(L−1) +
dur_weight·log P(L). Each skipped slide costs `skip_penalty`. P is a fixed duration
shape: a log-normal over the run length whose median is the lecture's units per slide
(n/m) and whose width (sigma of log L) is 0.5, discretised and normalised over
1..max_run. Nothing is fitted on data.
**Settings.** `dur_weight` = 1: the weight of the duration term (0 drops it).
`max_run` (default max(4, 6 × units per slide)): the longest a run may be.
`skip_penalty` (default forward_penalty).
**Decided: a fixed parametric duration shape (log-normal, median n/m, sigma 0.5).**
Reason: a duration model fitted on other lectures would need data no step provides, and
without any model `dur_weight` would have no effect. A test shows that `dur_weight`
changes the alignment, and the method is checked against exhaustive
search with the duration term included. Sigma 0.5 puts the most likely run length at
0.78 × n/m; with sigma 1 it would sit near one unit, which is the geometric shape
this method exists to avoid.

### adaptive_prior
**How it works.** dtw with the straight-line prior replaced by a pacing curve read from
the matrix. Each row gives the slide it expects, through a soft argmax at temperature
`tau`. That sequence is smoothed with a moving average of width `smooth`, made
non-decreasing (PAVA, Ayer et al. 1955), stretched to cover all the slides, and mixed
with the straight line (`alpha` is the curve's share).
**Settings.** `tau` = 0.05, `alpha` = 0.5, `smooth` = 9. With `pos_weight` = 0 the
curve has no effect.
**Design note.** The curve is passed to the shared emission function directly. The
decoder under the curve is always dtw, because the method is monotone.

## Path with going back

### dp
**How it works.** Viterbi (1967) over the slides. Moving forward costs
forward_penalty per slide, moving back costs backward_penalty per slide, and staying
earns stay_bonus. This is the way MaViLS (Anderer et al. 2024) penalises slide changes.
**Settings.** `no_slide_score` (none): when set, an extra "no slide" state with this
score is added (label -1).

### rank_dp
**How it works.** dp on ranks instead of similarities. Each value is replaced by its
rank within its row, scaled to 0..1 (the rank idea of Choi 2000).
**Settings.** Only the shared ones.
**Design note.** Ties share their average rank. Ranks in column order would turn a row
of equal scores, such as a unit that shares no word with any slide (common with lexical
scores), into a ramp that favours the last slide.

### cohesion_dp
**How it works.** dp in which the cost of changing slide follows the transcript's own
topic shifts. Every move into unit i (forward or back, but not a stay) is charged its dp
cost × (1 + strength·(0.5 − depth[i])). Here depth[i], in 0..1, is the TextTiling depth
of the gap before unit i. At strength 1, a move costs half as much at the deepest dip
and 1.5 times as much in the middle of a topic.
**Settings.** `window` = 3: the number of units on each side of a gap. `strength` = 1:
the valid range is 0..2. Above 2, a move into a deep dip would be rewarded, so such
values are rejected.
**Design note.** The cohesion is computed on the z-scored rows, without the positional
prior: the prior depends only on the row number, so it would swamp the content. The
depth climbs to the nearest peaks, as in Hearst, not to the global maximum.

### jump_dtw
**How it works.** dtw plus backward jumps of up to `max_jump` slides (Fremerey et al.
2010). A jump back of d slides costs d·jump_cost.
**Settings.** `max_jump` = 3 (0 gives dtw). `jump_cost` (default backward_penalty).

### forward_backward
**How it works.** The hidden Markov model behind dp: transitions are exp(dp's
transition scores), normalised per row, and emissions are the softmax of each prepared
row. Each unit takes its most probable slide given all the units (Baum et al. 1970;
Rabiner 1989). This is also MEA decoding of the same model.
**Settings.** Only the shared ones. A test checks it against exhaustive search.

### softdtw
**How it works.** Soft-DTW (Cuturi and Blondel 2017) used as a decoder: dtw's lattice
with each max replaced by a smoothed max at temperature `gamma`.
**Settings.** `gamma` = 0.2: smaller is closer to dtw. `mode` = expected: each unit takes
the most likely slide over all paths, so labels can step back. `mode` = path follows one
path back through the smoothed scores, so its labels are monotone. A test checks its
marginals against exhaustive search.

## Boundary-first

### pelt
**How it works.** The best segmentation with a fixed cost per slide change (Killick et
al. 2012). A change may jump any number of slides forward at the same price, and the
labels never go back. This is the exact DP; PELT's pruning is not needed at this size.
**Settings.** `change_penalty` = 1: higher means fewer slide changes. A test checks it
against exhaustive search.

### bocpd
**How it works.** Bayesian online change-point detection (Adams and MacKay 2007),
adapted to slides. It keeps several hypotheses about how long ago the slide changed,
each tied to one slide. At every unit a hypothesis either continues, adding
log(1−hazard) plus the unit's score on its slide, or (from the best hypothesis only) the
slide changes, adding log(hazard), to the unit's best slide at or after the current one
(the next slide when that best is the current one). The `max_hypotheses` most probable
are kept. At the end, the most probable hypothesis is traced back to the first unit,
and the labels are the slides along that trace.
**Settings.** `hazard` = 0.05: the probability of a slide change at each unit.
`max_hypotheses` = 64.
**Decided: trace back the best final hypothesis, so the labels never go back.**
Reason: every hypothesis moves forward, but an online read-out (the most probable
hypothesis at each unit) steps back whenever a younger hypothesis overtakes an older
one. The method is therefore not online.

### wbs
**Decided: dropped.** Reason: each random interval proposed only one split, so 60
intervals rarely gave the m−1 distinct boundaries a 40-slide deck needs (about 20 % on
clean 400 × 40 matrices); `wbs2`, which recurses inside each segment, stays.

### wbs2
**How it works.** WBS2 (Fryzlewicz 2020). Each segment draws up to `n_intervals` random
sub-intervals, takes the best split among them, and splits; both sides are then treated
the same way, until segments are shorter than 4 units or 4m splits have been found.
Because the number of slides is known, the m−1 strongest splits (at least n/(3m) apart)
are kept, without the paper's SDLL model selection. The runs get slides in strictly
increasing order.
**Settings.** `n_intervals` = 40, `seed` = 0.
**Design note.** (1) The split score uses the z-scored rows alone: with the positional
prior in it, every long stretch would look best split near its middle. (2) The
recursion goes breadth-first: depth-first, it would spend its 4m budget on the left part
of the transcript. On clean 400 × 40 and 500 × 20 matrices the method scores 97 %
(6–7 % with the prior in the score and a depth-first recursion). (3) With exactly m−1
boundaries, the labelling is forced (run k gets slide k), so on noisy data every wrong boundary shifts the
labels.

### texttiling
**How it works.** TextTiling (Hearst 1997). The cuts go at the m−1 deepest cohesion
dips, deepest first, at least max(2, n/(2m)) units apart. Each run then takes its best
slide after the previous run's slide. A transcript shorter than 2·window + 1 units falls
back to dtw (not dp, which can go back).
**Settings.** `window` = 3.
**Design note.** The cohesion leaves the prior out, and the depth climbs to the nearest
peaks (see cohesion_dp). On a clean 400 × 40 matrix with noise 0.2 the method scores
100 % (48 % with the prior inside the cohesion and the global-maximum depth).

### texttiling_dp
**How it works.** The same depth score, but exactly m−1 cuts are chosen by DP to
maximise the summed depth, at least max(1, n/(3m)) apart. With m−1 cuts every slide
gets exactly one run.
**Settings.** `window` = 3.
**Design note.** As texttiling: 100 % in the same test (10 % with the prior and the
global-maximum depth). The note on forced labelling (see wbs2) applies here too.

## Transport

### opw
**How it works.** Order-preserving Wasserstein transport (Su and Hua 2017/2019). It
finds a soft assignment of the units' mass to the slides, in which every unit sends
equal mass and every slide receives an equal share. The assignment is rewarded near the
diagonal (`lambda1`) and smoothed towards a Gaussian band around the diagonal
(`lambda2`, width `sigma`). Each unit then takes the slide it sends the most mass to,
so labels can step back.
**Settings.** `lambda1` = 1, `lambda2` = 0.1, `sigma` = 1, `n_iter` = 2000 (Sinkhorn
iterations). The paper's lambda1 = 50 belongs to its own cost scale; here costs are
scaled to 0..1.

### ot_unbalanced
**How it works.** Entropic transport (Chizat et al. 2018) in which the equal shares are
only encouraged (`reg_m`), so a slide nobody talks about can receive little. Each unit
takes its row's maximum.
**Settings.** `eps` = 0.05: smoothing. `reg_m` = 0.5: a large value gives balanced
transport, a small one lets each unit choose freely. `drop_mass` = 0: when above 0, a
unit sending less than this share of its mass is dropped and then filled by
`drop_fill` (carry, argmax, or none = -1). `n_iter` = 2000.

## Consensus

**Members' own inputs (step 9, docs/TUNING_DESIGN.md 2.6).** ensemble, score_fusion and
llm_rerank_kbest take `member_inputs`: per member name (`MEMBERS` in
`methods/registry.py`), a `Member` with that member's own matrix, cfg and settings (the
settings of the registry method of that name), and for ensemble's LLM voter its own
`LLM`, so its own model. `run_method(..., member_inputs=...)` passes it on. A member
without an entry, or a `Member` field left empty, takes the consensus method's own
matrix, cfg or LLM and the member's defaults, so without `member_inputs` every method
runs exactly as described below. A member's matrix must have the consensus matrix's
shape, and an unknown member name or setting is an error.

### ensemble
**How it works.** A vote per unit (Dietterich 2000) of six voters with one equal vote
each: dp, dtw, needleman_wunsch, cohesion_dp and pelt (each at its own defaults) and
llm_single_call (at its defaults). Ensemble therefore makes llm_single_call's one
request, or reuses its cached answer when llm_single_call was run before on the same
lecture with the same settings. A width-3 median then removes one-unit spikes.
**Tie rule.** With six voters, ties (3–3, 2–2–2, 2–2–1–1) are more likely than with
five. When several slides share the most votes, the unit takes the first member's label
(dp's) if it is among them, otherwise the lowest tied slide. Tests check both branches.
**Settings.** `members` (the five matrix voters above; llm_single_call is always
added), `smooth` = true.
**Decided: no weight settings and no generic extra-voter input; llm_single_call is a
fixed sixth voter, and all six votes count equally.** Reason: without an extra voter,
weights would do nothing (all votes would have the same weight). An unknown member is an
error.

### score_fusion
**How it works.** The sum rule (Kittler et al. 1998). It adds three per-unit
probability tables and takes each unit's best slide:
- soft dtw (gamma 0.2);
- cohesion_dp's hidden Markov model (window 3, strength 1);
- soft Needleman-Wunsch (gaps 0.4 and 0.1, gamma 0.3).

**Settings.** `nw_gamma` = 0.3: the softness of the soft Needleman-Wunsch member (its
own setting since step 9, because the hard needleman_wunsch, whose other settings that
member takes, has none). The shared DP settings reach all three members. Without `member_inputs` the members' own settings stay at the values
above; with it, the members take softdtw's, cohesion_dp's and needleman_wunsch's
settings (softdtw's `mode` has no counterpart in the posterior and is dropped).
**Design note.** The soft Needleman-Wunsch member has the two-state lattice of
needleman_wunsch, and its backward pass scores a "stay" with the current slide's
similarity (with the next slide's, the probabilities are off by up to 0.53). The soft
dtw, HMM and Needleman-Wunsch members all match exhaustive search.

### mbr
**How it works.** Minimum Bayes risk (Kumar and Byrne 2004) over dtw's k best paths.
The k best are found exactly, by keeping the k best partial paths in every cell. Each
unit then takes the slide that most of the paths give it, so labels can step back.
**Settings.** `k` = 100. `temperature` (none): with none, all paths count equally;
otherwise each path counts exp(score/temperature).
**Design note.** The k-best back-pointers are 32-bit integers: 8-bit ones give wrong
paths for k > 255 (72 of 400 in the test case).

## LLM methods

**Configuration.** `config/llm.toml` (in git) holds the model
(`qwen3-30b-a3b-instruct-2507`, the one model of every LLM method), temperature 0,
the thinking switch, retries, the requests per minute (the endpoint limits each account, so
parallel jobs must share them) and the cache path. **The endpoint's address and the key
are never in git.** The client reads the address from `LLM_API_BASE` and the key from
`LLM_API_KEY`, or the untracked `config/llm_credentials.toml` (one line,
`api_key = "..."`), which `.gitignore` excludes; both variables are set in the untracked
`cluster/env.local` (ACTIVATE.md). `LLM.from_config()` builds the client;
`LLM.from_config(offline=True)` answers only from the cache. The client uses only the
standard library, and nothing calls the API in this step.

**Cache.** Every answer is stored in `data/llm_cache.sqlite3` (in git) under the hash
of the **whole request**: the model, both messages (so every text cap and size), the
temperature, `max_tokens`, the response format and the thinking switch. Any change
therefore means a new call, never a replay of an answer to another request (for
example a truncated one under a smaller `max_tokens`). A call that fails, from the network or an HTTP error, is not cached. An
unusable answer is cached and counted.

**Common behaviour.**
- Slides and units are numbered from 0 in the prompts.
- An unusable answer (prose, broken JSON, cut off) gives `parse_failed`, and the labels
  fall back to the matrix argmax. The argmax is used rather than another ordered method
  so that a failed LLM run never quietly copies another method.
- `diagnostics` reports calls, cache hits, tokens and `truncated`, which counts answers
  that stopped at `max_tokens`, and the cost fields below.

**Cost (step 9, docs/TUNING_DESIGN.md 6.3).** Only the successful attempt's time counts.
`OpenAICompatClient.chat` times each attempt with `time.perf_counter()` from just before the
request to just after the answer is read and parsed; the rate-limit wait and the backoff
sleep are outside that window (returned apart as `wait_s`, for information only). It
returns `duration_s` (the successful attempt), `attempts`, `failed_attempts` and the
error kinds. `LLM.ask` stores `duration_s` and `attempts` in the cache record next to
the tokens. `LLM.stats()`, and so every method's `diagnostics`, keeps three groups apart:
- paid in this run: `calls`, `call_seconds`, `prompt_tokens`, `completion_tokens`;
- the cost of the answers used, fresh or cached: `used_calls`, `used_seconds`,
  `used_prompt_tokens`, `used_completion_tokens` (a cache hit brings its stored duration
  and tokens; `used_untimed` counts answers stored without a duration);
- failures, never cost: `failed_attempts` (failed before a call succeeded),
  `failed_calls` (calls that raised `LLMCallFailed`) and `errors` (by kind).

### llm_single_call
**How it works.** One call, with the whole deck and the whole transcript. The model
assigns every unit a slide, as row-range blocks, in any order. Rows that no block
covers take the previous row's slide.
**Settings.** `slide_chars` = 3000 and `row_chars` = 800: each text is cut to this
many characters. `max_tokens`: the default is min(4000, 256 + 12 per unit).

### llm_monotone
**How it works.** One call. The model says at which unit each slide starts, and the code
makes the starts non-decreasing (running maximum, first start = 0). Missing starts leave
the last slides unreached.
**Settings.** `slide_chars`, `row_chars`; `max_tokens`: the default is
min(1500, 64 + 8 per slide).

### hierarchical
**How it works.** The transcript is cut into packets of units, about one slide's worth
each. A *split* call asks where the lecture moves from the left half of a page range to
the right half, and the two halves are handled the same way. A node with one page, or
with few pages and few packets, becomes a *leaf*: one call assigns each of its units a
page. A *boundary* call then re-asks the units on either side of each page change. Page
order, coverage and range are enforced in code. An answer that cannot be parsed falls
back to the middle packet (for a split) or an even spread (for a leaf), and these
repairs are counted. It does not read the matrix.
**Settings.**
- `ask_for_reason` = false: whether the model adds a reason to each answer.
- `packet_size`, `leaf_max_packets` and `packet_chars` are sized per lecture: a packet
  is about one slide's worth of units (3..20), and a leaf holds about 12 slides' worth
  (3..8 packets).
- `leaf_max_pages` = 3.
- `max_depth` = 32 and `max_calls` = 600: safety ceilings.
- `page_preview_chars` = 1200 (split calls), `page_full_chars` = 8000 (leaf and boundary
  calls) and `section_chars` = 900: text caps.
- `use_boundary_refinement` = true, with `boundary_context_before` and
  `boundary_context_after` = 2.
- `tokens_per_unit` = 40, `tokens_per_unit_with_reason` = 120 and `max_answer_tokens` =
  8192: the answer budget.

**Design note.** Leaf and boundary answers get 64 + 40 tokens per unit (120 with
reasons), up to 8192, and cut-off answers are counted. A compact entry
`{"section_idx": 123, "page_index": 12}` is about 18–22 tokens, because Qwen tokenises
numbers digit by digit, and a leaf can hold up to 8 × 20 = 160 units; a budget of
64 + 16 per unit capped at 1200 covers at most 71, so long leaves would be cut off, fail
to parse and fall back to an even spread. `max_calls` is counted per lecture.

### llm_rerank_kbest
**How it works.** The alignments of five different methods - dtw, dp, pelt,
needleman_wunsch and cohesion_dp, each at its own defaults, in that fixed order - are
shown to the model as slide row ranges (each method on its own matrix, cfg and settings
when `member_inputs` gives them; see Consensus). Identical alignments are shown once, under the
first method's position. The model picks one. Candidate 1 is dtw's, and an unusable
answer also gives dtw's. The prompt says each candidate comes from a different
alignment method, and that a slide may appear more than once (dp and cohesion_dp can go
back).
**Settings.** `slide_chars`, `row_chars`; `max_tokens` = 64.
**Decided: the candidates are the answers of five different methods, deduplicated, in
a fixed order.** Reason: dtw's exact top-5 paths are near-duplicates - they differ from
the best path in at most 2 of 80 units in the test case - so the model's choice could
hardly move away from dtw.

### fusion
**How it works.** dp with the hierarchical answer as its positional prior. The
hierarchical labels give the expected slide of each unit, the prior charges
`pos_weight · |slide − expected| / m`, and dp's transitions decide.
**Settings.** `pos_weight` = 10, plus all of hierarchical's settings.

### llm_monotone_constrained
**How it works.** One call under a JSON schema that the server enforces (guided
generation, Willard and Louf 2023). The model gives the number of units spent on each
slide. The schema enforces exactly m whole numbers, each from 0 to n. The counts are
then rescaled to add up to n, and the starts are their running sum, so every answer is
monotone and reaches the end of the transcript.
**Rounding rule (largest remainder).** Each count c becomes c·n/sum, rounded down. The
units still missing go one each to the slides with the largest fractional parts, ties
to the lower slide. If all counts are 0, the same rule splits n evenly.
**Settings.** `slide_chars`, `row_chars`; `max_tokens` = 900.
**Decided: the counts are rescaled to add up to the number of units.** Reason: the
schema cannot constrain their sum. Unscaled, counts that add up to too much leave the
last slides unreached, and counts that add up to too little give the last slide the
rest. `diagnostics` records how far off the model was: `raw_sum` (the
sum the model gave) and `sum_off` (raw_sum − n), per lecture.

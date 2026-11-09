# Step 8: the tuning design (a proposal)

This is a written proposal. Nothing in it has been run: no method, no tuning, no job.
No code, config or test was changed for it.

It says, for each of the 31 methods (docs/METHODS.md), which settings are tuned, with
which values and why, which settings stay fixed and why, how the best setting is chosen,
how cost is counted, and how large the tuning is. The decisions of 2026-10-12 are
applied throughout and listed in section 7. The last of them (one LLM model only,
qwen3-30b-a3b-instruct-2507; section 3.1) was taken during the tuning run and is applied
to every count below.

Facts used below (read from the code and the data, 2026-10-12):

- The tuning set is 60 lectures, the test set 140, disjoint (docs/SELECTION.md).
- In the 60 tuning lectures: 1,047 slides and 6,350 units. The longest slide text has
  1,642 characters (26 slides have more than 1,000, none more than 3,000). The longest
  unit has 541 characters (none more than 800). The median lecture has 40 units and
  13.5 slides; the largest has 615 units. Units per slide run from 1.5 to 43 (median 3.7).
- A whole-lecture LLM prompt (all slides and all units) is about 2,800 tokens for the
  median tuning lecture and about 16,000 for the largest (characters / 4).

## 1. Settings shared by many methods

### 1.1 The similarity measure

Every method gets a similarity matrix (docs/SIMILARITY.md). Which measure built it is a
setting of every method that reads the matrix. **All 8 measures are tried** (decided):

| measure | what it brings |
|---|---|
| bm25 | pure word overlap; the lexical end |
| embed | pure dense meaning; the paraphrase end |
| splade | learned sparse words: word evidence that includes related words |
| cross_encoder | hybrid with each unit's top 5 slides re-ordered by a cross-encoder |
| image | the slide image alone, no slide text |
| hybrid | the default text blend of embed and bm25 |
| hybrid_splade | the blend of embed and splade |
| hybrid_image | a text blend blended with the image |

The default is hybrid. "× 8" in the counts below is this setting.

**The measures' own settings stay at their defaults** (decided): bm25 `k1`, `b`,
`min_token_len`, `remove_stopwords`, `stem`; the embedding model and its instruction;
cross_encoder `first_stage` and `top_k`; the blend weights; all as in
config/similarity.toml. The paper states this as a fixed choice. Reason: each of them
would multiply every method's grid, and the 8 measures already span lexical, dense,
learned sparse, re-ranked, image and blended scores.

Methods that do not read the matrix as a measure:
- hierarchical does not read it at all;
- llm_single_call, llm_monotone and llm_monotone_constrained read it only for their
  fallback (the matrix argmax when an answer is unusable). Their measure stays hybrid;
  see section 2.7;
- ensemble, llm_rerank_kbest and score_fusion do not have a measure of their own: each
  member reads its own tuned measure (section 2.6).

### 1.2 The shared DP settings (`DPConfig`)

These are passed to a method as `cfg`. Each method reads only some of them (table
below). Only the fields a method reads are tuned for it.

| setting | default | what it does | values | reason |
|---|---|---|---|---|
| `forward_penalty` | 0.05 | cost per slide moved forward; also the price of skipping slides and of a late start | 0.02, 0.05, 0.1 | half and double the default; its total effect is small (paid once per slide advanced), so a narrow range |
| `backward_penalty` | 0.5 | cost per slide moved back (only methods that can go back) | 0.25, 0.5, 1.0 | lecturers sometimes return to a slide; half makes returns easier, double makes them rare |
| `stay_bonus` | 0.2 | reward for staying on the same slide; the main smoothness knob | 0.1, 0.2, 0.4 | half and double: fewer or more slide changes |
| `pos_weight` | 8 | weight of the positional prior (a unit is expected near the diagonal) | 0, 4, 8, 16 | 0 switches the prior off; 4 and 16 halve and double it. It is the strongest shared knob, so it gets one more value |
| `standardize` | true | z-score each row first | fixed: true | the measures have very different raw scales (bm25 in 0..1 with many zeros, cosines in a narrow band, image scores that grow with unit length); the costs above only mean the same thing on z-scores |

Which method reads which field (from the code):

| method | forward_penalty | backward_penalty | stay_bonus | pos_weight |
|---|---|---|---|---|
| argmax | - | - | - | - |
| dtw, multiscale_dtw, anchored_dp, drop_dtw, semimarkov, adaptive_prior, softdtw, mbr | yes | - | yes | yes |
| needleman_wunsch, gotoh | - | - | yes | yes |
| dp, rank_dp, cohesion_dp, forward_backward | yes | yes | yes | yes |
| jump_dtw | yes | yes (as `jump_cost`) | yes | yes |
| pelt, bocpd, wbs2, opw, ot_unbalanced | - | - | - | yes |
| texttiling, texttiling_dp | (only in the dtw fallback) | - | (only in the dtw fallback) | yes |
| ensemble, llm_rerank_kbest, score_fusion | each member's tuned values | each member's tuned values | each member's tuned values | each member's tuned values |
| fusion | yes | yes | yes | replaced by fusion's own `pos_weight` |
| llm_single_call, llm_monotone, llm_monotone_constrained, hierarchical | - | - | - | - |

texttiling and texttiling_dp fall back to dtw only when a transcript has fewer than
2 · window + 1 units. No tuning lecture is that short (the shortest has 21 units), so
their forward_penalty and stay_bonus stay at the defaults.

Shorthand below: **F/S/P** = forward_penalty × stay_bonus × pos_weight = 3 × 3 × 4 = 36
combinations; **F/B/S/P** adds backward_penalty = 108; **P** = pos_weight alone = 4.

## 2. Per method

"Combinations" = settings combinations including the measure. Each combination runs on
all 60 tuning lectures.

### 2.1 Independent labelling

**argmax.** Each unit takes its most similar slide. No settings of its own.
- measure: the 8.
- Combinations: 8.

### 2.2 Monotone path

**dtw.** The best path that never goes back.
- shared: F/S/P.
- Combinations: 36 × 8 = 288.

**multiscale_dtw.** FastDTW: solve a coarsened matrix, then refine in a band.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `radius` | 4 | width of the band around the coarse path; small = the coarse path decides more, large = closer to dtw | 2, 4, 8 | half and double; 8 is close to plain dtw on 40-slide decks |
| `min_size` | 32 | below this many units, plain dtw is used | fixed: 32 | it sets how many coarsening levels there are, which moves the result the same way as `radius`; one knob is enough |

- shared: F/S/P. Combinations: 36 × 3 × 8 = 864.
- Note: lectures with at most 32 units are solved by plain dtw at every value.

**needleman_wunsch.** Global alignment; a unit may be a gap (then filled from its neighbours).

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `gap_segment` | 0.4 | cost of leaving a unit unassigned; low = many units are skipped and take a neighbour's slide | 0.2, 0.4, 0.8 | half and double |
| `gap_slide` | 0.1 | cost of each slide no unit is on; low = skipping slides is easy | 0.05, 0.1, 0.3 | half, and three times: decks often have slides nobody talks about, so the upper end is wider |

- shared: stay_bonus, pos_weight (3 × 4). Combinations: 12 × 9 × 8 = 864.

**gotoh.** needleman_wunsch with affine slide gaps: a block of skipped slides is cheap.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `gap_segment` | 0.4 | as in needleman_wunsch | 0.2, 0.4, 0.8 | as in needleman_wunsch |
| `gap_open` | 0.4 | cost of starting a run of skipped slides | 0.2, 0.4, 0.8 | half and double |
| `gap_extend` | 0.05 | cost of each further skipped slide in a run | 0.02, 0.05, 0.1 | half and double; 0.1 with gap_open 0.1 would be needleman_wunsch, which is tuned on its own |

- shared: stay_bonus, pos_weight. Combinations: 12 × 27 × 8 = 2,592.

**drop_dtw.** dtw in which a unit may be dropped.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `drop_percentile` | 60 | a unit is dropped when no reachable slide beats this percentile of its own row; higher = more drops | 40, 60, 80 | few, default and many drops |
| `drop_fill` | carry | what a dropped unit gets: the path's slide (carry) or its own best slide (argmax) | carry, argmax | the two fills that give a slide; `none` gives -1, which is always wrong |
| `slide_drop` | = forward_penalty | cost of each extra slide passed in one jump | fixed: = forward_penalty | it follows forward_penalty, which is already tuned |

- shared: F/S/P. Combinations: 36 × 6 × 8 = 1,728.

**anchored_dp.** Fix the surest units as anchors, then dtw between them.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `margin_q` | 0.75 | how sure a unit must be to anchor (a quantile of best-minus-second margins); higher = fewer anchors | 0.5, 0.75, 0.9 | many, default and few anchors; 1.0 means no anchors, which is dtw, tuned on its own |

- shared: F/S/P. Combinations: 36 × 3 × 8 = 864.

**semimarkov.** Runs of units, one slide each, with a duration term.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `dur_weight` | 1 | weight of the run-length term (runs near the lecture's units per slide are preferred) | 0, 1, 2 | 0 switches the term off; 2 doubles it |
| `max_run` | max(4, 6 × units per slide) | longest allowed run | fixed | a safety cap far above typical runs; the duration term already makes long runs unlikely |
| `skip_penalty` | = forward_penalty | cost of each skipped slide | fixed: = forward_penalty | it follows forward_penalty, which is already tuned |

- shared: F/S/P. Combinations: 36 × 3 × 8 = 864.

**adaptive_prior.** dtw with the straight-line prior replaced by a pacing curve read from the matrix.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `alpha` | 0.5 | the curve's share against the straight line | 0.25, 0.5, 0.75 | mostly line, even, mostly curve |
| `tau` | 0.05 | temperature of the soft argmax that reads the curve; low = each unit's top slide only | 0.05, 0.5 | sharp (default) and smooth |
| `smooth` | 9 | moving-average width of the curve | fixed: 9 | the curve is also made non-decreasing and mixed with the line; the width adds little on top |

- shared: forward_penalty, stay_bonus, and pos_weight 4, 8, 16 (not 0: with no prior the curve has no effect and the method is dtw). 3 × 3 × 3 = 27.
- Combinations: 27 × 6 × 8 = 1,296.

### 2.3 Path with going back

**dp.** Viterbi over slides; may go back.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `no_slide_score` | none | adds a "no slide" state | fixed: none | it can only produce -1, which is always wrong |

- shared: F/B/S/P. Combinations: 108 × 8 = 864.

**rank_dp.** dp on within-row ranks. No settings of its own.
- shared: F/B/S/P. Combinations: 108 × 8 = 864.

**cohesion_dp.** dp whose change costs follow the transcript's topic shifts.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `window` | 3 | units on each side of a gap when cohesion is measured | 3, 6 | short blocks for long units (sentences), longer blocks for short units (subtitle cues, lines) |
| `strength` | 1 | how much the topic shifts change the cost (0 = dp; at most 2) | 0.5, 1, 2 | weak, default and the strongest allowed |

- shared: F/B/S/P. Combinations: 108 × 6 × 8 = 5,184.

**jump_dtw.** dtw plus backward jumps.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `max_jump` | 3 | longest jump back, in slides (0 = dtw) | 1, 3, 8 | a step back, default, and a return to a far slide (an outline or an earlier figure) |
| `jump_cost` | = backward_penalty | cost per slide jumped back | fixed: = backward_penalty | it follows backward_penalty, which is tuned here |

- shared: F/B/S/P. Combinations: 108 × 3 × 8 = 2,592.

**forward_backward.** Each unit takes its most probable slide given all units. No settings of its own.
- shared: F/B/S/P. Combinations: 108 × 8 = 864.

**softdtw.** dtw with a smoothed max.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `gamma` | 0.2 | smoothing; small = close to dtw | 0.05, 0.2, 1.0 | near-dtw, default, strongly smoothed |
| `mode` | expected | `expected`: each unit's most likely slide (may step back); `path`: one monotone path | expected, path | the two decoders differ in kind |

- shared: F/S/P. Combinations: 36 × 6 × 8 = 1,728.

### 2.4 Boundary-first

**pelt.** Best segmentation with a fixed cost per slide change.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `change_penalty` | 1 | cost of each slide change; higher = fewer changes | 0.5, 1, 2, 4 | on z-scored emissions one change should cost about one unit's evidence; a doubling scale around it |

- shared: P. Combinations: 4 × 4 × 8 = 128.

**bocpd.** Change-point hypotheses, traced back from the best final one.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `hazard` | 0.05 | probability of a slide change at each unit; higher = more changes | 0.05, 0.1, 0.25, 0.5 | a change rate is about 1 / (units per slide); the tuning lectures have 1.5 to 43 units per slide (median 3.7), so the rate runs from about 0.02 to 0.67 (median 0.27) |
| `max_hypotheses` | 64 | how many hypotheses are kept | fixed: 64 | a search width, not a model choice; a wider beam only costs time |

- shared: P. Combinations: 4 × 4 × 8 = 128.

**wbs2.** Recursive random intervals; the m − 1 strongest splits.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `n_intervals` | 40 | random sub-intervals per segment; more = a more thorough split search | 20, 40, 80 | half and double |
| `seed` | 0 | the random draw | fixed: 0 | tuning a seed would fit noise |

- shared: P. Combinations: 4 × 3 × 8 = 96.

**texttiling.** Cut at the m − 1 deepest cohesion dips, then label the runs.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `window` | 3 | units on each side of a gap | 3, 6 | as in cohesion_dp |

- shared: P. Combinations: 4 × 2 × 8 = 64.

**texttiling_dp.** The same depth score; m − 1 cuts chosen by DP.
- `window`: 3, 6 (as texttiling). shared: P. Combinations: 64.

### 2.5 Transport

**opw.** Balanced transport rewarded near the diagonal.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `lambda1` | 1 | weight of the diagonal reward | 0.5, 1, 2 | half and double |
| `lambda2` | 0.1 | smoothing towards a band around the diagonal (also the entropic regulariser) | 0.05, 0.1, 0.3 | sharper, default, smoother |
| `sigma` | 1 | width of that band | 0.5, 1, 2 | narrow, default, wide |
| `n_iter` | 2000 | Sinkhorn iterations | fixed | an iteration cap; the solver stops when it converges |

- shared: P. Combinations: 4 × 27 × 8 = 864.

**ot_unbalanced.** Transport in which the equal shares are only encouraged.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `eps` | 0.05 | smoothing | 0.02, 0.05, 0.1 | sharper, default, smoother |
| `reg_m` | 0.5 | how strongly slides must get equal shares; large = balanced | 0.1, 0.5, 2 | nearly free, default, nearly balanced |
| `drop_mass` | 0 | a unit sending less than this share of its mass is dropped and refilled | 0, 0.5 | off, and drop the units the transport mostly discards |
| `drop_fill` | carry | fill for dropped units | fixed: carry | argmax would give almost the transport's own choice; none gives -1 |
| `n_iter` | 2000 | iterations | fixed | an iteration cap |

- shared: P. Combinations: 4 × 18 × 8 = 576.

### 2.6 Consensus

**Decided: the members use their TUNED settings, each with its own tuned measure.** A
consensus method is therefore tuned after its members, and its grid holds only its own
settings (the model is fixed, section 3.1). Each member reads the matrix of
its own tuned measure, so one lecture's consensus run reads up to as many matrices as it
has members (all are units × slides, so their outputs line up).

**This needs a code change** (made in step 9; one line each):
- `consensus.align_ensemble`: take, per voter, its matrix, cfg and own settings (and
  llm_single_call's tuned settings) instead of calling `VOTERS[name](S, cfg=cfg)` on one
  matrix with one cfg.
- `consensus.score_fusion_posterior` / `align_score_fusion`: take, per member, its matrix,
  cfg and settings, and pass them to `softdtw_posterior`, `cohesion_posterior` and
  `nw_posterior`, plus the soft Needleman-Wunsch `gamma` as a setting.
- `llm.rerank_candidates` / `align_llm_rerank_kbest`: take, per candidate method, its
  matrix, cfg and settings instead of one matrix and one cfg.
- `registry.run_method` (and the `Method.settings` listing): accept these per-member
  inputs, which the tuning runner fills from the members' tuned choices.

**ensemble.** A vote of five matrix voters and llm_single_call.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `smooth` | true | width-3 median removes one-unit spikes | true, false | on and off |
| `members` | the five | which matrix voters vote | fixed | decided (docs/METHODS.md) |
| the voters | - | dp, dtw, needleman_wunsch, cohesion_dp, pelt | fixed: each at its tuned settings and tuned measure | decided |
| the LLM voter | - | llm_single_call | fixed: llm_single_call with the one model (section 3.1) | decided |

- What is left of ensemble's own settings: only `smooth`. No shared cfg remains (each
  voter brings its own tuned cfg), no measure remains (each voter reads its own), and no
  model (the LLM voter uses the one model, qwen3-30b-a3b-instruct-2507).
- Combinations: 2. New LLM calls: none (the LLM voter's answer is llm_single_call's,
  already cached).

**score_fusion.** The sum of three per-unit posteriors.

| member | settings it takes (from the tuned member) | what does not map |
|---|---|---|
| soft dtw | softdtw's tuned `gamma`, F/S/P and measure | softdtw's `mode`: the posterior is always the marginal over all paths |
| cohesion HMM | cohesion_dp's tuned `window`, `strength`, F/B/S/P and measure | - |
| soft Needleman-Wunsch | needleman_wunsch's tuned `gap_segment`, `gap_slide`, stay_bonus, pos_weight and measure | its `gamma`: the hard needleman_wunsch has no such setting |

The soft Needleman-Wunsch `gamma` (0.3 in the code) becomes score_fusion's only own setting:

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| soft-NW `gamma` | 0.3 | how soft this member's posterior is; small = close to the hard alignment, large = spread over many slides, so it weighs less in the sum | 0.1, 0.3, 1.0 | sharper, the code's value, smoother |

- Combinations: 3.

**mbr.** Vote over dtw's k best paths. It has no member methods: the paths come from
dtw's lattice under mbr's own cfg and measure, so the decision above does not change it.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| `k` | 100 | how many near-best paths vote; more = further from dtw | 20, 100, 300 | few, default, many (300 is valid since the k > 255 fix) |
| `temperature` | none | none: equal votes; a number: votes weighted by path score (small = close to dtw) | none, 1.0 | equal votes, and score-weighted on the z-scale |

- shared: F/S/P. Combinations: 36 × 6 × 8 = 1,728.

### 2.7 LLM methods

See section 3 for the model, temperature and cache. The model is NOT a setting: every
LLM method uses the one model, qwen3-30b-a3b-instruct-2507 (decided 2026-10-12, section
3.1).

**llm_single_call.** One call; the model assigns every unit a slide, in blocks.

| setting | default | values | reason |
|---|---|---|---|
| model | qwen3-30b-a3b-instruct-2507 | fixed: qwen3-30b-a3b-instruct-2507 | section 3.1 (one model) |
| `slide_chars` | 3000 | fixed: 3000 | no tuning slide is longer than 3,000 characters (the longest has 1,642): the cap cuts nothing |
| `row_chars` | 800 | fixed: 800 | no tuning unit is longer than 800 characters (the longest has 541): the cap cuts nothing |
| `max_tokens` | min(4000, 256 + 12 per unit) | fixed | an answer budget: it only has to be large enough. A cut-off answer is counted (`truncated`) and checked after the run |
| measure | hybrid | fixed: hybrid | the method reads the matrix only for its fallback (argmax) when an answer is unusable; the answer itself never sees the matrix |

- Combinations: 1. Calls: 60.

**llm_monotone.** One call; the model says where each slide starts.
- model: fixed, qwen3-30b-a3b-instruct-2507 (section 3.1). `slide_chars`, `row_chars`: fixed (as above). `max_tokens`
  (min(1500, 64 + 8 per slide)): fixed (a budget; at 40 slides it is 384 tokens for a
  list of 40 numbers). measure: fixed, hybrid (fallback only).
- Combinations: 1. Calls: 60.

**llm_monotone_constrained.** One call under a server-enforced JSON schema; the model gives units per slide.
- model: fixed, qwen3-30b-a3b-instruct-2507 (section 3.1). `slide_chars`, `row_chars`: fixed (as above). `max_tokens` = 900: fixed
  (40 small numbers need far less). measure: fixed, hybrid (fallback only).
- Combinations: 1. Calls: 60.

**hierarchical.** Split, leaf and boundary calls down a tree of page ranges. It does not read the matrix.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| model | qwen3-30b-a3b-instruct-2507 | | fixed: qwen3-30b-a3b-instruct-2507 | section 3.1 (one model) |
| `use_boundary_refinement` | true | re-ask the units around each page change | true, false | on and off; "off" needs no new call (its calls are a prefix of "on") |
| `leaf_max_pages` | 3 | a node with at most this many pages (and few packets) is answered in one leaf call | 3, 5 | bigger leaves: the model sees more pages at once and fewer split decisions are made, each of which can go wrong |
| `packet_size`, `leaf_max_packets`, `packet_chars` | sized per lecture | units per packet (about one slide's worth), packets per leaf | fixed: sized | they follow the lecture's units per slide; a fixed override would be wrong for both short and long lectures |
| `boundary_context_before`, `_after` | 2, 2 | units shown on each side of a page change | fixed: 2 | 6 units per boundary call already cover a typical slide change; more would re-ask settled units |
| `ask_for_reason` | false | a reason per answer | fixed: false | it triples the answer tokens, and the reason is written after each label |
| `page_preview_chars` | 1200 | slide text cap in split calls | fixed | cuts only the few slides over 1,200 characters, and only in split calls |
| `page_full_chars`, `section_chars` | 8000, 900 | text caps in leaf and boundary calls | fixed | they cut nothing in the tuning set |
| `tokens_per_unit`, `tokens_per_unit_with_reason`, `max_answer_tokens` | 40, 120, 8192 | answer budgets | fixed | budgets only; cut-off answers are counted |
| `max_depth`, `max_calls` | 32, 600 | safety ceilings | fixed | not reached: at most 67 calls per tuning lecture are expected |
| measure | - | - | none | not read |

- Combinations: 2 × 2 = 4. Calls: about 2,350 (section 5).

**llm_rerank_kbest.** Five methods' alignments are shown; the model picks one.

| setting | default | values | reason |
|---|---|---|---|
| model | | fixed: qwen3-30b-a3b-instruct-2507 | section 3.1 (one model) |
| the candidates | dtw, dp, pelt, needleman_wunsch, cohesion_dp at their defaults | fixed: each at its tuned settings and tuned measure | decided (section 2.6) |
| `slide_chars`, `row_chars` | 3000, 800 | fixed | they cut nothing (as above) |
| `max_tokens` | 64 | fixed | the answer is one number |

- No shared cfg or measure is left: the candidates bring their own.
- Combinations: 1. Calls: 60.

**fusion.** dp with the hierarchical answer as its positional prior.

| setting | default | how it changes the alignment | values | reason |
|---|---|---|---|---|
| model | | | fixed: qwen3-30b-a3b-instruct-2507 | section 3.1 (one model) |
| hierarchical's settings | as hierarchical | the curve fusion follows | fixed: hierarchical's tuned settings | fusion is tuned after hierarchical, so every call is a cache hit |
| `pos_weight` | 10 | how strongly dp is held to the hierarchical curve (it replaces cfg's pos_weight) | 5, 10, 20 | half and double |
| shared cfg | defaults | dp's transitions | forward_penalty, backward_penalty, stay_bonus: 3 × 3 × 3 | as in section 1.2 |
| measure | hybrid | dp's emissions | the 8 | it reads the matrix |

- Combinations: 3 × 27 × 8 = 648. New LLM calls: none.

## 3. The LLM part

### 3.1 The LLM

**Decided, 2026-10-12: single model qwen3-30b-a3b-instruct-2507; glm-5.3-flash and
qwen3.5-397b-a17b dropped.** The model is no longer a tuned setting: every LLM and mixed
method (llm_single_call, llm_monotone, llm_monotone_constrained, hierarchical,
llm_rerank_kbest, ensemble's LLM voter, fusion) uses qwen3-30b-a3b-instruct-2507. The
choice of three below is kept as the history of the design; the two other models' calls
already made in the first, cancelled tuning job stay in the LLM cache but none of their
rows is in the tuning results.

History (the design as first proposed):

The endpoint's model list (one GET of `/v1/models`, 2026-10-12) has 14 models. Sizes and context
lengths are from the provider's model page.

| model | size | context | why |
|---|---|---|---|
| `qwen3-30b-a3b-instruct-2507` | 30.5B total, 3.3B active (MoE), FP8 | 256K | **the fast one.** the provider lists it as "good performance, fast". It is an instruct model without a thinking mode, so no hidden reasoning tokens eat the answer budgets. It is the current default, so earlier work is comparable. |
| `qwen3.5-397b-a17b` | 397B total, 17B active (MoE), int4 | 256K | **the powerful one.** The largest open-weight model on the list, which the provider describes as the most powerful open-weight model it hosts. Its thinking can be switched off with the same switch. |
| `glm-5.3-flash` | 320B total, 18B active (MoE) | 1M | **strong and cheap to serve, from another family.** A large model whose architecture the provider says "significantly" reduces serving cost. A third family (not Qwen) keeps the results from being one family's behaviour. |

All three hold the largest tuning prompt (about 16,000 tokens) many times over. (Only
qwen3-30b-a3b-instruct-2507 is used; see the decision above.)

Not chosen, with the reason:

| model | reason |
|---|---|
| meta-llama-3.1-8b-instruct | small and old; the provider's fastest, but the 3B-active Qwen covers "fast" with a newer model |
| openai-gpt-oss-120b | a reasoning model: its reasoning tokens count against tight answer budgets (64 tokens for llm_rerank_kbest) |
| deepseek-v4-flash-0731 | strong, but the provider warns of biased answers and recommends temperature 1.0; one large MoE from another family is enough |
| mistral-medium-3.5-128b | large and dense, so slow per token |
| gemma-4-31b-it | dense 31B: slower per token than the 3B-active model and smaller than the two large ones |
| qwen3.6-35b-a3b, qwen3.8-27b | the same family as the fast pick; qwen3.8 thinks by default |
| devstral-2-123b, qwen3-coder-next | tuned for code |
| qwen3-omni-30b-a3b-instruct | tuned for audio and voice |
| apertus-70b-instruct-2509 | listed as "loading" at the time of the list |

**Decided: two checks before any tuning call**, on qwen3.5-397b-a17b and glm-5.3-flash
(one call each): that `enable_thinking = false` really switches thinking off, and that the
server enforces the JSON schema that llm_monotone_constrained sends. (History: the checks
were made in step 9 on all three models and passed; the two other models were dropped
afterwards.)

### 3.2 Fixed for every LLM method

- **Temperature 0.** Fixed, as decided (the provider recommends 0.6 to 1.0 for these models; that
  is not used).
- **Thinking off** (`disable_thinking = true`). Fixed. Thinking text would use up the
  answer budgets.
- Endpoint, timeout and retries: config/llm.toml, unchanged.

### 3.3 The LLM methods' other settings

All listed in section 2.7. In short:

| method | tuned (the model is fixed: qwen3-30b-a3b-instruct-2507) | fixed |
|---|---|---|
| llm_single_call | nothing | text caps (cut nothing), max_tokens (budget), measure (fallback only) |
| llm_monotone | nothing | the same |
| llm_monotone_constrained | nothing | the same |
| hierarchical | use_boundary_refinement (true, false), leaf_max_pages (3, 5) | packet sizing, boundary context, reasons, text caps, budgets, ceilings |
| llm_rerank_kbest | nothing | the candidates (tuned members), text caps, max_tokens |
| fusion | pos_weight (5, 10, 20), forward/backward penalty, stay bonus, measure (8) | hierarchical's settings (its tuned ones) |
| ensemble | smooth | the voters (tuned members), the LLM voter (llm_single_call with the one model) |

### 3.4 What is cached, and what forces new calls

Every answer is stored in `data/llm_cache.sqlite3` under the hash of the **whole request**:
model, both messages, temperature, max_tokens, response format, thinking switch.

A new call is made when anything in the request changes:
- the model;
- any text in the prompt: a text cap that cuts something, the packet or leaf sizes, the
  boundary context, `ask_for_reason`;
- the candidates shown to llm_rerank_kbest, which change with its members' settings and
  measures (fixed once the members are tuned);
- `max_tokens` or the response format.

No new call is made when:
- the measure changes for llm_single_call, llm_monotone or llm_monotone_constrained (the
  prompt has no matrix in it);
- hierarchical runs with `use_boundary_refinement = false` after `true` (the same split
  and leaf calls);
- hierarchical's leaf size changes, for every split call above the leaves (the same
  prompts);
- fusion's own settings change (it re-uses hierarchical's answers);
- ensemble's settings change (its LLM voter re-uses llm_single_call's answer).

A failed call (network, HTTP error, no answer after the retries) is not cached; it is
asked again on the next run. An unusable answer (prose, broken JSON, cut off) is cached
and counted.

## 4. The tuning rule

1. **Only the 60 tuning lectures.** Every setting is chosen on them. The 140 test lectures
   are never used for tuning, not even to look at.
2. **The metric.** Each method gives one slide per unit. Its accuracy on a lecture is the
   share of the lecture's units whose slide equals the true slide (`gold_labels`; a -1 is
   always wrong). The selection metric is the **mean of this accuracy over the 60
   lectures**, each lecture counting once. (Weighting by units would let the few long
   MaViLS lectures, up to 615 units, outweigh the many 40-unit ones.)
3. **One choice per method.** For each method, the whole grid (measure included) is run
   and one combination is chosen.
4. **Near-ties.** Every combination within **0.5 percentage points** of the best mean
   accuracy counts as tied. (0.5 points is about one unit wrong in each of 12 of the 60
   lectures; smaller differences are within what a few lectures can swing.) Among the
   tied ones:
   1. the **cheaper** one wins: for LLM settings, fewer successful calls on the tuning
      set, then the smaller sum of their durations (section 6); for non-LLM settings,
      only settings that change the amount of work count (mbr `k`, wbs2 `n_intervals`,
      multiscale_dtw `radius`, jump_dtw `max_jump`), and the smaller value is cheaper;
   2. then the one **closest to the defaults** (fewest settings away from the default;
      the default measure is hybrid);
   3. then the higher mean accuracy.
5. **The order.**
   1. The 8 matrices for the 60 tuning lectures are built first (not tuning; a step of
      its own).
   2. The base methods: the 22 non-LLM base methods (argmax to ot_unbalanced) and mbr,
      in any order; and the LLM base methods llm_single_call, llm_monotone,
      llm_monotone_constrained and hierarchical (refinement on before off).
   3. Then the consensus and mixed methods, each after its members:
      - score_fusion after softdtw, cohesion_dp and needleman_wunsch;
      - llm_rerank_kbest after dtw, dp, pelt, needleman_wunsch and cohesion_dp;
      - ensemble after dp, dtw, needleman_wunsch, cohesion_dp, pelt and llm_single_call;
      - fusion after hierarchical, with hierarchical's tuned settings.

## 5. The size of the tuning

### 5.1 Combinations

| method | combinations | × 60 lectures |
|---|---|---|
| argmax | 8 | 480 |
| dtw | 288 | 17,280 |
| multiscale_dtw | 864 | 51,840 |
| needleman_wunsch | 864 | 51,840 |
| gotoh | 2,592 | 155,520 |
| drop_dtw | 1,728 | 103,680 |
| anchored_dp | 864 | 51,840 |
| semimarkov | 864 | 51,840 |
| adaptive_prior | 1,296 | 77,760 |
| dp | 864 | 51,840 |
| rank_dp | 864 | 51,840 |
| cohesion_dp | 5,184 | 311,040 |
| jump_dtw | 2,592 | 155,520 |
| forward_backward | 864 | 51,840 |
| softdtw | 1,728 | 103,680 |
| pelt | 128 | 7,680 |
| bocpd | 128 | 7,680 |
| wbs2 | 96 | 5,760 |
| texttiling | 64 | 3,840 |
| texttiling_dp | 64 | 3,840 |
| opw | 864 | 51,840 |
| ot_unbalanced | 576 | 34,560 |
| score_fusion | 3 | 180 |
| mbr | 1,728 | 103,680 |
| **non-LLM total (24 methods)** | **25,115** | **1,506,900** |
| llm_single_call | 1 | 60 |
| llm_monotone | 1 | 60 |
| llm_monotone_constrained | 1 | 60 |
| hierarchical | 4 | 240 |
| llm_rerank_kbest | 1 | 60 |
| fusion | 648 | 38,880 |
| ensemble | 2 | 120 |
| **LLM and mixed total (7 methods)** | **658** | **39,480** |
| **all 31** | **25,773** | **1,546,380** |

The non-LLM compute time is not estimated: no method may be timed before the go.

### 5.2 LLM calls and hours (a planning estimate, NOT the reported cost)

Wall-clock hours assume 20 requests per minute (config/llm.toml; the endpoint limits each
account, so all jobs share it): hours = calls / 1,200.

| method | calls | how counted | hours at 20/min |
|---|---|---|---|
| llm_single_call | 60 | 60 lectures × 1 call (one model) | 0.05 |
| llm_monotone | 60 | the same | 0.05 |
| llm_monotone_constrained | 60 | the same | 0.05 |
| hierarchical | about 2,350 | about 1,420 calls at leaf_max_pages 3 (400 split, 400 leaf, 620 boundary) plus about 930 new ones at 5 (its split calls are shared); refinement off adds none | about 2.0 |
| llm_rerank_kbest | 60 | 60 lectures (one candidate set per lecture) | 0.05 |
| fusion | 0 | re-uses hierarchical | 0 |
| ensemble | 0 | re-uses llm_single_call | 0 |
| **total** | **about 2,590** | | **about 2.2** |

How the hierarchical numbers were estimated: the tree was walked on each tuning lecture
with its true slides standing in for the model's answers (splits where the true slides
change sides, leaves as the true slides, a boundary call at each change). Real answers
move the tree, so this is a planning figure; the most calls on one lecture was 67, far
below `max_calls` (600). With the tree shapes shared fully, the count would be about
1,580 instead of 2,350.

The total is not very large. **hierarchical drives it (about 90 %)**, through the second
leaf size (about + 930 calls). The 8 measures add no LLM calls:
the single-call prompts hold no matrix, and llm_rerank_kbest's candidates are fixed by
its tuned members. If calls run one after another and one call lasts longer than 3
seconds, the time is set by the call durations, not by the rate; this estimate does not
include that.

Tokens, for scale: a whole-lecture prompt totals about 275,000 tokens over the 60 tuning
lectures, so each single-call method sends about 0.28 million prompt tokens (one
model).

## 6. The cost rule

### 6.1 The rule (as decided)

- **LLM cost.** Only successful calls count. Failed attempts are never counted as cost.
- **The time of a call** is only the duration of its successful attempt. Retries and
  sleep (the rate-limit wait and the backoff) are ignored, and so is queueing.
- So an LLM method's cost on a lecture = the number of successful calls and the sum of
  their durations, plus their prompt and answer tokens.
- **Non-LLM cost** = compute time (the time of `run_method` on ready matrices; building
  a matrix is the measure's cost, reported apart, once per measure).
- **Mixed methods** (ensemble, fusion, llm_rerank_kbest) = both: their calls and the
  compute time of their matrix part.
- This reported cost is kept apart from the planning estimate in section 5.2.

### 6.2 What methods/llm_client.py records today

| needed | today | |
|---|---|---|
| failed calls not counted | yes, in effect: a call that finally fails raises `LLMCallFailed` and is neither counted in `calls` nor cached | but it is not recorded anywhere either |
| a success flag per call | no | success is only implied by a cache entry |
| the duration of the successful attempt | **no** | nothing is timed; the cache record has only a `created` timestamp |
| retries kept apart | **no** | retries, backoff (2 · 1.6^attempt s, up to 60 s) and the rate-limit wait happen inside `OpenAICompatClient.chat` and are invisible to the caller |
| tokens | yes | `prompt_tokens` and `completion_tokens` per answer, in the cache record and in `LLM.stats()` |
| cost when the answer comes from the cache | **no** | a cache hit counts as a hit with no cost; ensemble, fusion and any rerun would report no cost |

### 6.3 The change (decided; made in step 9)

The rule as worded: **retries and sleep (rate-limit wait, backoff) are
ignored; the only time that counts is the duration of the successful attempt. Failed
attempts are never counted as cost.**

1. **`OpenAICompatClient.chat`**: time each attempt with `time.perf_counter()` from just before
   `urlopen` to just after the answer is read and parsed. The rate-limit wait
   (`_throttle`) and the backoff sleep are outside this window. Return, with the answer:
   `duration_s` (the successful attempt only), `attempts`, and the failed attempts (count
   and error kinds). The waits may be returned too, but only for information, never as
   cost.
2. **`LLM.ask`**: store `duration_s` and `attempts` in the cache record next to the
   tokens. Keep separate counters: `call_seconds` (sum of `duration_s` of successful
   calls), `failed_attempts` (retried attempts that later succeeded) and `failed_calls`
   (calls that raised `LLMCallFailed`); neither failure counter enters the cost.
3. **The cost belongs to the answer, not to the run.** On a cache hit, add the stored
   `duration_s` and tokens to the method's cost as well (as "cost of the answers used"),
   and keep "paid in this run" apart. Then a method's cost does not depend on whether its
   answers were already cached, and ensemble and fusion report the cost of the answers
   they use.
4. **Per method and lecture**: the methods' `diagnostics` already take the difference of
   `LLM.stats()` before and after; with the fields above they would carry the successful
   calls, their summed duration and tokens.
5. **Non-LLM**: the tuning runner (not written yet) times `run_method` per lecture with
   `time.perf_counter()`.
6. **Old cache entries** have no duration. The cache now holds only step 5's discipline
   answers, so no method answer would lack one if the change comes before the tuning.

One limit stays: the client cannot see how long a request waited in the endpoint's own queue
inside a successful attempt. The measured duration is the whole successful HTTP request
(server queue, reading the prompt, writing the answer).

## 7. Decisions (2026-10-12)

1. **Consensus members use their tuned settings, each with its own tuned measure**:
   ensemble's five voters and its LLM voter (llm_single_call; one model since decision 6),
   llm_rerank_kbest's five candidates, and score_fusion's three members (softdtw,
   cohesion_dp, needleman_wunsch, where their settings map; the soft Needleman-Wunsch
   `gamma` becomes score_fusion's own setting). These methods are tuned after their
   members, and their grids hold only their own settings (plus the model for
   llm_rerank_kbest; fixed since decision 6). This needs the code change listed in section 2.6.
2. **All 8 similarity measures are tried** by every method that reads a matrix as its
   measure (section 1.1).
3. **The measures' own settings stay at their defaults** (config/similarity.toml). The
   paper states this as a fixed choice.
4. **Two model checks before tuning**, on qwen3.5-397b-a17b and glm-5.3-flash: thinking
   off (`enable_thinking = false`) and the JSON schema enforced (section 3.1). (History:
   made in step 9; both models were dropped by decision 6.)
5. **The llm_client change of section 6.3 is made before the tuning.** Retries and sleep
   (rate-limit wait, backoff) are ignored; only the duration of the successful attempt
   counts; failed attempts are never counted as cost.
6. **2026-10-12: single model qwen3-30b-a3b-instruct-2507; glm-5.3-flash and
   qwen3.5-397b-a17b dropped.** The model is no longer a tuned setting; every LLM and
   mixed method (ensemble's LLM voter and fusion included) uses this model (section 3.1).
   The counts of sections 2.7, 3 and 5 follow it.

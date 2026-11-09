# Importance labelling guideline

Every transcript unit of the 140 test lectures gets one importance label, 0, 1 or 2. The
label says how much the unit carries the content of its **true slide**: the slide the
corpus's ground truth gives it. The labels are judgments by Claude Opus 5.5 (Claude Code
agents) following this guideline, made on the date of labelling (recorded in
`data/importance/labels_meta.json`).

## What you get

One plain-text pack per lecture (a long lecture comes in parts, `<lecture>.part<i>.txt`):

- a header: `lecture`, `corpus`, `n_slides`, `n_units` (and `part` for a part);
- `SLIDES`: `slide <k>: <text>`, the slide's text on one line (its lines joined by
  " / "; cut at 700 characters, marked "[...]"). The text is the PDF text layer, the
  corpus's own OCR or our OCR, so it can be noisy;
- `UNITS`: one line per unit, in speaking order: `u<index> [slide <k>] <text>`, where k
  is the unit's true slide.

Use only the pack. Do not look for other files, results or methods.

## What you label

Each unit, judged against the content of its true slide (`[slide k]`), with the units
before and after it as context. The parts of a long lecture follow each other: read them
in order.

| label | name | the unit ... |
|---|---|---|
| **2** | core | states or explains content that is on the slide, or the slide's key point: its claim, definition, formula, figure, result, step or bullet |
| **1** | supporting | is related to the slide but is not its core statement: elaboration, example, motivation, aside, repetition, an answer to a question, a partial or vague reference to the slide |
| **0** | filler | greetings, admin and logistics, transitions ("next slide", "let's move on"), hesitations and fragments, jokes, off-topic talk, technical trouble, unintelligible ASR output |

## Rules

1. Judge the content, not the grammar or the speech-recognition errors. A garbled unit
   whose meaning is still clear is labelled by its meaning.
2. When torn between two levels, choose 1.
3. A unit that has both core content and filler gets the higher level only if the core
   part is clear.
4. The label does not depend on the slide's own quality: a slide with little or noisy
   text can still have core units (what the speaker says about what the slide shows).
5. No target distribution: label each unit on its own; any mix of 0s, 1s and 2s is fine.
6. Every unit gets exactly one label. Do not skip units, do not add units.

## What you write

One CSV file per batch, named `<batch>.csv` (for example `b01.csv`), UTF-8, with this
header and one row per unit of every file of the batch:

    corpus,lecture_id,unit_index,label
    m3av_b,CHI-003EC,0,0
    m3av_b,CHI-003EC,1,2

- `corpus` and `lecture_id`: exactly as in the pack's header;
- `unit_index`: the unit's number without the "u" (`u17` -> `17`);
- `label`: 0, 1 or 2.

Nothing else: no comment column, no blank or extra lines. `scripts/importance_merge.py`
refuses the files if a unit is missing, given twice, unknown, or has another label.

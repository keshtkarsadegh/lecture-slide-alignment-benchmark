# Recurring cases met during labelling

Written after the labelling from the labellers' reports; not part of the guideline
(`docs/IMPORTANCE.md`) the labellers used.

## Reliability

A second labelling of 10 test lectures (897 units, `data/importance/second/`) agrees
with the full labelling on 89.9% of the units exactly and on 99.8% within one level;
Cohen's weighted kappa is 0.82 with linear and 0.84 with quadratic weights. Almost all
disagreements are between neighbouring levels (79 units 1 against 2, 10 units 0 against
1, 2 units 0 against 2). The per-lecture table is `data/importance/agreement.csv`
(`python scripts/importance_merge.py --agreement data/importance/raw
data/importance/second`); the two lectures that agree least are short ones,
m3av_a/CHI-66B82 (41 units, 56% exact) and mcd/sp_2317 (27 units, 63% exact).

A blind check of 100 sampled units (`data/importance/local/check_sample.md`, drawn by
`scripts/importance_sample.py`; not distributed, since it quotes the corpora's text, and
drawn again by the same seed) was labelled independently by a second Claude model,
Claude Fable 5.1, without seeing the labellers' labels (`data/importance/check_manual.csv`).
It agrees with the labellers on 81% of the units exactly and on 100% within one level:
15 of the 19 disagreements are units the checker called core (2) and the labellers
supporting (1), and the checker never disagreed by two levels (`python
scripts/importance_sample.py --check data/importance/check_sample_key.csv
data/importance/check_manual.csv`). The checker is from the same model family as the
labellers, so this is a consistency check, not an independent human validation.

## Cases

The labelling agents reported the cases below, which the guideline's rules do not
settle on their own. This section records how they were handled. Several cases were
handled differently by different agents, as noted.

- **A host introducing the speaker over a title or admin slide.** The speaker's
  biography and thanks mostly 0; the research topic mostly 1; reading out the talk title
  1 or 2, depending on the agent.
- **The speaker stating the talk title on the title slide.** Mostly 2. Only the
  speaker's name or affiliation: 1, though some agents gave 0 and some 2.
- **Acknowledgement slides naming people or funders.** Naming them: 1 or 2, depending
  on the agent; bare thanks: 0.
- **A unit whose content belongs to the previous or next slide** (the speech running
  ahead of or behind the slide change). Judged against the true slide, as the guideline
  says; usually 1.
- **Questions and answers under an unrelated end slide.** Substantive answers 1,
  logistics 0.
- **Blackboard-style decks whose slide text is the final board state** (MLS-E8022).
  Only content matching the board text 2; other teaching content 1.
- **Slide text cut at 700 characters.** Some agents assumed the hidden text covers the
  spoken content (2); others applied the tie rule (rule 2: 1).
- **Sentence fragments cut by unit boundaries.** 0 unless the meaning is clear.

These differences between agents are measured by the agreement check: the second
labelling of 10 lectures compared with the full labelling in
`data/importance/agreement.csv`. They are a limitation of the labels.

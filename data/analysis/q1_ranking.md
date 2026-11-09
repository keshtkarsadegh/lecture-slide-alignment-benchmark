# Q1: the methods by mean test accuracy, with cost

Mean accuracy over the 140 test lectures, in %. Rank 1 = highest; equal accuracies share a rank. Each cost: total over the lectures / per lecture; '-' where the method's kind has no such cost. compute s: the method's own time; build s: the similarity tables it reads (with the model loads); LLM calls, LLM s and tokens (prompt + completion): the answers it used (successful calls, successful attempt only). A combined method carries its members' cost.

| rank | method | kind | accuracy % | compute s | build s | LLM calls | LLM s | tokens |
|---|---|---|---|---|---|---|---|---|
| 1 | score_fusion | non-LLM | 72.21 | 29.52 / 0.211 | 3,181 / 22.72 | - | - | - |
| 2 | gotoh | non-LLM | 72.07 | 8.08 / 0.0577 | 3,181 / 22.72 | - | - | - |
| 3 | needleman_wunsch | non-LLM | 71.37 | 2.43 / 0.0173 | 3,181 / 22.72 | - | - | - |
| 4 | dtw | non-LLM | 71.34 | 0.262 / 0.00187 | 3,181 / 22.72 | - | - | - |
| 5 | softdtw | non-LLM | 71.15 | 0.744 / 0.00532 | 3,181 / 22.72 | - | - | - |
| 6 | ensemble | mixed | 71.07 | 3.94 / 0.0281 | 3,872 / 27.66 | 140 / 1.00 | 1,632 / 11.66 | 885,786 / 6,327 |
| 7 | adaptive_prior | non-LLM | 71.06 | 0.323 / 0.00231 | 3,181 / 22.72 | - | - | - |
| 8 | anchored_dp | non-LLM | 70.94 | 0.35 / 0.0025 | 3,873 / 27.66 | - | - | - |
| 9 | drop_dtw | non-LLM | 70.90 | 1.90 / 0.0136 | 3,181 / 22.72 | - | - | - |
| 10 | multiscale_dtw | non-LLM | 70.78 | 0.565 / 0.00404 | 3,181 / 22.72 | - | - | - |
| 11 | semimarkov | non-LLM | 70.48 | 25.22 / 0.18 | 3,181 / 22.72 | - | - | - |
| 12 | llm_rerank_kbest | mixed | 70.23 | 3.94 / 0.0281 | 3,181 / 22.72 | 140 / 1.00 | 166 / 1.18 | 982,437 / 7,017 |
| 13 | mbr | non-LLM | 70.22 | 1.52 / 0.0108 | 3,181 / 22.72 | - | - | - |
| 14 | pelt | non-LLM | 69.83 | 0.466 / 0.00333 | 3,181 / 22.72 | - | - | - |
| 15 | jump_dtw | non-LLM | 67.99 | 0.293 / 0.00209 | 3,181 / 22.72 | - | - | - |
| 16 | cohesion_dp | non-LLM | 67.98 | 0.593 / 0.00423 | 3,181 / 22.72 | - | - | - |
| 17 | fusion | mixed | 67.70 | 0.296 / 0.00212 | 3,181 / 22.72 | 1,541 / 11.01 | 5,933 / 42.38 | 3,979,274 / 28,423 |
| 18 | dp | non-LLM | 67.55 | 0.213 / 0.00152 | 3,181 / 22.72 | - | - | - |
| 19 | forward_backward | non-LLM | 65.98 | 0.236 / 0.00169 | 3,181 / 22.72 | - | - | - |
| 20 | bocpd | non-LLM | 62.15 | 0.898 / 0.00641 | 3,181 / 22.72 | - | - | - |
| 21 | rank_dp | non-LLM | 61.57 | 0.314 / 0.00224 | 3,181 / 22.72 | - | - | - |
| 22 | ot_unbalanced | non-LLM | 61.24 | 1.12 / 0.00803 | 3,873 / 27.66 | - | - | - |
| 23 | opw | non-LLM | 54.27 | 5.42 / 0.0387 | 3,873 / 27.66 | - | - | - |
| 24 | llm_single_call | LLM | 51.96 | - | 691 / 4.94 | 140 / 1.00 | 1,632 / 11.66 | 885,786 / 6,327 |
| 25 | argmax | non-LLM | 50.62 | 0.0129 / 9.24e-05 | 3,181 / 22.72 | - | - | - |
| 26 | hierarchical | LLM | 49.65 | - | 0 / 0 | 1,541 / 11.01 | 5,933 / 42.38 | 3,979,274 / 28,423 |
| 27 | wbs2 | non-LLM | 35.47 | 10.01 / 0.0715 | 691 / 4.94 | - | - | - |
| 28 | llm_monotone | LLM | 33.30 | - | 691 / 4.94 | 140 / 1.00 | 522 / 3.73 | 825,873 / 5,899 |
| 29 | llm_monotone_constrained | LLM | 32.12 | - | 691 / 4.94 | 140 / 1.00 | 217 / 1.55 | 829,705 / 5,926 |
| 30 | texttiling | non-LLM | 29.07 | 0.361 / 0.00258 | 0.999 / 0.00714 | - | - | - |
| 31 | texttiling_dp | non-LLM | 28.90 | 1.23 / 0.00881 | 691 / 4.94 | - | - | - |

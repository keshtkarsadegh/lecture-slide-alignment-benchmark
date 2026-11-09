# Q3: each method's best and worst category level

Over all levels of all axes (corpus is not an axis here). Best level: the smallest rank among the methods within the level; worst: the largest. A tie in rank goes to the higher accuracy (best) or the lower (worst); a level list means a tie in both. Accuracy in %.

| method | kind | overall rank | overall % | best level | rank | % | worst level | rank | % |
|---|---|---|---|---|---|---|---|---|---|
| score_fusion | non-LLM | 1 | 72.21 | discipline: Computer and information sciences | 1 | 72.60 | discipline: Engineering and technology | 6 | 57.56 |
| gotoh | non-LLM | 2 | 72.07 | tables: some | 1 | 78.77 | discipline: Social sciences | 9 | 74.35 |
| needleman_wunsch | non-LLM | 3 | 71.37 | noise: low | 2 | 71.40 | noise: medium | 11 | 69.08 |
| dtw | non-LLM | 4 | 71.34 | text_source: corpus_corrected_ocr | 3 | 68.56 | discipline: Social sciences | 11 | 73.81 |
| softdtw | non-LLM | 5 | 71.15 | discipline: Social sciences | 1 | 78.37 | length: short | 17 | 71.05 |
| ensemble | mixed | 6 | 71.07 | tables: some | 5 | 77.55 | discipline: Medical and health sciences | 12 | 71.41 |
| adaptive_prior | non-LLM | 7 | 71.06 | length: short | 1 | 73.94 | discipline: Engineering and technology | 15 | 54.71 |
| anchored_dp | non-LLM | 8 | 70.94 | length: medium | 1 | 72.54 | length: long | 14 | 67.08 |
| drop_dtw | non-LLM | 9 | 70.90 | discipline: Natural sciences (other) | 3 | 77.36 | length: medium | 13 | 69.09 |
| multiscale_dtw | non-LLM | 10 | 70.78 | discipline: Computer and information sciences | 5 | 71.61 | discipline: Medical and health sciences | 14 | 66.35 |
| semimarkov | non-LLM | 11 | 70.48 | discipline: Medical and health sciences | 2 | 73.91 | length: medium | 17 | 68.39 |
| llm_rerank_kbest | mixed | 12 | 70.23 | text_density: low | 5 | 55.70 | noise: medium | 15 | 67.92 |
| mbr | non-LLM | 13 | 70.22 | text_source: pdf_text_layer | 2 | 73.77 | noise: low | 19 | 66.70 |
| pelt | non-LLM | 14 | 69.83 | length: short | 10 | 72.24 | length: medium | 18 | 67.88 |
| jump_dtw | non-LLM | 15 | 67.99 | length: medium | 9 | 69.83 | text_source: tesseract_ocr | 18 | 82.51 |
| cohesion_dp | non-LLM | 16 | 67.98 | length: medium | 10 | 69.53 | discipline: Engineering and technology | 19 | 52.50 |
| fusion | mixed | 17 | 67.70 | text_source: tesseract_ocr | 1 | 87.20 | discipline: Social sciences | 21 | 67.52 |
| dp | non-LLM | 18 | 67.55 | noise: low | 13 | 68.23 | text_source: pdf_text_layer | 19 | 57.86 |
| forward_backward | non-LLM | 19 | 65.98 | discipline: Social sciences | 15 | 70.91 | text_source: pdf_text_layer | 20 | 56.20 |
| bocpd | non-LLM | 20 | 62.15 | discipline: Social sciences | 17 | 70.04 | discipline: Medical and health sciences | 22 | 45.26 |
| rank_dp | non-LLM | 21 | 61.57 | text_source: tesseract_ocr | 20 | 77.63 | text_density: low | 22 | 46.17 |
| ot_unbalanced | non-LLM | 22 | 61.24 | discipline: Social sciences | 20 | 67.68 | text_source: pdf_text_layer | 23 | 49.69 |
| opw | non-LLM | 23 | 54.27 | length: medium | 23 | 62.04 | discipline: Medical and health sciences | 26 | 29.99 |
| llm_single_call | LLM | 24 | 51.96 | text_source: pdf_text_layer | 21 | 55.37 | discipline: Natural sciences (other) | 27 | 27.17 |
| argmax | non-LLM | 25 | 50.62 | length: long | 23 | 42.04 | discipline: Computer and information sciences | 26 | 53.99 |
| hierarchical | LLM | 26 | 49.65 | text_source: tesseract_ocr | 24 | 73.93 | text_source: pdf_text_layer | 26 | 27.13 |
| wbs2 | non-LLM | 27 | 35.47 | discipline: Natural sciences (other) | 26 | 28.84 | discipline: Social sciences | 31 | 25.41 |
| llm_monotone | LLM | 28 | 33.30 | text_source: tesseract_ocr | 27 | 49.84 | text_source: pdf_text_layer | 31 | 10.19 |
| llm_monotone_constrained | LLM | 29 | 32.12 | images: low | 27 | 34.91 | text_density: low | 31 | 24.80 |
| texttiling | non-LLM | 30 | 29.07 | length: medium | 28 | 33.65 | discipline: Medical and health sciences | 31 | 9.58 |
| texttiling_dp | non-LLM | 31 | 28.90 | text_source: tesseract_ocr | 29 | 46.95 | noise: medium | 31 | 24.68 |

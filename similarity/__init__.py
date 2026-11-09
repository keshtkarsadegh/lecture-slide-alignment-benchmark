"""The 8 similarity scorers (step 4; docs/SIMILARITY.md).

Each scorer turns one lecture's unit texts, slide texts and (for the image scorers)
slide images into the (n_units, n_slides) matrix the alignment methods read.
`build_matrix(name, lecture, ctx)` builds one; `SCORERS` lists them with their check
result; the settings are in config/similarity.toml.
"""

from similarity.registry import (SCORERS, Context, Lecture, MissingSlideImage, Scorer,
                                 SimilarityUnavailable, build_matrix, images_zip,
                                 lecture_from_record, load_settings, matrix_key)

__all__ = ["SCORERS", "Context", "Lecture", "MissingSlideImage", "Scorer",
           "SimilarityUnavailable", "build_matrix", "images_zip", "lecture_from_record",
           "load_settings", "matrix_key"]

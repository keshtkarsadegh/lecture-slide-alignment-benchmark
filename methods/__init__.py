"""The 31 transcript-to-slide alignment methods (step 3; docs/METHODS.md).

Every method maps a similarity matrix (units x slides) - and, for the LLM methods, the
slide and unit texts - to one 0-based slide per unit. `METHODS` lists them with their
group and settings; `run_method` runs one by name.
"""

from methods.common import NO_SLIDE, DPConfig, Member
from methods.registry import (GROUPS, MEMBERS, METHODS, Method, gold_labels,
                              lecture_texts, run_method, to_slide_index)

__all__ = ["NO_SLIDE", "DPConfig", "Member", "GROUPS", "MEMBERS", "METHODS", "Method",
           "run_method", "lecture_texts", "gold_labels", "to_slide_index"]

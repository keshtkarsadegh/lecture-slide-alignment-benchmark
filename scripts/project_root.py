"""The project directory outside the repository: PROJECT_ROOT.

The virtual environment (envs/lsab), the Apptainer images and the model weights
(models_lsab/) do not belong in git: they are large, or tens of thousands of files on a
project space with an inode quota. They live in one directory next to the repository
clone, and every job and script finds them through it, so no file in git names a user or
a cluster path. ACTIVATE.md says how to set it.

PROJECT_ROOT is taken from the environment; unset, it is the directory that holds the
clone (the layout the pipeline was run with). Config paths are relative to it; an
absolute path in a config is used as it stands.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """$PROJECT_ROOT, else the parent of the repository root.

    The variable is used as given, not resolved, so every path is built from exactly the
    string that was set (a symlinked location stays as written).
    """
    env = os.environ.get("PROJECT_ROOT", "").strip()
    return Path(env) if env else REPO_ROOT.parent


def project_path(p: str | Path) -> Path:
    """A path from a config: relative ones are under PROJECT_ROOT."""
    p = Path(p)
    return p if p.is_absolute() else project_root() / p

# Activating the environment

The virtual environment, the Apptainer images and the model weights live outside the
repository, in one directory next to the clone on the shared project space. Every job,
script and config finds them through PROJECT_ROOT (scripts/project_root.py); unset, it is
the directory that holds the clone. Set it once per shell when the layout differs:

    export PROJECT_ROOT=/path/to/project-space    # holds envs/, models_lsab/ and the clone
    ENV=$PROJECT_ROOT/envs/lsab

`sbatch` passes the variable on to the jobs.

## The site values: cluster/env.local

The Slurm account, the proxy through which compute nodes reach the internet, and the
address and key of the OpenAI-compatible LLM endpoint are not in the repository. They are
set once in `cluster/env.local`, which `.gitignore` keeps out of git; copy the template
and fill it in:

    cp cluster/env.local.example cluster/env.local    # then edit the values

Source it in every login shell, before submitting a job or running a script that needs
the network or the endpoint:

    source cluster/env.local

It sets `SBATCH_ACCOUNT` (sbatch reads it, so no `#SBATCH` line names an account and a
job is submitted with plain `sbatch cluster/<job>.sbatch`), `CLUSTER_PROXY`,
`LLM_API_BASE` and `LLM_API_KEY` (the key may instead go into the untracked
`config/llm_credentials.toml`). The jobs that download or call the endpoint source the
same file themselves and export `https_proxy`/`http_proxy` from `CLUSTER_PROXY`; the
jobs that run in an Apptainer container with `--containall` (OCR, layout detection, the
model scorers) neither download nor call the endpoint, so they need none of these values.

Activate it (login or compute node):

    export PYTHONNOUSERSITE=1          # keep ~/.local site-packages off sys.path
    source $ENV/bin/activate

It was created once, from the repository root, with the system Python 3.12:

    /usr/bin/python3.12 -m venv $ENV
    PYTHONNOUSERSITE=1 $ENV/bin/pip install --no-cache-dir "pytest==8.4.2"

Step 2 added (login node, through the proxy):

    export https_proxy=$CLUSTER_PROXY http_proxy=$CLUSTER_PROXY    # from cluster/env.local
    PYTHONNOUSERSITE=1 $ENV/bin/pip install --no-cache-dir "pymupdf==1.26.7" "openpyxl==3.1.5"

Step 3 added (login node, through the proxy; `--no-compile` writes no .pyc files):

    PYTHONNOUSERSITE=1 $ENV/bin/pip install --no-cache-dir --no-compile "numpy==2.2.6"

Step 1 (`scripts/download.py`) needs only the standard library; step 2 (`scripts/prepare.py`)
needs PyMuPDF and openpyxl; step 3 (`methods/`) needs numpy (all pinned in
`pyproject.toml`); pytest is for the tests.

Step 2b runs Tesseract from an Apptainer image (not a Python package), built once by
`cluster/00_tesseract_image.sbatch` from `cluster/tesseract.def` into

    $PROJECT_ROOT/envs/lsab-tesseract/tesseract.sif

`scripts/ocr.py` needs `module load apptainer/1.4.3` (compute node); the virtual
environment above runs it.

Step 4 added (login node, through the proxy): bm25's optional stemmer, pure Python:

    PYTHONNOUSERSITE=1 $ENV/bin/pip install --no-cache-dir --no-compile "snowballstemmer==2.2.0"

The model scorers' `models` extra (torch, transformers, colpali-engine; `pyproject.toml`)
is not installed yet.

Step 5 adds no package to the virtual environment (scripts/attributes.py,
scripts/assign_discipline.py and the validation sheet use the standard library, numpy and
PyMuPDF). The layout detector runs from an Apptainer image, built once by
`cluster/00_layout_image.sbatch` from `cluster/layout.def` into

    $PROJECT_ROOT/envs/lsab/images/layout.sif

(python 3.12.15-slim-bookworm by digest; torch 2.13.0+cu126, torchvision 0.28.0+cu126,
transformers 5.17.0, pillow 12.3.0, numpy 2.2.6; its full `pip freeze` is in
/opt/pip-freeze.txt inside the image). One image file instead of torch in the venv: the
project space is close to its inode quota. The model weights live in

    $PROJECT_ROOT/models_lsab/docling-layout-heron

(downloaded by `python scripts/detect_layout.py fetch` at the pinned revision; the shared
hf_cache is not used).

Step 10 adds no package to the virtual environment. The model scorers (splade,
cross_encoder, image) run from an Apptainer image with the `models` extra, built once by
`cluster/10a_similarity_image.sbatch` from `cluster/similarity.def` into

    $PROJECT_ROOT/models_lsab/images/similarity.sif

with its full `pip freeze` in /opt/pip-freeze.txt inside the image. The same job fetches
their weights (pinned in `config/similarity_weights.toml`) into $PROJECT_ROOT/models_lsab/.

Step 16 added (login node, through the proxy): matplotlib, for Q4's heatmap
(`scripts/analysis_q4.py`); pip also brings its dependencies (contourpy, cycler,
fonttools, kiwisolver, pillow, pyparsing, python-dateutil, six):

    PYTHONNOUSERSITE=1 $ENV/bin/pip install --no-cache-dir --no-compile "matplotlib==3.11.2"

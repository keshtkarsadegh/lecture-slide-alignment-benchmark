#!/usr/bin/env python3
"""Step 1: download the four corpora into data/raw/<corpus>/.

Each corpus is fetched from its official public source at a pinned version, verified,
and described in data/raw/<corpus>/SOURCE.json (URLs, date, version, licence, and the
size and sha256 of every file kept). Only what transcript-to-slide alignment needs is
kept: the slides, the transcripts and the ground-truth alignment. Videos and audio are
skipped; SOURCE.json says what was skipped and why.

Downloads are kept as archives, one per corpus, rather than unpacked: the Lustre file
system has an inode budget, and later steps can read members straight from a tar or zip.
For the GitHub corpora the archive is a filtered copy of the repository tarball, holding
only the selected files, each checked against the git blob hash GitHub publishes.

The script is idempotent: a corpus whose SOURCE.json matches the files on disk (size and
sha256) is not downloaded again; its contents are re-verified and the run moves on.

If a source needs a manual step (login, licence click, request form, quota page) the
script stops for that corpus with a message saying what to do. It never works around
access controls.

Usage:
    python scripts/download.py --corpus all
    python scripts/download.py --corpus mavils [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "raw"
SOURCE_JSON = "SOURCE.json"
USER_AGENT = "lecture-slide-alignment-benchmark/0.1 (scripts/download.py)"
CHUNK = 1 << 20
RETRIES = 3
TIMEOUT = 120  # seconds per socket operation, not per download


class ManualStepRequired(Exception):
    """The source needs a person (login, licence click, form): stop for this corpus."""


class VerificationError(Exception):
    """A download does not match what the source publishes or what we expect."""


# --------------------------------------------------------------------------- helpers


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """The hash git (and GitHub's tree API) gives a file: sha1 over a 'blob <n>\\0' header
    plus the bytes. Checking it ties every kept file to the exact pinned commit."""
    h = hashlib.sha1(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    # Write-then-rename so an interrupted run never leaves a half-written SOURCE.json,
    # which the idempotency check would otherwise have to second-guess.
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def open_url(url: str, headers: dict | None = None):
    # urllib picks up https_proxy/http_proxy from the environment, which is how the
    # compute nodes reach the internet.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def access_error(url: str, err: urllib.error.HTTPError) -> Exception:
    """Map HTTP errors that mean 'a person must act' to ManualStepRequired."""
    if err.code in (401, 403):
        return ManualStepRequired(
            f"{url} answered HTTP {err.code}: the source needs a login or an access "
            "approval. Get access in a browser first; this script does not bypass it."
        )
    return err


def http_json(url: str, headers: dict | None = None) -> dict:
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with open_url(url, headers) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 403 and "api.github.com" in url:
                raise RuntimeError(
                    f"{url}: HTTP 403 from the GitHub API (probably the 60 requests/hour "
                    "anonymous limit on the shared proxy). Wait an hour or export "
                    "GITHUB_TOKEN and rerun."
                ) from e
            if e.code in (401, 403, 404):
                raise access_error(url, e) from e
            last = e
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            last = e
        log(f"  retry {attempt}/{RETRIES} for {url}: {last}")
        time.sleep(5 * attempt)
    raise RuntimeError(f"giving up on {url}: {last}")


def download_file(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[int, str]:
    """Stream url to dest (via dest.part), hashing on the way; return (size, sha256)."""
    tmp = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with open_url(url) as r, open(tmp, "wb") as f:
                h, n = hashlib.sha256(), 0
                for chunk in iter(lambda: r.read(CHUNK), b""):
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            size, digest = n, h.hexdigest()
            check_file(dest.name, size, digest, expected_size, expected_sha256)
            os.replace(tmp, dest)
            return size, digest
        except ManualStepRequired:
            tmp.unlink(missing_ok=True)
            raise
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            if e.code in (401, 403, 404):
                raise access_error(url, e) from e
            last = e
        except (urllib.error.URLError, OSError, http.client.HTTPException, VerificationError) as e:
            # A short read shows up as a size/sha mismatch, so mismatches are retried too.
            tmp.unlink(missing_ok=True)
            last = e
        log(f"  retry {attempt}/{RETRIES} for {url}: {last}")
        time.sleep(5 * attempt)
    if isinstance(last, VerificationError):
        raise last
    raise RuntimeError(f"giving up on {url}: {last}")


def check_file(
    name: str,
    size: int,
    sha256: str,
    expected_size: int | None,
    expected_sha256: str | None,
) -> None:
    if expected_size is not None and size != expected_size:
        raise VerificationError(f"{name}: size {size} != published {expected_size}")
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise VerificationError(f"{name}: sha256 {sha256} != published {expected_sha256}")


# ------------------------------------------------------------------ content checks


@dataclass(frozen=True)
class Role:
    """A kind of file the corpus must contain (slides, transcripts, alignment), found by
    a regex on archive member paths. `expected` is an exact count where the pinned
    version fixes it; otherwise at least `minimum` must be present."""

    name: str
    pattern: str
    expected: int | None = None
    minimum: int = 1


def list_archive_members(path: Path) -> list[str]:
    """Regular-file member paths of a .zip or .tar[.gz]. Reading the zip also checks every
    member's CRC, and reading a tar.gz to the end checks the gzip CRC, so a corrupt
    archive fails here even when no published checksum exists."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise VerificationError(f"{path.name}: CRC error in member {bad}")
            return [i.filename for i in zf.infolist() if not i.is_dir()]
    try:
        with tarfile.open(path, "r:*") as tf:
            return [m.name for m in tf if m.isfile()]
    except (tarfile.TarError, OSError, EOFError) as e:
        raise VerificationError(f"{path.name}: unreadable archive: {e}") from e


def check_roles(members: list[str], roles: tuple[Role, ...]) -> tuple[dict, list[str]]:
    counts: dict[str, int] = {}
    problems: list[str] = []
    for role in roles:
        rx = re.compile(role.pattern)
        n = sum(1 for m in members if rx.search(m))
        counts[role.name] = n
        if role.expected is not None and n != role.expected:
            problems.append(f"{role.name}: {n} files match {role.pattern!r}, expected {role.expected}")
        elif role.expected is None and n < role.minimum:
            problems.append(f"{role.name}: {n} files match {role.pattern!r}, expected >= {role.minimum}")
    return counts, problems


# ------------------------------------------------------------------- corpus types


@dataclass
class Corpus:
    """What every corpus shares; subclasses say how to fetch it."""

    name: str
    title: str
    homepage: str
    paper: str
    licence: str
    licence_url: str
    archive: str  # the kept file whose members carry slides/transcripts/alignment
    roles: tuple[Role, ...]
    kept: str
    skipped: tuple[str, ...] = ()
    notes: str = ""

    def fetch(self, dest: Path) -> dict:
        """Download into dest; return {'version': ..., 'urls': [...], 'files': [...]}"""
        raise NotImplementedError


@dataclass
class GitHubCorpus(Corpus):
    owner: str = ""
    repo: str = ""
    commit: str = ""
    include: tuple[str, ...] = ()  # path prefixes to keep
    exclude: tuple[str, ...] = (r"(^|/)\.DS_Store$",)  # regexes to drop

    def selects(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.include) and not any(
            re.search(x, path) for x in self.exclude
        )

    def tree(self) -> dict[str, tuple[str, int]]:
        """{path: (git blob sha1, size)} for every regular file at the pinned commit."""
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/git/trees/{self.commit}?recursive=1"
        headers = {"Accept": "application/vnd.github+json"}
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        data = http_json(url, headers)
        if data.get("truncated"):
            raise VerificationError(f"{url}: tree listing truncated by GitHub")
        # Mode 100644/100755 only: symlinks (120000) and submodules are not file content.
        return {
            e["path"]: (e["sha"], e["size"])
            for e in data["tree"]
            if e["type"] == "blob" and e["mode"] in ("100644", "100755")
        }

    def fetch(self, dest: Path) -> dict:
        api = f"https://api.github.com/repos/{self.owner}/{self.repo}/git/trees/{self.commit}?recursive=1"
        tarball = f"https://codeload.github.com/{self.owner}/{self.repo}/tar.gz/{self.commit}"
        wanted = {p: v for p, v in self.tree().items() if self.selects(p)}
        if not wanted:
            raise VerificationError(f"nothing in {self.owner}/{self.repo}@{self.commit} matches {self.include}")
        log(f"  {len(wanted)} files selected from the tree; streaming {tarball}")
        out = dest / self.archive
        last: Exception | None = None
        for attempt in range(1, RETRIES + 1):
            try:
                with open_url(tarball) as r:
                    repack_github_tarball(r, wanted, out, self.commit)
                break
            except (urllib.error.URLError, OSError, http.client.HTTPException, tarfile.TarError, EOFError) as e:
                last = e
                log(f"  retry {attempt}/{RETRIES} for {tarball}: {e}")
                time.sleep(5 * attempt)
        else:
            raise RuntimeError(f"giving up on {tarball}: {last}")
        return {
            "version": {"git_commit": self.commit, "repository": f"https://github.com/{self.owner}/{self.repo}"},
            "urls": [tarball, api],
            "files": [file_record(out, tarball, None)],
            "selection": {
                "include_prefixes": list(self.include),
                "exclude_regexes": list(self.exclude),
                "files_selected": len(wanted),
                "per_file_check": "git blob sha1 against the GitHub tree API at the pinned commit",
            },
        }


def repack_github_tarball(
    stream: BinaryIO,
    wanted: dict[str, tuple[str, int]],
    out: Path,
    commit: str | None = None,
) -> None:
    """Copy the wanted members of a GitHub tar.gz stream into an uncompressed tar at out.

    The top-level '<repo>-<sha>/' directory is stripped so member paths equal repository
    paths. Every member is checked against its git blob sha1 and size; a mismatch, a
    missing file, or a tarball for another commit aborts without leaving `out` behind.
    """
    tmp = out.with_name(out.name + ".part")
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as src, tarfile.open(
            tmp, "w", format=tarfile.PAX_FORMAT
        ) as dst:
            for m in src:
                if not m.isfile():
                    continue
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                if rel not in wanted:
                    continue
                data = src.extractfile(m).read()
                sha1, size = wanted[rel]
                if len(data) != size or git_blob_sha1(data) != sha1:
                    raise VerificationError(f"{rel}: content does not match git blob {sha1}")
                # Fixed owner/mode and the commit's mtime make the archive reproducible.
                ti = tarfile.TarInfo(rel)
                ti.size, ti.mtime, ti.mode = len(data), m.mtime, 0o644
                dst.addfile(ti, io.BytesIO(data))
                seen.add(rel)
            # GitHub writes the commit id into the tarball's global pax header.
            got = src.pax_headers.get("comment")
        missing = sorted(set(wanted) - seen)
        if missing:
            raise VerificationError(f"{len(missing)} selected files missing from tarball, e.g. {missing[:3]}")
        if commit is not None and got is not None and got != commit:
            raise VerificationError(f"tarball is for commit {got}, expected {commit}")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class RemoteFile:
    url: str
    name: str
    sha256: str | None = None  # as published by the source
    size: int | None = None  # as published by the source, or as first seen (soft pin)


@dataclass
class HTTPCorpus(Corpus):
    files: tuple[RemoteFile, ...] = ()
    version: dict = field(default_factory=dict)

    def fetch(self, dest: Path) -> dict:
        records = []
        for rf in self.files:
            path = dest / rf.name
            if path.exists() and rf.sha256 is not None and sha256_file(path) == rf.sha256:
                # A file with the published hash is already complete; a previous run
                # stopped later (e.g. at the content check), so do not fetch 2 GB again.
                log(f"  {rf.name} already present with the published sha256")
            else:
                log(f"  downloading {rf.url}")
                download_file(rf.url, path, expected_size=rf.size, expected_sha256=rf.sha256)
            records.append(file_record(path, rf.url, rf.sha256))
        return {"version": self.version, "urls": [rf.url for rf in self.files], "files": records}


def file_record(path: Path, url: str, published_sha256: str | None) -> dict:
    return {
        "path": path.name,
        "url": url,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "published_sha256": published_sha256,
    }


# ------------------------------------------------------------------- the corpora

NOT_VIDEO = "videos are not needed: slides, transcripts and alignment are available without them"

CORPORA: dict[str, Corpus] = {
    c.name: c
    for c in (
        GitHubCorpus(
            name="mavils",
            title="MaViLS: Matching Videos to Lecture Slides",
            homepage="https://github.com/andererka/MaViLS",
            paper="Anderer, Reich, Woelfel. Interspeech 2024. doi:10.21437/Interspeech.2024-978",
            licence="Apache-2.0 (the repository's licence; no separate data licence is stated)",
            licence_url="https://github.com/andererka/MaViLS/blob/main/LICENSE",
            owner="andererka",
            repo="MaViLS",
            commit="b3c55a869a24a8f3f3881aa3e89fea9b1c81c0ad",
            include=("data/lectures/", "data/audioscripts/", "data/ground_truth_files/"),
            archive="mavils-b3c55a869a24.tar",
            roles=(
                Role("slides_pdf", r"^data/lectures/[^/]+\.pdf$", 21),
                Role("transcripts_srt", r"^data/audioscripts/[^/]+\.srt$", 20),
                Role("alignment_xlsx", r"^data/ground_truth_files/[^/]+\.xlsx$", 24),
            ),
            kept="data/lectures (slide PDFs), data/audioscripts (faster-whisper SRT), "
            "data/ground_truth_files (human sentence-to-slide labels, incl. 4 control re-labelings)",
            skipped=(
                "lecture videos (hosted on Kaggle, need a Kaggle login); " + NOT_VIDEO,
                "code, results/, jupyter_notebooks/, data/unlabeled_ground_truth/ (not alignment data)",
            ),
        ),
        HTTPCorpus(
            name="m3av",
            title="M3AV: A Multimodal, Multigenre, and Multipurpose Audio-Visual Academic Lecture Dataset",
            homepage="https://github.com/Jack-ZC8/M3AV-dataset",
            paper="Chen et al. ACL 2024. arXiv:2403.14168",
            licence="CC BY-NC-SA 4.0 (video copyright stays with the original owners)",
            licence_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
            files=(
                RemoteFile(
                    url="https://huggingface.co/datasets/CHHHH/M3AV_v1.0/resolve/"
                    "fd0ef99ec1fcf5a4ab946317e32923209b802389/dataset_v1.0_noaudio.tar.gz",
                    name="dataset_v1.0_noaudio.tar.gz",
                    # Published in download/M3AV_v1.0_sha256.txt of the GitHub repo and as
                    # the LFS oid on HuggingFace; both agree.
                    sha256="59fc0eb4ab1d978be338dcbb961be9d27dfd3139db255bfc4648c694e810fca0",
                    size=2024906617,
                ),
            ),
            version={
                "dataset": "M3AV v1.0",
                "huggingface_repo": "datasets/CHHHH/M3AV_v1.0",
                "huggingface_revision": "fd0ef99ec1fcf5a4ab946317e32923209b802389",
                "published_checksums": "https://github.com/Jack-ZC8/M3AV-dataset/blob/"
                "1aa6fa8992fe0bdd28583d3ad997a28f1cc18c03/download/M3AV_v1.0_sha256.txt",
            },
            archive="dataset_v1.0_noaudio.tar.gz",
            roles=(
                Role("slide_segment_jpg", r"^\./dataset_v1\.0_noaudio/[^/]+/ocr/seg_imgs/[^/]+\.jpg$", 24246),
                Role("slide_ocr_json", r"^\./dataset_v1\.0_noaudio/[^/]+/ocr/ocr_result\.json$", 1113),
                Role("transcript_json", r"^\./dataset_v1\.0_noaudio/[^/]+/speech/final\+timestamps\.json$", 1113),
                Role("alignment_json", r"^\./dataset_v1\.0_noaudio/[^/]+/ocr/seg_speech_ocr\.json$", 1113),
            ),
            kept="dataset_v1.0_noaudio.tar.gz: everything in M3AV v1.0 except audio (1113 talks: "
            "slide-segment images and OCR, timestamped transcripts, speech-to-segment files, "
            "paper sentences for CS talks, YouTube URLs)",
            skipped=(
                "audio (dataset_v1.0_onlyaudio.tar.gz.00-.08, about 43.5 GB): not needed for "
                "slides, transcripts or alignment",
                "videos: not distributed (only YouTube URLs, raw/*.ytbUrl)",
            ),
            notes="Slides are not PDFs: they are images of the slide segments cut from the "
            "video (ocr/seg_imgs), with OCR (ocr/ocr_result.json); ocr/seg_speech_ocr.json "
            "ties speech to the segments.",
        ),
        GitHubCorpus(
            name="mcd",
            title="MCD: Multimodal Conference Dataset",
            homepage="https://github.com/meghamariamkm2002/MCD",
            paper="Megha Mariam K M, Balasubramanian, Jawahar. CVPR Findings 2026. arXiv:2605.05831",
            licence="none stated (no LICENSE file, no licence in README or project page)",
            licence_url="",
            owner="meghamariamkm2002",
            repo="MCD",
            commit="91052bc619bfe3dd621f48c2fd55758cd343133f",
            include=("Dataset/PRES/",),
            archive="mcd-91052bc619bf.tar",
            roles=(
                Role("slides_png", r"^Dataset/PRES/test/input/vid-seg/slides/sp_\d+/sp_\d+_\d+\.png$", 204),
                Role("transcripts_txt", r"^Dataset/PRES/test/input/vid-seg/trans/sp_\d+/sp_\d+_\d+\.txt$", 204),
                Role("gt_slide_json", r"^Dataset/PRES/test/GT-slide/sp_\d+\.json$", 20),
            ),
            kept="Dataset/PRES (presentation set: per-talk slide images and transcript segments "
            "paired by segment id, ground truth, paper segments; test and few-shot 'eg' splits)",
            skipped=(
                "Dataset/EXP (explanatory videos to paper; no slides)",
                "Prompts/, static/ (not data)",
                "videos: not in the repository; " + NOT_VIDEO,
            ),
            notes="The transcript-to-slide alignment is the pairing of trans/sp_X/sp_X_k.txt "
            "with slides/sp_X/sp_X_k.png; GT-slide/GT-pres map slides to paper segments.",
        ),
        GitHubCorpus(
            name="slide_highlight",
            title="Slide-Highlight: Attend to What I Say: Highlighting Relevant Content on Slides",
            homepage="https://github.com/meghamariamkm2002/Slide_Highlight",
            paper="none found; the repository README is the only description",
            licence="none stated (no LICENSE file, no licence in README)",
            licence_url="",
            owner="meghamariamkm2002",
            repo="Slide_Highlight",
            commit="e8872ca50a695cdf846554ad9adbab82af9b3b9b",
            include=("Dataset/slides/", "Dataset/transcripts/", "Dataset/GT/", "Results/Corr_trans/"),
            archive="slide_highlight-e8872ca50a69.tar",
            roles=(
                Role("slides_png", r"^Dataset/slides/[^/]+_slides/vid_\d+/PART_\d+_[^/]+\.png$", 150),
                Role("transcripts_srt", r"^Dataset/transcripts/[^/]+_trans/vid_\d+/PART_\d+_[^/]+\.srt$", 150),
                Role("gt_highlight_json", r"^Dataset/GT/[^/]+_Slides_GT/vid_\d+/PART_\d+_[^/]+\.json$", 161),
            ),
            kept="Dataset/slides (slide PNGs), Dataset/transcripts (per-slide transcript "
            "segments PART_<k>_<slide id> in json/srt/tsv/txt/vtt), Dataset/GT (highlight "
            "annotations), Results/Corr_trans (corrected transcripts)",
            skipped=(
                "Dataset/videos (14 MP4, about 94 MB); " + NOT_VIDEO,
                "Dataset/Layout (layout-detection images and JSON, about 355 MB; derived, not alignment)",
                "Results/AFTER_Corr, Results/BEFORE_Corr, methods.ipynb (method outputs and code)",
            ),
            notes="The transcript-to-slide alignment is the segment naming: transcript "
            "PART_<k>_<id>.* belongs to slide image PART_<k>_<id>.png. NeurIPS vid_0054 also "
            "has 11 slide images in a subfolder 38992648/ named <id>.png; GT has 161 files "
            "against 150 transcript segments.",
        ),
    )
}


# ------------------------------------------------------------------ orchestration


def verify_contents(corpus: Corpus, dest: Path) -> dict:
    members = list_archive_members(dest / corpus.archive)
    counts, problems = check_roles(members, corpus.roles)
    if not corpus.roles:
        # A corpus with no content checks would pass vacuously.
        problems.append("no slide/transcript/alignment roles defined for this corpus")
    return {
        "passed": not problems,
        "archive": corpus.archive,
        "archive_members": len(members),
        "role_counts": counts,
        "problems": problems,
    }


def is_complete(dest: Path) -> bool:
    """True when SOURCE.json exists, records a passed verification, and every file it
    lists is on disk with the recorded size and sha256."""
    src = dest / SOURCE_JSON
    try:
        record = json.loads(src.read_text())
    except (OSError, ValueError):
        return False
    files = record.get("files") or []
    if not files or not record.get("verification", {}).get("passed"):
        return False
    for f in files:
        p = dest / f["path"]
        if not p.is_file() or p.stat().st_size != f["size"] or sha256_file(p) != f["sha256"]:
            return False
    return True


def run_corpus(corpus: Corpus, root: Path, force: bool = False) -> str:
    """Download (unless complete), verify, write SOURCE.json. Returns 'downloaded' or
    'skipped'; raises on failure, leaving no SOURCE.json for a failed download."""
    dest = root / corpus.name
    dest.mkdir(parents=True, exist_ok=True)
    src = dest / SOURCE_JSON

    if not force and is_complete(dest):
        record = json.loads(src.read_text())
        verification = verify_contents(corpus, dest)
        if not verification["passed"]:
            raise VerificationError("; ".join(verification["problems"]))
        if record.get("verification") != verification:
            # The files are unchanged but the checks were refined; keep the record current.
            record["verification"] = verification
            record["verified_utc"] = now_utc()
            write_json_atomic(src, record)
        return "skipped"

    src.unlink(missing_ok=True)
    retrieved = now_utc()
    fetched = corpus.fetch(dest)
    verification = verify_contents(corpus, dest)
    if not verification["passed"]:
        raise VerificationError("; ".join(verification["problems"]))
    record = {
        "corpus": corpus.name,
        "title": corpus.title,
        "homepage": corpus.homepage,
        "paper": corpus.paper,
        "licence": corpus.licence,
        "licence_url": corpus.licence_url,
        "retrieved_utc": retrieved,
        "verified_utc": now_utc(),
        "version": fetched["version"],
        "urls": fetched["urls"],
        "files": fetched["files"],
        **({"selection": fetched["selection"]} if "selection" in fetched else {}),
        "kept": corpus.kept,
        "skipped": list(corpus.skipped),
        "notes": corpus.notes,
        "verification": verification,
        "script": "scripts/download.py",
    }
    write_json_atomic(src, record)
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default="all", help=f"one of {', '.join(CORPORA)}, or all")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="target directory (default: data/raw)")
    ap.add_argument("--force", action="store_true", help="download again even if complete")
    args = ap.parse_args(argv)

    key = args.corpus.lower().replace("-", "_")
    if key != "all" and key not in CORPORA:
        ap.error(f"unknown corpus {args.corpus!r}; choose from {', '.join(CORPORA)}, all")
    names = list(CORPORA) if key == "all" else [key]

    status: dict[str, str] = {}
    for name in names:
        log(f"== {name}")
        try:
            status[name] = run_corpus(CORPORA[name], args.root, args.force)
        except ManualStepRequired as e:
            status[name] = "MANUAL STEP NEEDED"
            log(f"  STOP: {e}")
        except Exception as e:  # report and carry on with the other corpora
            status[name] = "FAILED"
            log(f"  FAILED: {type(e).__name__}: {e}")
        else:
            v = json.loads((args.root / name / SOURCE_JSON).read_text())["verification"]
            log(f"  {status[name]}; verified: {v['archive_members']} members, {v['role_counts']}")

    log("== summary")
    for name, s in status.items():
        log(f"  {name:16s} {s}")
    if any(s == "FAILED" for s in status.values()):
        return 1
    if any(s == "MANUAL STEP NEEDED" for s in status.values()):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Step 5: build the fixed English word list that the noise attribute is measured against.

The list is SCOWL (Spell Checker Oriented Word Lists, Kevin Atkinson), release
2020.12.07, http://wordlist.aspell.net/, taken from the release tarball (pinned by
sha256). SCOWL is the source of the word lists most spell checkers ship (aspell,
hunspell's en_US, Debian's wamerican), so "in the list" means "a spell checker would
accept it". Its licence (the file Copyright in the tarball) allows copying and
redistribution with the notice, which the output's header carries.

Which parts of SCOWL:
  * sizes 10 to 70. 70 is SCOWL's "large" size: below it, common technical words of the
    lectures (dataset, pytorch, github) are missing and would count as noise; above it
    (80, 95) come rare and archaic forms that make OCR fragments look like words;
  * the spellings english (common to all), american and british;
  * the parts words, upper (capitalised words), proper-names and abbreviations. Names and
    abbreviations are in because slides are full of them (author names, "kg", "fig"):
    noise is meant to catch text that is not a word at all, mostly OCR damage.
    Contractions and the variant spellings are out.
Every entry is lowercased; only entries made of the letters a-z alone are kept (the
apostrophe forms and the few accented Latin-1 entries cannot match a token, which is
ASCII letters only in scripts/attributes.py).

Output: config/wordlist_en_scowl70.txt.gz, one word per line after '#' header lines,
sorted, gzip with a fixed timestamp (identical input, identical file). It is in git: the
noise numbers depend on it, so it is a definition, not data.

Usage (login node: a small download through the proxy):
    source cluster/env.local && export https_proxy=$CLUSTER_PROXY
    python scripts/build_wordlist.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "config" / "wordlist_en_scowl70.txt.gz"

URL = ("https://downloads.sourceforge.net/project/wordlist/SCOWL/2020.12.07/"
       "scowl-2020.12.07.tar.gz")
SHA256 = "5587667caa20c4891390c2d42dbb4d5c4c3f41bee77af1457ece3ba23fb859cc"
SIZES = (10, 20, 35, 40, 50, 55, 60, 70)
SPELLINGS = ("english", "american", "british")
PARTS = ("words", "upper", "proper-names", "abbreviations")
MEMBER = re.compile(r"^scowl-2020\.12\.07/final/(%s)-(%s)\.(\d+)$"
                    % ("|".join(SPELLINGS), "|".join(PARTS)))
WORD = re.compile(r"^[a-z]+$")


def wanted(member: str) -> bool:
    m = MEMBER.match(member)
    return bool(m) and int(m.group(3)) in SIZES


def build(tar_bytes: bytes) -> tuple[list[str], list[str]]:
    """The sorted word set and the members it was read from."""
    words: set[str] = set()
    used: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not (m.isfile() and wanted(m.name)):
                continue
            used.append(m.name.rsplit("/", 1)[1])
            # SCOWL's files are Latin-1.
            for line in tf.extractfile(m).read().decode("latin-1").splitlines():
                w = line.strip().lower()
                if WORD.match(w):
                    words.add(w)
    return sorted(words), sorted(used)


def write(words: list[str], used: list[str], path: Path) -> None:
    header = [
        "# English word list for the noise attribute (scripts/attributes.py).",
        "# Built by scripts/build_wordlist.py from SCOWL 2020.12.07",
        f"#   {URL}",
        f"#   sha256 {SHA256}",
        f"# sizes {','.join(map(str, SIZES))}; spellings {', '.join(SPELLINGS)}; "
        f"parts {', '.join(PARTS)}; lowercased, a-z only; {len(words)} words.",
        f"# members: {' '.join(used)}",
        "# Copyright 2000-2018 by Kevin Atkinson (and the copyrights listed in SCOWL's",
        "# Copyright file). Permission to use, copy, modify, distribute and sell these word",
        "# lists, the associated scripts, the output created from the scripts, and its",
        "# documentation for any purpose is hereby granted without fee, provided that the",
        "# above copyright notice appears in all copies and that both that copyright",
        "# notice and this permission notice appear in supporting documentation. Kevin",
        "# Atkinson makes no representations about the suitability of this array for any",
        "# purpose. It is provided \"as is\" without express or implied warranty.",
    ]
    data = ("\n".join(header + words) + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0,
                                               filename="") as gz:
        gz.write(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tarball", type=Path, help="a local copy instead of downloading")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.tarball:
        raw = args.tarball.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=120) as resp:
            raw = resp.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        print(f"sha256 mismatch: {digest} != {SHA256}", file=sys.stderr)
        return 1
    words, used = build(raw)
    write(words, used, args.out)
    print(f"{len(words)} words from {len(used)} members -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

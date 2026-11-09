"""Tests for scripts/download.py: verification and idempotency only.

Everything runs on tiny synthetic files; nothing touches the network. Network-facing
code is exercised by feeding it local streams or by replacing Corpus.fetch.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

import download as d


# ---------------------------------------------------------------- small builders


def make_tar_gz(files: dict[str, bytes], top: str = "Repo-abc123", comment: str | None = "abc123") -> bytes:
    """A GitHub-style tarball: one top-level directory, commit id in the pax header."""
    buf = io.BytesIO()
    pax = {"comment": comment} if comment else {}
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers=pax) as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size, ti.mtime = len(data), 1_700_000_000
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def blob_map(files: dict[str, bytes]) -> dict[str, tuple[str, int]]:
    return {p: (d.git_blob_sha1(b), len(b)) for p, b in files.items()}


@dataclass
class FakeCorpus(d.Corpus):
    """Fetch writes a small zip and counts how often it was called."""

    calls: int = 0

    def fetch(self, dest: Path) -> dict:
        self.calls += 1
        path = dest / self.archive
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("talk1/slides.pdf", b"%PDF-1.4 fake")
            zf.writestr("talk1/transcript.srt", b"1\n00:00:00,000 --> 00:00:01,000\nhello\n")
            zf.writestr("talk1/alignment.csv", b"sentence,slide\n0,1\n")
        return {"version": {"v": "1"}, "urls": ["file://fake"], "files": [d.file_record(path, "file://fake", None)]}


def fake_corpus(roles=None) -> FakeCorpus:
    return FakeCorpus(
        name="fake",
        title="Fake",
        homepage="",
        paper="",
        licence="CC0",
        licence_url="",
        archive="fake.zip",
        roles=roles
        if roles is not None
        else (
            d.Role("slides", r"\.pdf$", 1),
            d.Role("transcripts", r"\.srt$", 1),
            d.Role("alignment", r"alignment\.csv$", 1),
        ),
        kept="all",
    )


# ------------------------------------------------------------------- hashing


def test_git_blob_sha1_matches_git():
    # Known values from `git hash-object`: the empty blob and "hello\n".
    assert d.git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert d.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_check_file_size_and_sha():
    data = b"abc"
    sha = hashlib.sha256(data).hexdigest()
    d.check_file("x", 3, sha, 3, sha)  # passes
    d.check_file("x", 3, sha, None, None)  # nothing published: nothing to compare
    with pytest.raises(d.VerificationError, match="size"):
        d.check_file("x", 3, sha, 4, None)
    with pytest.raises(d.VerificationError, match="sha256"):
        d.check_file("x", 3, sha, None, "0" * 64)


# --------------------------------------------------------------- content roles


def test_check_roles_exact_and_minimum():
    members = ["a/s1.pdf", "a/s2.pdf", "a/t.srt"]
    roles = (d.Role("slides", r"\.pdf$", 2), d.Role("transcripts", r"\.srt$"), d.Role("gt", r"\.xlsx$"))
    counts, problems = d.check_roles(members, roles)
    assert counts == {"slides": 2, "transcripts": 1, "gt": 0}
    assert len(problems) == 1 and problems[0].startswith("gt:")

    _, problems = d.check_roles(members, (d.Role("slides", r"\.pdf$", 3),))
    assert problems and "expected 3" in problems[0]


def test_list_archive_members_zip_and_tar(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("dir/", b"")
        zf.writestr("dir/x.txt", b"x")
    assert d.list_archive_members(z) == ["dir/x.txt"]

    t = tmp_path / "a.tar"
    t.write_bytes(make_tar_gz({"y.txt": b"y"}))  # gzip; tarfile 'r:*' detects it
    assert d.list_archive_members(t) == ["Repo-abc123/y.txt"]


def test_list_archive_members_rejects_corrupt_zip(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x.txt", b"original content")
    raw = bytearray(z.read_bytes())
    i = raw.index(b"original content")
    raw[i] ^= 0xFF  # flip a data byte: CRC no longer matches
    z.write_bytes(bytes(raw))
    with pytest.raises(d.VerificationError, match="CRC"):
        d.list_archive_members(z)


def test_list_archive_members_rejects_garbage(tmp_path):
    p = tmp_path / "x.tar.gz"
    p.write_bytes(b"not an archive at all")
    with pytest.raises(d.VerificationError):
        d.list_archive_members(p)


# --------------------------------------------------------- GitHub repacking


def test_github_selection():
    c = d.GitHubCorpus(
        name="g", title="", homepage="", paper="", licence="", licence_url="", archive="g.tar",
        roles=(), kept="", include=("data/a/", "data/b/"),
    )
    assert c.selects("data/a/x.pdf")
    assert not c.selects("data/a/.DS_Store")
    assert not c.selects("data/c/x.pdf")
    assert not c.selects("code/run.py")


def test_repack_keeps_only_wanted_and_verifies(tmp_path):
    files = {"data/a.pdf": b"pdf", "data/b.srt": b"srt", "video/v.mp4": b"big video"}
    wanted = blob_map({k: v for k, v in files.items() if k.startswith("data/")})
    out = tmp_path / "out.tar"
    d.repack_github_tarball(io.BytesIO(make_tar_gz(files)), wanted, out, "abc123")
    with tarfile.open(out) as tf:
        names = sorted(m.name for m in tf)
        assert names == ["data/a.pdf", "data/b.srt"]  # top dir stripped, video dropped
        assert tf.extractfile("data/a.pdf").read() == b"pdf"
    assert not (tmp_path / "out.tar.part").exists()


def test_repack_is_deterministic(tmp_path):
    files = {"data/a.pdf": b"pdf", "data/b.srt": b"srt"}
    a, b = tmp_path / "a.tar", tmp_path / "b.tar"
    d.repack_github_tarball(io.BytesIO(make_tar_gz(files)), blob_map(files), a)
    d.repack_github_tarball(io.BytesIO(make_tar_gz(files)), blob_map(files), b)
    assert d.sha256_file(a) == d.sha256_file(b)


def test_repack_rejects_content_mismatch(tmp_path):
    files = {"data/a.pdf": b"pdf"}
    wanted = blob_map({"data/a.pdf": b"PDF"})  # the tree says different bytes
    out = tmp_path / "out.tar"
    with pytest.raises(d.VerificationError, match="git blob"):
        d.repack_github_tarball(io.BytesIO(make_tar_gz(files)), wanted, out)
    assert not out.exists() and not (tmp_path / "out.tar.part").exists()


def test_repack_rejects_missing_file(tmp_path):
    files = {"data/a.pdf": b"pdf"}
    wanted = blob_map({"data/a.pdf": b"pdf", "data/b.srt": b"srt"})
    with pytest.raises(d.VerificationError, match="missing"):
        d.repack_github_tarball(io.BytesIO(make_tar_gz(files)), wanted, tmp_path / "out.tar")


def test_repack_rejects_wrong_commit(tmp_path):
    files = {"data/a.pdf": b"pdf"}
    with pytest.raises(d.VerificationError, match="commit"):
        d.repack_github_tarball(
            io.BytesIO(make_tar_gz(files, comment="other")), blob_map(files), tmp_path / "out.tar", "abc123"
        )


# ------------------------------------------------------------- idempotency


def test_run_corpus_downloads_then_skips(tmp_path):
    c = fake_corpus()
    assert d.run_corpus(c, tmp_path) == "downloaded"
    record = json.loads((tmp_path / "fake" / "SOURCE.json").read_text())
    assert record["verification"]["passed"]
    assert record["verification"]["role_counts"] == {"slides": 1, "transcripts": 1, "alignment": 1}
    assert record["files"][0]["sha256"] == d.sha256_file(tmp_path / "fake" / "fake.zip")
    assert record["licence"] == "CC0" and record["retrieved_utc"]

    assert d.run_corpus(c, tmp_path) == "skipped"
    assert c.calls == 1  # the second run did not fetch
    assert d.run_corpus(c, tmp_path, force=True) == "downloaded"
    assert c.calls == 2


def test_changed_file_triggers_redownload(tmp_path):
    c = fake_corpus()
    d.run_corpus(c, tmp_path)
    archive = tmp_path / "fake" / "fake.zip"
    archive.write_bytes(archive.read_bytes() + b"\0")  # size and hash now differ
    assert not d.is_complete(tmp_path / "fake")
    assert d.run_corpus(c, tmp_path) == "downloaded"
    assert c.calls == 2


def test_missing_file_or_source_json_is_incomplete(tmp_path):
    c = fake_corpus()
    d.run_corpus(c, tmp_path)
    dest = tmp_path / "fake"
    assert d.is_complete(dest)
    (dest / "fake.zip").unlink()
    assert not d.is_complete(dest)
    assert not d.is_complete(tmp_path / "never-downloaded")


def test_failed_verification_writes_no_source_json(tmp_path):
    c = fake_corpus(roles=(d.Role("alignment", r"\.xlsx$", 1),))
    with pytest.raises(d.VerificationError):
        d.run_corpus(c, tmp_path)
    assert not (tmp_path / "fake" / "SOURCE.json").exists()


def test_corpus_without_roles_fails(tmp_path):
    with pytest.raises(d.VerificationError, match="no slide"):
        d.run_corpus(fake_corpus(roles=()), tmp_path)


def test_refined_checks_update_record_without_refetch(tmp_path):
    d.run_corpus(fake_corpus(roles=(d.Role("slides", r"\.pdf$", 1),)), tmp_path)
    c = fake_corpus()  # more roles than at download time
    assert d.run_corpus(c, tmp_path) == "skipped"
    assert c.calls == 0
    record = json.loads((tmp_path / "fake" / "SOURCE.json").read_text())
    assert set(record["verification"]["role_counts"]) == {"slides", "transcripts", "alignment"}


def test_http_corpus_reuses_file_with_published_sha(tmp_path, monkeypatch):
    payload = b"archive bytes"
    dest = tmp_path / "h"
    dest.mkdir()
    (dest / "a.bin").write_bytes(payload)
    c = d.HTTPCorpus(
        name="h", title="", homepage="", paper="", licence="", licence_url="", archive="a.bin", roles=(), kept="",
        files=(d.RemoteFile("https://example.invalid/a.bin", "a.bin", hashlib.sha256(payload).hexdigest()),),
    )
    monkeypatch.setattr(d, "download_file", lambda *a, **k: pytest.fail("must not download"))
    result = c.fetch(dest)
    assert result["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_cli_rejects_unknown_corpus():
    with pytest.raises(SystemExit):
        d.main(["--corpus", "nope"])

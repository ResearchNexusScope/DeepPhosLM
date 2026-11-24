"""Idempotent, hash-verified download of the MMseqs2 win64 binary.

Usage (from anywhere)::

    python tools/fetch_mmseqs.py
    python tools/fetch_mmseqs.py --force
    python tools/fetch_mmseqs.py --dest /path/to/tools/bin/mmseqs

The published release archive is downloaded to a temporary ``.part`` file, its
SHA-256 is checked against the digest published with the release, and only a
verified archive is extracted into ``tools/bin/mmseqs/``.  A marker file records
the verified digest so subsequent invocations are no-ops.  No network traffic
happens at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import requests

MMSEQS_URL = (
    "https://github.com/soedinglab/MMseqs2/releases/download/"
    "20-ae60e/mmseqs-win64.zip"
)
PUBLISHED_SHA256 = "56d79e866cbd0080f17fb51693c750f210ee315f96774b6c1b2eb89924b52540"
MARKER_FILENAME = ".mmseqs_sha256"
BINARY_NAMES = frozenset({"mmseqs", "mmseqs.exe"})

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "tools" / "bin" / "mmseqs"

_DOWNLOAD_CHUNK = 1 << 20


def find_repo_root() -> Path:
    """Return the repository root (parent of ``tools/``)."""
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_binaries(dest: Path) -> List[Path]:
    """Return all installed ``mmseqs`` / ``mmseqs.exe`` paths under ``dest``."""
    if not dest.is_dir():
        return []
    return sorted(
        path for path in dest.rglob("*") if path.is_file() and path.name in BINARY_NAMES
    )


def is_installed(dest: Path, expected_sha256: str) -> bool:
    """True only when a binary exists and the marker pins the expected hash."""
    marker = Path(dest) / MARKER_FILENAME
    if not marker.is_file() or not find_binaries(Path(dest)):
        return False
    try:
        pinned = marker.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return False
    return pinned == expected_sha256.strip().lower()


def verify_sha256(path: Path, expected_sha256: str) -> str:
    """Verify a file against the published digest; fail loudly on mismatch."""
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.strip().lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}. "
            "Refusing to extract an unverified archive."
        )
    return actual


def download_archive(url: str, target: Path, timeout: float) -> Path:
    """Stream ``url`` into ``target`` (never following a silent status error)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(target, "wb") as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                if chunk:
                    handle.write(chunk)
    if target.stat().st_size == 0:
        raise RuntimeError(f"Downloaded archive is empty: {url}")
    return target


def _safe_members(archive: zipfile.ZipFile) -> Iterator[Tuple[zipfile.ZipInfo, Path]]:
    """Yield archive members as safe relative paths, rejecting path traversal."""
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
            raise RuntimeError(f"Refusing absolute path in ZIP: {info.filename!r}")
        parts = [part for part in name.split("/") if part not in ("", ".")]
        if not parts:
            continue
        if ".." in parts:
            raise RuntimeError(f"Refusing path traversal in ZIP: {info.filename!r}")
        yield info, Path(*parts)


def extract_archive(archive_path: Path, dest: Path) -> List[Path]:
    """Extract a verified archive into ``dest``, stripping a sole root folder.

    Returns the list of extracted files.  Every member is checked for unsafe
    paths before anything is written.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = list(_safe_members(archive))
        if not members:
            raise RuntimeError(f"Archive contains no files: {archive_path}")
        top_levels = {relative.parts[0] for _, relative in members}
        strip_root = len(top_levels) == 1 and all(
            len(relative.parts) > 1 for _, relative in members
        )
        extracted: List[Path] = []
        for info, relative in members:
            target = dest.joinpath(*relative.parts[1:] if strip_root else relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)
            if os.name != "nt":
                target.chmod(target.stat().st_mode | 0o755)
            extracted.append(target)
    return extracted


def install(
    *,
    dest: Path = DEFAULT_DEST,
    url: str = MMSEQS_URL,
    expected_sha256: str = PUBLISHED_SHA256,
    timeout: float = 120.0,
    force: bool = False,
) -> Tuple[str, List[Path], bool]:
    """Download, verify and extract MMseqs2; return ``(digest, binaries, skipped)``."""
    dest = Path(dest)
    if is_installed(dest, expected_sha256) and not force:
        return expected_sha256.lower(), find_binaries(dest), True

    temp_archive = dest.parent / f".{dest.name}-{hashlib.sha1(url.encode()).hexdigest()[:8]}.zip.part"
    try:
        download_archive(url, temp_archive, timeout)
        digest = verify_sha256(temp_archive, expected_sha256)
        extract_archive(temp_archive, dest)
    finally:
        if temp_archive.exists():
            temp_archive.unlink()

    binaries = find_binaries(dest)
    if not binaries:
        raise RuntimeError(f"No mmseqs binary found after extraction into {dest}")
    (dest / MARKER_FILENAME).write_text(digest + "\n", encoding="utf-8")
    return digest, binaries, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and SHA-256-verify the MMseqs2 win64 binary."
    )
    parser.add_argument("--url", default=MMSEQS_URL, help="Release ZIP URL.")
    parser.add_argument(
        "--sha256", default=PUBLISHED_SHA256, help="Expected SHA-256 of the ZIP."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Extraction directory (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="HTTP timeout in seconds."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if already verified."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    digest, binaries, skipped = install(
        dest=args.dest,
        url=args.url,
        expected_sha256=args.sha256,
        timeout=args.timeout,
        force=args.force,
    )
    if skipped:
        print(f"MMseqs2 already verified; SHA-256 {digest}")
    else:
        print(f"Verified MMseqs2 archive SHA-256 {digest}")
    for binary in binaries:
        print(f"  binary: {binary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

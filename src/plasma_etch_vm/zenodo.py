"""Download files of Zenodo record 17122442 with size / MD5 verification and a disk guard.

Usage (CLI): ``pevm download --config data.yaml [--days all]``.
Only files listed in the config (``zenodo.small_files`` + ``zenodo.day_files``) are fetched, so the
download allowance is controlled in one place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

USER_AGENT = "plasma-etch-vm/0.1 (research download script)"


@dataclass(frozen=True)
class RemoteFile:
    key: str
    size: int
    md5: str
    url: str


def fetch_record(api_url: str, cache: Path | None = None) -> dict:
    """Return the record JSON (cached on disk if ``cache`` is given and exists)."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read().decode("utf-8")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(payload, encoding="utf-8")
    return json.loads(payload)


def list_files(record: dict) -> dict[str, RemoteFile]:
    out = {}
    for f in record["files"]:
        algo, digest = f["checksum"].split(":", 1)
        if algo != "md5":
            raise ValueError(f"unexpected checksum algorithm {algo}")
        out[f["key"]] = RemoteFile(f["key"], int(f["size"]), digest, f["links"]["self"])
    return out


def md5sum(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, remote: RemoteFile) -> bool:
    return path.exists() and path.stat().st_size == remote.size and md5sum(path) == remote.md5


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def download(remote: RemoteFile, dest_dir: Path, min_free_gb: float = 15.0, retries: int = 8) -> Path:
    """Download one file to ``<name>.part`` (resumable via HTTP Range), verify size + MD5, then rename.

    * refuses to start if free disk space after the download would drop below ``min_free_gb``;
    * survives dropped connections and short reads (resumes from the current ``.part`` size);
    * handles servers that ignore ``Range`` (status 200) by restarting the file;
    * the final name only appears after verification, so globbing ``Day_*.nc`` never sees partial files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / remote.key
    if verify(dest, remote):
        log.info("%s already present and verified", remote.key)
        return dest
    part = dest_dir / (remote.key + ".part")
    have0 = part.stat().st_size if part.exists() else 0
    need = remote.size - have0
    if free_gb(dest_dir) - need / 1e9 < min_free_gb:
        raise RuntimeError(
            f"refusing to download {remote.key}: free {free_gb(dest_dir):.1f} GB - {need / 1e9:.2f} GB "
            f"would drop below {min_free_gb} GB"
        )
    t0 = time.perf_counter()
    failures = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have == remote.size:
            break
        if have > remote.size:
            part.unlink()
            have = 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(remote.url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                with open(part, mode) as fh:
                    shutil.copyfileobj(resp, fh, length=4 << 20)
        except Exception as exc:  # network hiccup -> resume
            failures += 1
            log.warning("attempt %d for %s failed at %.1f MB: %s", failures, remote.key, have / 1e6, exc)
            if failures > retries:
                raise RuntimeError(f"download of {remote.key} failed after {retries} retries") from exc
            time.sleep(min(60, 2**failures))
    dt = time.perf_counter() - t0
    if md5sum(part) != remote.md5:
        part.unlink()
        raise RuntimeError(f"checksum mismatch for {remote.key} (partial file removed)")
    os.replace(part, dest)
    got = (remote.size - have0) / 1e6
    log.info("%s: %.1f MB in %.0f s (%.2f MB/s)", remote.key, got, dt, got / max(dt, 1e-9))
    return dest


def write_manifest(files: dict[str, RemoteFile], raw_dir: Path, out: Path) -> list[dict]:
    """Write a manifest (size, md5, verified flag) of the remote record vs. local files."""
    rows = []
    for key, rf in sorted(files.items()):
        local = raw_dir / key
        rows.append(
            {
                "file": key,
                "size_bytes": rf.size,
                "md5": rf.md5,
                "url": rf.url,
                "downloaded": local.exists(),
                "verified": bool(local.exists() and verify(local, rf)),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows

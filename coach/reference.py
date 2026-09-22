"""Per-sentence own-voice reference reads.

A reference is the sentence read the way the learner wants to sound — their
own best take, a teacher, or a native speaker they recorded. Own-voice always
wins: this store never treats a TTS render as the reference. Files live under
the config data dir (default ~/.fluency-coach/references), keyed by sentence
text, with a JSON sidecar of the sentence and saved_at.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from .diff import _WORD_RE


def references_dir() -> Path:
    from .config import load
    return load().data_dir / "references"


class _ReferencesDir:
    """Path-like stand-in so older ``DIR / name`` reads follow config."""

    def path(self) -> Path:
        return references_dir()

    def __fspath__(self) -> str:
        return str(self.path())

    def __truediv__(self, other):
        return self.path() / other

    def __str__(self) -> str:
        return str(self.path())

    def mkdir(self, *args, **kwargs):
        return self.path().mkdir(*args, **kwargs)

    def exists(self) -> bool:
        return self.path().exists()


DIR = _ReferencesDir()


def key(sentence: str) -> str:
    toks = _WORD_RE.findall(sentence.lower())
    return hashlib.sha1(" ".join(toks).encode()).hexdigest()[:16]


def path_for(sentence: str) -> Path:
    return references_dir() / f"{key(sentence)}.wav"


def has(sentence: str) -> bool:
    return path_for(sentence).exists()


def origin(sentence: str) -> str:
    """'human' when an own-voice recording is stored, else 'none'."""
    return "human" if has(sentence) else "none"


def save(sentence: str, wav_path) -> Path:
    d = references_dir()
    d.mkdir(parents=True, exist_ok=True)
    dst = path_for(sentence)
    fd, tmp_name = tempfile.mkstemp(prefix=".ref-", suffix=".wav", dir=d)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(str(wav_path), tmp)
        os.replace(tmp, dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    side = d / f"{key(sentence)}.json"
    fd, side_name = tempfile.mkstemp(prefix=".ref-", suffix=".json", dir=d)
    os.close(fd)
    side_tmp = Path(side_name)
    try:
        side_tmp.write_text(
            json.dumps(
                {
                    "sentence": sentence,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(side_tmp, side)
    except Exception:
        side_tmp.unlink(missing_ok=True)
        raise
    return dst


def remove(sentence: str) -> bool:
    """Drop the saved reference for this sentence. True if something went."""
    d = references_dir()
    gone = False
    for suffix in (".wav", ".json"):
        p = d / f"{key(sentence)}{suffix}"
        if p.exists():
            p.unlink()
            gone = True
    return gone


def describe(sentence: str) -> str | None:
    p = path_for(sentence)
    if not p.exists():
        return None
    when = None
    side = p.with_suffix(".json")
    if side.exists():
        try:
            when = json.loads(side.read_text(encoding="utf-8")).get("saved_at")
        except (OSError, json.JSONDecodeError):
            when = None
    if not when:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    return f"reference saved {when}"

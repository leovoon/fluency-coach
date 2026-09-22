"""Jev judgment layer via the Experiential Labs System One endpoint.

One batched request per sentence (endpoint is charged per call):
  - Noul  per suspect word  — was it actually read?
  - Choice per suspect word — error category
  - Score  for the sentence — pronunciation evidence (transcript-shaped)

All phonetic truth lives in the ASR/diff layers; Jev judges text-level intent.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import httpx

API_URL = os.environ.get("EXP_API", "https://api.experientiallabs.ai/v1/systemone")
MODEL = "jev-latest"

# The endpoint validates its own model output (a decision inconsistent with its
# probability distribution comes back as an error). That is a bad sample, not a
# bad request — the same payload can answer cleanly on a fresh sample — so these
# are retried with backoff. Real request errors (4xx validation, quota, refusal)
# surface immediately.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRY_MARKERS = ("malformed response",)
_RETRY_DELAYS_S = (1.0, 3.0, 8.0)


def _error_detail(resp) -> str:
    """Readable one-liner from an error body. HTML bodies (Cloudflare outage
    pages in front of the upstream) get their <title>, not 300 chars of markup."""
    text = resp.text.strip()
    if text[:1] == "<":
        m = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        if m:
            title = " ".join(m.group(1).split())
            return f"error page: {title or '(untitled)'}"
        return "error page (html)"
    return text[:300]

ERROR_CATEGORIES = {
    "substituted-word": "Reader spoke a different real word in this slot (passage 'quiet', transcript 'quick').",
    "skipped": "No transcript token near the expected position; timing shows the reader moved on.",
    "inserted": "Reader added words the passage does not have between neighboring words.",
    "distorted-meaning": "Phonetically related but meaning-changing ('desert' for 'dessert').",
    "no-error": "The word was present after all — the local diff mis-flagged a valid variant.",
}

PRON_LEVELS = [
    "Clean: every expected word appears in the transcript exactly or as a standard variant; no phonetic misspellings.",
    "Minor: one or two words show a small phonetic distortion consistent with a single wrong sound ('three' for 'tree'); still recognizable.",
    "Heavy: several words are distorted or recognized as unrelated words; the sentence is reconstructable only with the passage in hand.",
    "Failed: words are missing from the transcript where the reader clearly attempted them; the ASR produced nothing usable.",
]

def _key() -> str:
    key = os.environ.get("EXPERIENTIAL_API_KEY")
    if key:
        return key
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-a", os.environ.get("USER", ""),
             "-s", "experiential_api_key", "-w"],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            "Keychain lookup timed out (service: experiential_api_key)"
        ) from e
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(
            "No EXPERIENTIAL_API_KEY env and no Keychain entry "
            "(service: experiential_api_key)"
        )
    return out.stdout.strip()


def judge_sentence(sentence: str, transcript: str, suspects: list[dict]) -> dict:
    """One batched Jev request for a stabilized sentence.

    suspects: [{"index": i, "word": "quiet", "status": "sub"|"skip", "spoken": "quick"|None}]
    Returns {"read": {i: prob}, "category": {i: {...}}, "pronunciation": score}.
    """
    questions: dict = {}

    for s in suspects[:10]:
        i = s["index"]
        spoken = s.get("spoken")
        questions[f"read_{i}"] = {
            "type": "noul",
            "instructions": {
                "expected_word": s["word"],
                "asr_heard_nearby": spoken if spoken else "(nothing heard in this slot)",
                "question": (
                    f"Did the reader actually speak the expected word `{s['word']}`? "
                    "Accept exact matches, inflection variants (read/reads), homophones "
                    "(there/their), and close phonetic renderings the ASR transcribed "
                    "differently. Do not count as read if the reader skipped it."
                ),
            },
            "criteria": {
                "true": "The expected word was spoken, possibly in a variant or mis-transcribed form.",
                "false": "The expected word was not spoken in this position.",
            },
        }
        if len(questions) < 11:
            questions[f"cat_{i}"] = {
                "type": "choice",
                "instructions": {
                    "expected_word": s["word"],
                    "local_diff_flagged": s["status"],
                    "asr_heard_nearby": spoken if spoken else "(nothing heard)",
                    "question": (
                        f"Classify what happened when the reader reached the expected word "
                        f"`{s['word']}`, based on the transcript."
                    ),
                },
                "criteria": ERROR_CATEGORIES,
            }

    questions["pronunciation"] = {
        "type": "score",
        "instructions": {
            "passage_sentence": sentence,
            "asr_transcript": transcript,
            "question": (
                "Rate the pronunciation evidence in this transcript against the passage. "
                "This is transcript evidence only — you cannot hear audio."
            ),
        },
        "criteria": PRON_LEVELS,
    }

    state = {
        "passage_sentence": sentence,
        "asr_transcript": transcript,
        "note": "The transcript comes from a streaming ASR reading the passage aloud.",
    }

    data = _post({"state": state, "model": MODEL, "questions": questions})

    read: dict[int, float] = {}
    category: dict[int, dict] = {}
    for qid, ans in data.get("answers", {}).items():
        if qid.startswith("read_"):
            read[int(qid.split("_")[1])] = ans.get("noul", 1.0)
        elif qid.startswith("cat_"):
            category[int(qid.split("_")[1])] = {
                "choice": ans.get("choice"),
                "probabilities": ans.get("probabilities", {}),
                "confidence": ans.get("confidence"),
            }
    pron = data.get("answers", {}).get("pronunciation", {})
    return {
        "read": read,
        "category": category,
        "pronunciation": {
            "score": pron.get("score"),
            "legend": pron.get("legend", {}),
            "confidence": pron.get("confidence"),
        },
        "model": data.get("model"),
        "usage": data.get("usage"),
    }


def _post(payload: dict) -> dict:
    """POST once per attempt, retrying transient provider failures.

    No Idempotency-Key: a replay would return the same rejected sample; we want
    a fresh one.
    """
    last: Exception = RuntimeError("Jev API call failed")
    for attempt in range(len(_RETRY_DELAYS_S) + 1):
        if attempt:
            time.sleep(_RETRY_DELAYS_S[attempt - 1])
        try:
            resp = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {_key()}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
        except httpx.TransportError as e:
            last = e
            continue
        if resp.status_code < 400:
            return resp.json()
        detail = _error_detail(resp)
        last = RuntimeError(f"Jev API {resp.status_code}: {detail}")
        if resp.status_code not in _RETRY_STATUS \
                and not any(m in resp.text for m in _RETRY_MARKERS):
            break
    raise last


if __name__ == "__main__":
    # Charged endpoint. Never call it unless the operator passes --run.
    if "--run" not in sys.argv:
        print("refusing live call; pass --run")
    else:
        demo = judge_sentence(
            "The quiet forest woke slowly.",
            "The quick forest woke slowly.",
            [{"index": 1, "word": "quiet", "status": "sub", "spoken": "quick"}],
        )
        print(json.dumps(demo, indent=2))

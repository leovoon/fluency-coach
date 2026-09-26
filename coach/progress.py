"""Attempt history: append-only JSONL log plus weekly medians.

Every scored attempt appends one row to <data_dir>/history.jsonl. The metrics
are whatever the scoring pass already computed — no new models, no new calls.
Reads compare this week's medians to last week's, not to your own improving
reference recordings.
"""
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def history_path(data_dir: Path) -> Path:
    return Path(data_dir) / "history.jsonl"


def log_attempt(data_dir: Path, row: dict) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           **row}
    with history_path(data_dir).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(data_dir: Path) -> list[dict]:
    p = history_path(data_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def collect_metrics(alignment, marks: dict | None, stress: dict | None,
                    wav: str | None) -> dict:
    """Pull the already-computed numbers for one scored attempt."""
    words = list(alignment.words) if alignment else []
    total = len(words)
    match = sum(1 for w in words if w.status == "match")
    m = {
        "wer": round(1 - match / total, 3) if total else None,
        "words": total,
        "stress_flat": 0,   # beats you flattened
        "stress_extra": 0,  # small words you pushed too loud
        "breath_ok": 0, "breath_miss": 0,
        "link_ok": 0, "link_choppy": 0,
        "hesitations": 0,   # pauses no mark planned
    }
    for s in (stress or {}).values():
        if s == "flat":
            m["stress_flat"] += 1
        elif s == "extra":
            m["stress_extra"] += 1
    for v in (marks or {}).values():
        kind, ok = v.get("kind"), v.get("ok")
        if kind == "pause":
            m["breath_ok" if ok is True else "breath_miss"] += 1
        elif kind == "link":
            m["link_ok" if ok is True else "link_choppy"] += 1
        elif ok is False:
            m["hesitations"] += 1
    if wav:
        import soundfile as sf
        try:
            m["duration_s"] = round(sf.info(str(wav)).duration, 2)
        except Exception:
            pass
    return m


def _median(rows: list[dict], key: str, scale: float = 1.0,
            ndigits: int = 1):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return round(statistics.median(vals) * scale, ndigits)


def _fmt(v):
    return "—" if v is None else (f"{v:.0%}" if isinstance(v, float) and v <= 1
                                  else str(v))


def weekly_summary(rows: list[dict]) -> str:
    """One small line per week: median WER and missed beats, last two weeks."""
    if not rows:
        return ""
    by_week: dict[str, list[dict]] = {}
    for r in rows:
        try:
            week = datetime.fromisoformat(r["ts"]).strftime("%G-W%V")
        except Exception:
            continue
        by_week.setdefault(week, []).append(r)
    weeks = sorted(by_week)[-2:]
    lines = []
    for w in weeks:
        rs = by_week[w]
        lines.append(
            f"{len(rs)} takes · wer {_fmt(_median(rs, 'wer', ndigits=3))} · "
            f"flattened {_median(rs, 'stress_flat')}/sent · "
            f"missed breaths {_median(rs, 'breath_miss')} · "
            f"choppy links {_median(rs, 'link_choppy')}"
        )
    if len(lines) == 1:
        return f"this week: {lines[0]}"
    return f"this week: {lines[1]}\nlast week: {lines[0]}"

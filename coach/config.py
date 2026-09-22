"""Engine and feature settings. One file (coach.yaml) plus CLI flags.

Own-voice reference always wins over TTS. Swap STT/TTS or drop to a lighter
tier without editing Python: change asr.id, tts.engine, or device.

Load order: tier preset, then yaml, then CLI. A ``--tier`` flag re-applies
that preset (so it beats yaml feature toggles) and then identity keys from
yaml (device, model id, voice, lang, data_dir, host, port) are put back.
``--no-*``, ``--device``, ``--host``, and ``--port`` still win last.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ASR_ID = "moondream/parakeet-redux"
DEFAULT_TTS_VOICE = "af_heart"
DEFAULT_TTS_LANG = "en-us"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

TIERS = ("lite", "standard", "pro")
DEVICE_PREFS = ("auto", "cpu", "mps", "cuda")
TTS_ENGINES = ("kokoro", "system", "none")

# Feature bundle plus the TTS engine that bundle implies.
_PRESETS: dict[str, dict] = {
    "lite": {
        "melody": False,
        "flow": False,
        "phonemes": False,
        "jev": False,
        "tts": False,
        "ref_compare": True,
        "tts_engine": "none",
    },
    "standard": {
        "melody": True,
        "flow": True,
        "phonemes": False,
        "jev": False,
        "tts": True,
        "ref_compare": True,
        "tts_engine": "kokoro",
    },
    "pro": {
        "melody": True,
        "flow": True,
        "phonemes": True,
        "jev": True,
        "tts": True,
        "ref_compare": True,
        "tts_engine": "kokoro",
    },
}

_VALUE_FLAGS = {"--tier", "--device", "--host", "--port"}
_BOOL_FLAGS = {
    "--no-jev": "jev",
    "--no-tts": "tts",
    "--no-phonemes": "phonemes",
    "--no-melody": "melody",
    "--no-flow": "flow",
}
_KNOWN_FLAGS = _VALUE_FLAGS | set(_BOOL_FLAGS) | {"--help", "-h"}


@dataclass(frozen=True)
class Settings:
    tier: str
    device_pref: str
    device: str
    asr_engine: str
    asr_id: str
    tts_engine: str
    tts_voice: str
    tts_lang: str
    jev_api: str
    jev_model: str
    jev_api_key: str
    melody: bool
    flow: bool
    phonemes: bool
    jev: bool
    tts: bool
    ref_compare: bool
    data_dir: Path
    host: str
    port: int


def default_data_dir() -> Path:
    env = os.environ.get("FLUENCY_DATA")
    if env and env.strip():
        return Path(os.path.expandvars(env)).expanduser()
    return Path.home() / ".fluency-coach"


def resolve_device(pref: str) -> str:
    """Resolve auto without importing torch.

    auto -> mps on Apple Silicon (Darwin + machine startswith arm), else cpu.
    """
    pref = str(pref).strip().lower()
    if pref not in DEVICE_PREFS:
        raise RuntimeError(
            f"device must be auto, cpu, mps, or cuda (got {pref!r})"
        )
    if pref != "auto":
        return pref
    machine = platform.machine().lower()
    if platform.system() == "Darwin" and machine.startswith("arm"):
        return "mps"
    return "cpu"


def config_path() -> Path | None:
    env = os.environ.get("FLUENCY_CONFIG")
    if env and env.strip():
        path = Path(os.path.expandvars(env)).expanduser()
        if not path.exists():
            raise RuntimeError(f"FLUENCY_CONFIG not found: {path}")
        return path
    local = Path.cwd() / "coach.yaml"
    if local.is_file():
        return local
    return None


def load(argv: list[str] | None = None) -> Settings:
    raw = _read_config()
    flags, _positional = _split_argv(argv)
    tier = _tier(raw.get("tier", "standard"))
    fields = _defaults()
    _apply_preset(fields, tier)
    _apply_yaml(fields, raw)
    if "--tier" in flags:
        tier = _tier(flags["--tier"])
        _apply_preset(fields, tier)
        _apply_yaml(fields, raw, identity_only=True)
    fields["tier"] = tier
    _apply_cli(fields, flags)
    return _build(fields)


def passage_arg(argv: list[str] | None = None) -> str | None:
    """First positional argument. Tokens starting with -- are never a path."""
    _flags, positional = _split_argv(argv)
    return positional[0] if positional else None


def _defaults() -> dict:
    return {
        "tier": "standard",
        "device_pref": "auto",
        "asr_engine": "photon",
        "asr_id": DEFAULT_ASR_ID,
        "tts_engine": "kokoro",
        "tts_voice": DEFAULT_TTS_VOICE,
        "tts_lang": DEFAULT_TTS_LANG,
        "jev_api": "",
        "jev_model": "",
        "jev_api_key": "",
        "melody": True,
        "flow": True,
        "phonemes": False,
        "jev": False,
        "tts": True,
        "ref_compare": True,
        "data_dir": default_data_dir(),
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
    }


def _tier(value) -> str:
    tier = str(value).strip().lower()
    if tier not in TIERS:
        raise RuntimeError(f"tier must be lite, standard, or pro (got {value!r})")
    return tier


def _apply_preset(fields: dict, tier: str) -> None:
    fields["tier"] = tier
    fields.update(_PRESETS[tier])


def _apply_yaml(fields: dict, raw: dict, identity_only: bool = False) -> None:
    if not raw:
        return
    if "device" in raw:
        fields["device_pref"] = str(raw["device"]).strip().lower()
    if "device_pref" in raw:
        fields["device_pref"] = str(raw["device_pref"]).strip().lower()
    asr = raw.get("asr")
    if isinstance(asr, dict):
        if "engine" in asr:
            fields["asr_engine"] = str(asr["engine"]).strip().lower()
        if "id" in asr:
            fields["asr_id"] = str(asr["id"]).strip()
    if "asr_id" in raw:
        fields["asr_id"] = str(raw["asr_id"]).strip()
    if "asr_engine" in raw:
        fields["asr_engine"] = str(raw["asr_engine"]).strip().lower()
    tts = raw.get("tts")
    if isinstance(tts, dict):
        if "engine" in tts and not identity_only:
            fields["tts_engine"] = str(tts["engine"]).strip().lower()
        if "voice" in tts:
            fields["tts_voice"] = str(tts["voice"]).strip()
        if "lang" in tts:
            fields["tts_lang"] = str(tts["lang"]).strip()
        if "enabled" in tts and not identity_only:
            fields["tts"] = _as_bool(tts["enabled"], "tts.enabled")
    elif isinstance(tts, (bool, str)) and not identity_only:
        fields["tts"] = _as_bool(tts, "tts")
    jev = raw.get("jev")
    if isinstance(jev, dict):
        if "api" in jev and jev["api"]:
            fields["jev_api"] = str(jev["api"]).strip()
        if "model" in jev and jev["model"]:
            fields["jev_model"] = str(jev["model"]).strip()
        if "api_key" in jev and jev["api_key"]:
            fields["jev_api_key"] = str(jev["api_key"]).strip()
    if not identity_only:
        for key in ("melody", "flow", "phonemes", "ref_compare"):
            if key in raw:
                fields[key] = _as_bool(raw[key], key)
        # jev is either a flat bool switch or a mapping (api/model) — a mapping
        # implies the feature is on unless a flat jev key also sets it false.
        if isinstance(raw.get("jev"), (bool, str)):
            fields["jev"] = _as_bool(raw["jev"], "jev")
        elif isinstance(raw.get("jev"), dict):
            fields["jev"] = True
        if "read_back" in raw:
            fields["tts"] = _as_bool(raw["read_back"], "read_back")
    if "data_dir" in raw and raw["data_dir"]:
        fields["data_dir"] = Path(os.path.expandvars(str(raw["data_dir"]))).expanduser()
    if "host" in raw:
        fields["host"] = str(raw["host"]).strip()
    if "port" in raw:
        fields["port"] = int(raw["port"])


def _apply_cli(fields: dict, flags: dict) -> None:
    if "--device" in flags:
        fields["device_pref"] = str(flags["--device"]).strip().lower()
    if "--host" in flags:
        fields["host"] = str(flags["--host"]).strip()
    if "--port" in flags:
        try:
            fields["port"] = int(flags["--port"])
        except ValueError as e:
            raise RuntimeError(f"--port must be an integer (got {flags['--port']!r})") from e
    for flag, key in _BOOL_FLAGS.items():
        if flag in flags:
            fields[key] = False


def _build(fields: dict) -> Settings:
    engine = str(fields["tts_engine"]).strip().lower()
    if engine not in TTS_ENGINES:
        raise RuntimeError(
            f"tts.engine must be kokoro, system, or none (got {engine!r})"
        )
    fields["tts_engine"] = engine
    asr_engine = str(fields["asr_engine"]).strip().lower() or "photon"
    fields["asr_engine"] = asr_engine
    if not str(fields["asr_id"]).strip():
        raise RuntimeError("asr.id must not be empty")
    host = str(fields["host"]).strip()
    if not host:
        raise RuntimeError("host must not be empty")
    port = int(fields["port"])
    if not (0 < port < 65536):
        raise RuntimeError(f"port out of range: {port}")
    data_dir = fields["data_dir"]
    if not isinstance(data_dir, Path):
        data_dir = Path(os.path.expandvars(str(data_dir))).expanduser()
    return Settings(
        tier=fields["tier"],
        device_pref=str(fields["device_pref"]).strip().lower(),
        device=resolve_device(fields["device_pref"]),
        asr_engine=fields["asr_engine"],
        asr_id=str(fields["asr_id"]).strip(),
        tts_engine=engine,
        tts_voice=str(fields["tts_voice"]).strip() or DEFAULT_TTS_VOICE,
        tts_lang=str(fields["tts_lang"]).strip() or DEFAULT_TTS_LANG,
        jev_api=str(fields["jev_api"]).strip(),
        jev_model=str(fields["jev_model"]).strip(),
        jev_api_key=str(fields["jev_api_key"]).strip(),
        melody=bool(fields["melody"]),
        flow=bool(fields["flow"]),
        phonemes=bool(fields["phonemes"]),
        jev=bool(fields["jev"]),
        tts=bool(fields["tts"]),
        ref_compare=bool(fields["ref_compare"]),
        data_dir=data_dir,
        host=host,
        port=port,
    )


def _read_config() -> dict:
    path = config_path()
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    data = _load_yaml(text, path)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must be a mapping")
    return data


def _load_yaml(text: str, path: Path) -> dict | None:
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)
    try:
        return yaml.safe_load(text)
    except Exception as e:
        raise RuntimeError(f"failed to parse {path}: {e}") from e


def _parse_simple_yaml(text: str) -> dict:
    """Indent-based subset used only when PyYAML is missing. No lists."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.split("#", 1)[0].rstrip()
        if ":" not in line:
            continue
        key, _, val = line.strip().partition(":")
        key = key.strip()
        val = val.strip().strip("'\"")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(val)
    return root


def _scalar(val: str):
    low = val.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if val.isdigit():
        return int(val)
    return val


def _as_bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    raise RuntimeError(f"{key} must be true or false (got {value!r})")


def _split_argv(argv: list[str] | None) -> tuple[dict[str, str], list[str]]:
    tokens = _tokens(argv)
    flags: dict[str, str] = {}
    positional: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"-h", "--help"}:
            flags["--help"] = "1"
            i += 1
            continue
        if tok.startswith("--"):
            if "=" in tok:
                key, val = tok.split("=", 1)
                _take_flag(flags, key, val)
                i += 1
                continue
            if tok in _VALUE_FLAGS:
                if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                    raise RuntimeError(f"{tok} needs a value")
                _take_flag(flags, tok, tokens[i + 1])
                i += 2
                continue
            if tok not in _KNOWN_FLAGS:
                raise RuntimeError(f"unknown flag {tok}")
            flags[tok] = "1"
            i += 1
            continue
        positional.append(tok)
        i += 1
    return flags, positional


def _take_flag(flags: dict[str, str], key: str, val: str) -> None:
    if key not in _VALUE_FLAGS:
        raise RuntimeError(f"unknown flag {key}")
    if val == "":
        raise RuntimeError(f"{key} needs a value")
    flags[key] = val


def _tokens(argv: list[str] | None) -> list[str]:
    if argv is None:
        return list(sys.argv[1:])
    argv = list(argv)
    # main() passes sys.argv. argv[0] is the program, never a passage.
    if argv and argv[0] == sys.argv[0]:
        return argv[1:]
    return argv

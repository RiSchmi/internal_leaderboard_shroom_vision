"""Upload validation for the internal leaderboard.

Accepts either individual ``*.jsonl`` prediction files (one per language,
like ``final_submission.en.jsonl``) or a single ``.zip`` containing them.

Every record must have exactly the expected structure:

    {"id": <str>, "labels": [{"start": int, "end": int,
                              "prob": float, "label": str}, ...]}

Anything else is rejected with a human-readable error message.
"""

from __future__ import annotations

import io
import json
import re
import zipfile

LANGUAGES = ("en", "fr", "it", "zh")

REQUIRED_RECORD_KEYS = {"id", "labels"}
REQUIRED_SPAN_KEYS = {"start", "end", "prob", "label"}
VALID_SPAN_LABELS = {
    "invention",
    "mischaracterization",
    "mischaracterisation",
    "ocr",
    "ocr problem",
    "miscounting",
    "miscount",
    "other",
}

_LANG_RE = re.compile(r"(?:^|[._-])(" + "|".join(LANGUAGES) + r")(?:[._-]|$)")


class ValidationError(Exception):
    """Raised when an uploaded file does not match the expected format."""


def detect_language(filename: str) -> str:
    """Infer the language code from a filename like ``final_submission.en.jsonl``."""
    stem = filename.rsplit("/", 1)[-1]
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    match = _LANG_RE.search(stem.lower())
    if not match:
        raise ValidationError(
            f"'{filename}': cannot detect language. The filename must contain "
            f"one of {list(LANGUAGES)} separated by '.', '_' or '-' "
            "(e.g. final_submission.en.jsonl)."
        )
    return match.group(1)


def _check_span(span, filename: str, line_no: int) -> None:
    if not isinstance(span, dict):
        raise ValidationError(f"'{filename}' line {line_no}: span is not an object.")
    if set(span.keys()) != REQUIRED_SPAN_KEYS:
        raise ValidationError(
            f"'{filename}' line {line_no}: span keys {sorted(span.keys())} != "
            f"required {sorted(REQUIRED_SPAN_KEYS)}."
        )
    if not isinstance(span["start"], int) or not isinstance(span["end"], int):
        raise ValidationError(
            f"'{filename}' line {line_no}: span start/end must be integers."
        )
    if span["end"] <= span["start"] or span["start"] < 0:
        raise ValidationError(
            f"'{filename}' line {line_no}: invalid span range "
            f"[{span['start']}, {span['end']})."
        )
    try:
        prob = float(span["prob"])
    except (TypeError, ValueError):
        raise ValidationError(f"'{filename}' line {line_no}: span prob is not a number.")
    if not 0.0 <= prob <= 1.0:
        raise ValidationError(
            f"'{filename}' line {line_no}: span prob {prob} outside [0, 1]."
        )
    if str(span["label"]).strip().lower() not in VALID_SPAN_LABELS:
        raise ValidationError(
            f"'{filename}' line {line_no}: unknown span label '{span['label']}'."
        )


def parse_jsonl(raw: bytes, filename: str) -> list[dict]:
    """Parse and strictly validate one prediction .jsonl file."""
    records = []
    seen_ids = set()
    text = raw.decode("utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"'{filename}' line {line_no}: invalid JSON ({exc}).")
        if not isinstance(rec, dict):
            raise ValidationError(f"'{filename}' line {line_no}: record is not an object.")
        if not REQUIRED_RECORD_KEYS.issubset(rec.keys()):
            raise ValidationError(
                f"'{filename}' line {line_no}: record keys {sorted(rec.keys())} "
                f"must include {sorted(REQUIRED_RECORD_KEYS)}."
            )
        if not isinstance(rec["id"], str) or not rec["id"]:
            raise ValidationError(f"'{filename}' line {line_no}: 'id' must be a non-empty string.")
        if rec["id"] in seen_ids:
            raise ValidationError(f"'{filename}' line {line_no}: duplicate id '{rec['id']}'.")
        seen_ids.add(rec["id"])
        if not isinstance(rec["labels"], list):
            raise ValidationError(f"'{filename}' line {line_no}: 'labels' must be a list.")
        for span in rec["labels"]:
            _check_span(span, filename, line_no)
        records.append({"id": rec["id"], "labels": rec["labels"]})
    if not records:
        raise ValidationError(f"'{filename}': file contains no records.")
    return records


def collect_uploads(uploaded_files) -> dict[str, list[dict]]:
    """
    Turn Streamlit UploadedFile objects (jsonl and/or zip) into
    {language: [records]}. Raises ValidationError on any problem.
    """
    per_language: dict[str, list[dict]] = {}

    def add(filename: str, raw: bytes) -> None:
        lang = detect_language(filename)
        if lang in per_language:
            raise ValidationError(
                f"Duplicate language '{lang}' (second file: '{filename}'). "
                "Upload at most one file per language."
            )
        per_language[lang] = parse_jsonl(raw, filename)

    for uf in uploaded_files:
        name = uf.name
        raw = uf.getvalue()
        if name.lower().endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
            except zipfile.BadZipFile:
                raise ValidationError(f"'{name}' is not a valid zip archive.")
            members = [
                m for m in zf.namelist()
                if m.lower().endswith(".jsonl") and not m.startswith("__MACOSX")
            ]
            if not members:
                raise ValidationError(f"'{name}' contains no .jsonl files.")
            for member in members:
                add(member, zf.read(member))
        elif name.lower().endswith(".jsonl"):
            add(name, raw)
        else:
            raise ValidationError(
                f"'{name}': unsupported file type. Upload .jsonl files or a .zip."
            )

    if not per_language:
        raise ValidationError("No prediction files found in the upload.")
    return per_language

"""Scoring: run evaluation/validate_shroom.py against the gold test split."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

from .validation import LANGUAGES, ValidationError

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(_THIS_DIR)              # internal_leaderboard/
REPO_ROOT = os.path.dirname(APP_DIR)              # ACL-August-8-2026/ (monorepo)

# Locate validate_shroom.py. A vendored copy inside lib/ makes the app
# self-contained when it is deployed as its own repo (Streamlit Cloud); the
# monorepo copy under evaluation/ is preferred when present so the leaderboard
# always tracks the canonical scorer.
VALIDATOR_CANDIDATES = [
    os.path.join(REPO_ROOT, "evaluation", "validate_shroom.py"),
    os.path.join(_THIS_DIR, "validate_shroom.py"),
]

# Gold lookup order: leaderboard-local copy (always present, works on Streamlit
# Cloud where participant_kit/.../data is gitignored), then the participant_kit
# original when running inside the monorepo.
GOLD_DIRS = [
    os.path.join(APP_DIR, "data", "gold"),
    os.path.join(
        REPO_ROOT, "participant_kit", "baselines", "hallushift++",
        "data", "splits", "test",
    ),
]
GOLD_TEMPLATE = "shroom-vision.test.{lang}.labeled.jsonl"


def _validator_path() -> str:
    for path in VALIDATOR_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "validate_shroom.py not found. Looked in: " + ", ".join(VALIDATOR_CANDIDATES)
    )


def _load_validator():
    path = _validator_path()
    spec = importlib.util.spec_from_file_location("validate_shroom", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_shroom"] = module
    spec.loader.exec_module(module)
    return module


def gold_path(lang: str) -> str:
    for gold_dir in GOLD_DIRS:
        path = os.path.join(gold_dir, GOLD_TEMPLATE.format(lang=lang))
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No gold file found for language '{lang}'.")


def _load_gold_ids(lang: str) -> set[str]:
    ids = set()
    with open(gold_path(lang), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def score_submission(per_language: dict[str, list[dict]]) -> dict:
    """
    Score {language: [prediction records]} with validate_shroom.score().

    All languages are concatenated and scored in ONE run so the "overall"
    metrics are pooled exactly like validate_shroom would pool them, and the
    per-language breakdown comes from its native `per_language` output.

    Returns the full validate_shroom results dict, augmented with
    ``languages`` and ``coverage`` info.
    """
    validator = _load_validator()

    coverage = {}
    gold_ids_by_lang = {}
    for lang, records in per_language.items():
        if lang not in LANGUAGES:
            raise ValidationError(f"Unsupported language '{lang}'.")
        gold_ids = _load_gold_ids(lang)
        gold_ids_by_lang[lang] = gold_ids
        pred_ids = {r["id"] for r in records}
        unknown = pred_ids - gold_ids
        if unknown:
            sample = ", ".join(sorted(unknown)[:5])
            raise ValidationError(
                f"[{lang}] {len(unknown)} prediction id(s) not present in the "
                f"gold test split (e.g. {sample}). Are these test-split predictions?"
            )
        coverage[lang] = {
            "predicted": len(pred_ids),
            "gold": len(gold_ids),
            "missing": len(gold_ids - pred_ids),
        }
    # Languages that were not uploaded at all: scored with empty predictions
    # so the overall metrics stay comparable across submissions.
    for lang in LANGUAGES:
        if lang not in per_language:
            gold_ids_by_lang[lang] = _load_gold_ids(lang)
            coverage[lang] = {
                "predicted": 0,
                "gold": len(gold_ids_by_lang[lang]),
                "missing": len(gold_ids_by_lang[lang]),
            }

    # Concatenate predictions and gold across ALL languages into temp files.
    # Gold ids without a prediction get an EMPTY prediction so every
    # submission is scored on the full test set (no cherry-picking).
    with tempfile.TemporaryDirectory() as tmp:
        pred_file = os.path.join(tmp, "predictions.jsonl")
        ref_file = os.path.join(tmp, "reference.jsonl")
        with open(pred_file, "w", encoding="utf-8") as fh:
            for lang in LANGUAGES:
                covered = set()
                for rec in per_language.get(lang, []):
                    covered.add(rec["id"])
                    fh.write(json.dumps({"id": rec["id"], "predictions": rec["labels"]}) + "\n")
                for missing_id in sorted(gold_ids_by_lang[lang] - covered):
                    fh.write(json.dumps({"id": missing_id, "predictions": []}) + "\n")
        with open(ref_file, "w", encoding="utf-8") as out:
            for lang in LANGUAGES:
                with open(gold_path(lang), "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            out.write(line.rstrip("\n") + "\n")

        results = validator.score(
            input_path=pred_file,
            pred_key="predictions",
            gold_key="labels",
            ref_path=ref_file,
        )

    results["languages"] = sorted(per_language)
    results["coverage"] = coverage
    return results

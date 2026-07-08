#!/usr/bin/env python3
"""
SHROOM-visions hallucination-detection scorer (unified).

Combines the character-level machinery of validate_shroom.py with the
aggregation conventions of the official scorer.py, so the numbers reported
here mirror the challenge master's final evaluation.

Both the gold reference and the predictions are expressed as a list of spans:

    [{"start": <int>, "end": <int>, "prob": <float>, "label": <str>}, ...]

`start`/`end` are character offsets into the model `response` string,
`prob` is the fraction of annotators that marked that span (empirical
probability, e.g. 1/3, 2/3, 1.0) and `label` is one of:

    invention | mischaracterization | ocr | miscounting | other

The scorer converts every span list into per-character vectors and reports:

  * Span identification ...... presence-based (prob > 0, NO 0.5 threshold,
                               matching scorer.score_iou): mean IoU, micro
                               IoU, character-level precision / recall / F1
  * Confidence calibration ... mean Spearman (mean of per-sentence Spearman,
                               scorer.py style), micro Spearman (global
                               pooled correlation), MAE
  * Classification ........... accuracy (intersection & gold-hallucinated)
                               and macro-F1 of the predicted category

Results are written to a JSON file with a fresh run id under --output-dir.

Usage
-----
Single file holding both gold ("labels") and predictions under <pred_key>:

    python validate_shroom.py --input preds.jsonl --pred-key predictions

Separate gold reference and prediction files (matched by "id"):

    python validate_shroom.py --input preds.jsonl --pred-key predictions \\
        --ref data/splits/test/shroom-vision.test.en.labeled.jsonl
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import uuid
from collections import defaultdict

import numpy as np

# Canonical hallucination categories.
CATEGORIES = ["invention", "mischaracterization", "ocr", "miscounting", "other"]

# Normalisation map for the various spellings seen in the data / submissions.
_LABEL_ALIASES = {
    "invention": "invention",
    "mischaracterization": "mischaracterization",
    "mischaracterisation": "mischaracterization",
    "ocr": "ocr",
    "ocr problem": "ocr",
    "ocrproblem": "ocr",
    "miscounting": "miscounting",
    "miscount": "miscounting",
    "other": "other",
}


def normalize_label(label):
    """Map a raw label string onto one of the canonical CATEGORIES."""
    if label is None:
        return "other"
    key = str(label).strip().lower()
    return _LABEL_ALIASES.get(key, "other")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    """Load a .jsonl file into a list of dicts (blank lines ignored)."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_no}: {exc}"
                ) from exc
    return rows


# --------------------------------------------------------------------------- #
# Span -> per-character conversion
# --------------------------------------------------------------------------- #
def spans_to_char_matrix(spans, length):
    """
    Convert a list of spans into a (length x n_categories) probability matrix.

    For each character and category the value is the maximum `prob` over all
    spans of that category covering the character (spans of the same category
    are expected to be disjoint, but `max` makes overlaps harmless).
    """
    mat = np.zeros((length, len(CATEGORIES)), dtype=np.float64)
    if not spans:
        return mat
    cat_index = {c: i for i, c in enumerate(CATEGORIES)}
    for span in spans:
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        prob = span.get("prob", 1.0)
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            prob = 1.0
        cat = cat_index[normalize_label(span.get("label"))]
        # Clamp to the valid character range.
        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end <= start:
            continue
        np.maximum(mat[start:end, cat], prob, out=mat[start:end, cat])
    return mat


def char_matrix_to_vectors(mat):
    """
    From the (length x n_categories) matrix derive:
      * hall_prob  : per-char hallucination probability (sum over categories,
                     capped at 1.0 — each annotator picks one category, so the
                     category probabilities add up to the total marked fraction)
      * dom_label  : per-char dominant category index, or -1 if no span
    """
    if mat.shape[0] == 0:
        return np.zeros(0), np.full(0, -1, dtype=int)
    hall_prob = np.minimum(mat.sum(axis=1), 1.0)
    dom_label = np.where(mat.sum(axis=1) > 0, mat.argmax(axis=1), -1)
    return hall_prob, dom_label


# --------------------------------------------------------------------------- #
# Correlation helpers (numpy-only, no scipy dependency)
# --------------------------------------------------------------------------- #
def pearson(x, y):
    if len(x) < 2:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a):
    """Average-rank of the data (ties share the mean rank)."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # Resolve ties by averaging.
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # mean of ranks (1-based) in the tie block
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return pearson(_rankdata(x), _rankdata(y))


def sentence_spearman(ref_vec, pred_vec):
    """
    Per-sentence Spearman in the style of scorer.score_cor:
      * ref_vec  = summed gold span probs per char (uncapped)
      * pred_vec = predicted per-char prob
      * constant series => 0/1 exact-match fallback.
    """
    ref_cmps = {round(float(v), 8) for v in ref_vec}
    pred_cmps = {round(float(v), 8) for v in pred_vec}
    if len(pred_cmps) == 1 or len(ref_cmps) == 1:
        if len(pred_cmps) != len(ref_cmps):
            return 0.0
        if ref_cmps == {0.0}:
            return float(pred_cmps == {0.0})
        return float(pred_cmps != {0.0})
    return spearman(
        np.asarray(ref_vec, dtype=np.float64),
        np.asarray(pred_vec, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Core scoring
# --------------------------------------------------------------------------- #
def score(
    input_path,
    pred_key,
    gold_key="labels",
    ref_path=None,
    text_key="response",
    id_key="id",
):
    """Compute all metrics and return a results dict."""
    rows = load_jsonl(input_path)

    # Build the gold lookup (either from a separate reference file or inline).
    gold_by_id = None
    if ref_path:
        gold_by_id = {}
        for r in load_jsonl(ref_path):
            if id_key in r:
                gold_by_id[r[id_key]] = r

    # Pooled per-character accumulators (capped probs) for micro Spearman / MAE.
    all_pred_prob = []
    all_gold_prob = []
    # Per-sentence Spearman (scorer.py style) -> mean Spearman.
    sent_cor = []

    # Span-identification accumulators.
    iou_per_sample = []
    tp = fp = fn = 0  # micro character counts

    # Classification accumulators (over characters hallucinated in BOTH).
    cls_correct_inter = cls_total_inter = 0
    # Accuracy over every gold-hallucinated character (pred miss => wrong).
    cls_correct_gold = cls_total_gold = 0
    # gold_label_idx -> pred_label_idx -> count  (pred index len(CATEGORIES) == "none")
    confusion = np.zeros((len(CATEGORIES), len(CATEGORIES) + 1), dtype=np.int64)

    per_lang = defaultdict(lambda: {"iou": [], "pred": [], "gold": [], "cor": []})

    n_scored = 0
    n_missing_gold = 0
    n_missing_pred = 0

    for row in rows:
        rid = row.get(id_key)

        # Resolve gold record.
        gold_row = row
        if gold_by_id is not None:
            gold_row = gold_by_id.get(rid)
            if gold_row is None:
                n_missing_gold += 1
                continue

        text = gold_row.get(text_key, row.get(text_key, ""))
        length = len(text)
        if length == 0:
            continue

        gold_spans = gold_row.get(gold_key, []) or []
        pred_spans = row.get(pred_key, None)
        if pred_spans is None:
            n_missing_pred += 1
            pred_spans = []

        gold_mat = spans_to_char_matrix(gold_spans, length)
        pred_mat = spans_to_char_matrix(pred_spans, length)

        gold_prob, gold_lab = char_matrix_to_vectors(gold_mat)
        pred_prob, pred_lab = char_matrix_to_vectors(pred_mat)

        # ---- Confidence calibration: pool every character. ----
        all_pred_prob.append(pred_prob)
        all_gold_prob.append(gold_prob)
        # Per-sentence Spearman (scorer.py style): gold summed (uncapped),
        # pred assigned; constant sentences fall back to a 0/1 exact match.
        sent_cor.append(sentence_spearman(gold_mat.sum(axis=1), pred_prob))

        # ---- Span identification: PRESENCE (prob > 0), no 0.5 threshold. ----
        gold_bin = gold_prob > 0
        pred_bin = pred_prob > 0
        inter = int(np.logical_and(gold_bin, pred_bin).sum())
        union = int(np.logical_or(gold_bin, pred_bin).sum())
        iou = 1.0 if union == 0 else inter / union
        iou_per_sample.append(iou)
        tp += inter
        fp += int(np.logical_and(pred_bin, ~gold_bin).sum())
        fn += int(np.logical_and(gold_bin, ~pred_bin).sum())

        # ---- Classification / confusion (per character). ----
        gold_hall_idx = np.where(gold_bin)[0]
        for i in gold_hall_idx:
            g = int(gold_lab[i])
            if g < 0:
                continue
            p = int(pred_lab[i]) if pred_bin[i] and pred_lab[i] >= 0 else len(CATEGORIES)
            confusion[g, p] += 1
            cls_total_gold += 1
            if p == g:
                cls_correct_gold += 1
            if pred_bin[i] and pred_lab[i] >= 0:
                cls_total_inter += 1
                if p == g:
                    cls_correct_inter += 1

        lang = gold_row.get("language", row.get("language", "unknown"))
        per_lang[lang]["iou"].append(iou)
        per_lang[lang]["pred"].append(pred_prob)
        per_lang[lang]["gold"].append(gold_prob)
        per_lang[lang]["cor"].append(sent_cor[-1])

        n_scored += 1

    # ---- Aggregate pooled vectors. ----
    pred_all = np.concatenate(all_pred_prob) if all_pred_prob else np.zeros(0)
    gold_all = np.concatenate(all_gold_prob) if all_gold_prob else np.zeros(0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    micro_iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0

    # Per-category F1 from the confusion matrix (over gold-hallucinated chars).
    f1_values = []
    for ci in range(len(CATEGORIES)):
        c_tp = int(confusion[ci, ci])
        c_fn = int(confusion[ci, :].sum() - c_tp)
        c_fp = int(confusion[:, ci].sum() - c_tp)
        p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
        r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
        lf1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        f1_values.append(lf1)

    # Per-language span identification + calibration.
    per_language = {}
    for lang, acc in per_lang.items():
        lp = np.concatenate(acc["pred"]) if acc["pred"] else np.zeros(0)
        lg = np.concatenate(acc["gold"]) if acc["gold"] else np.zeros(0)
        per_language[lang] = {
            "num_samples": len(acc["iou"]),
            "mean_iou": round(float(np.mean(acc["iou"])), 6) if acc["iou"] else None,
            "mean_spearman": round(float(np.nanmean(acc["cor"])), 6) if acc["cor"] else None,
            "micro_spearman": round(spearman(lp, lg), 6),
        }

    run_id = uuid.uuid4().hex[:12]
    results = {
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "input_file": os.path.abspath(input_path),
        "reference_file": os.path.abspath(ref_path) if ref_path else None,
        "pred_key": pred_key,
        "gold_key": gold_key,
        "num_samples_scored": n_scored,
        "num_characters": int(len(pred_all)),
        "num_missing_gold": n_missing_gold,
        "num_missing_predictions": n_missing_pred,
        "span_identification": {
            "mean_iou": round(float(np.mean(iou_per_sample)), 6) if iou_per_sample else None,
            "micro_iou": round(micro_iou, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "char_tp": tp,
            "char_fp": fp,
            "char_fn": fn,
        },
        "confidence_calibration": {
            "mean_spearman": round(float(np.nanmean(sent_cor)), 6) if sent_cor else None,
            "micro_spearman": round(spearman(pred_all, gold_all), 6),
            "mae": round(float(np.mean(np.abs(pred_all - gold_all))), 6) if len(pred_all) else None,
        },
        "classification": {
            "accuracy_on_intersection": round(cls_correct_inter / cls_total_inter, 6)
            if cls_total_inter
            else None,
            "accuracy_on_gold_hallucinated": round(cls_correct_gold / cls_total_gold, 6)
            if cls_total_gold
            else None,
            "macro_f1": round(float(np.mean(f1_values)), 6) if f1_values else None,
        },
        "per_language": per_language,
    }
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Score SHROOM-visions hallucination-detection predictions."
    )
    parser.add_argument("--input", required=True, help="Path to the .jsonl file to score.")
    parser.add_argument(
        "--pred-key",
        required=True,
        help="Key in each JSON object holding the predicted span list.",
    )
    parser.add_argument(
        "--gold-key",
        default="labels",
        help="Key holding the gold span list (default: labels).",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Optional separate gold reference .jsonl (matched on --id-key). "
        "If omitted, gold is read from --input via --gold-key.",
    )
    parser.add_argument("--id-key", default="id", help="Key used to match records (default: id).")
    parser.add_argument(
        "--text-key", default="response", help="Key holding the response text (default: response)."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores"),
        help="Directory to write the score JSON to (default: ./scores).",
    )
    args = parser.parse_args()

    results = score(
        input_path=args.input,
        pred_key=args.pred_key,
        gold_key=args.gold_key,
        ref_path=args.ref,
        text_key=args.text_key,
        id_key=args.id_key,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"score_{results['run_id']}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # Console summary.
    si = results["span_identification"]
    cc = results["confidence_calibration"]
    cl = results["classification"]
    print(f"Run id ............... {results['run_id']}")
    print(f"Samples scored ....... {results['num_samples_scored']}")
    print(f"Characters ........... {results['num_characters']}")
    print("-- Span identification (presence-based, no 0.5 threshold) --")
    print(f"  mean IoU ........... {si['mean_iou']}")
    print(f"  micro IoU .......... {si['micro_iou']}")
    print(f"  precision/recall/F1  {si['precision']} / {si['recall']} / {si['f1']}")
    print("-- Confidence calibration --")
    print(f"  mean Spearman ...... {cc['mean_spearman']}")
    print(f"  micro Spearman ..... {cc['micro_spearman']}")
    print(f"  MAE ................ {cc['mae']}")
    print("-- Classification --")
    print(f"  acc (intersection) . {cl['accuracy_on_intersection']}")
    print(f"  acc (gold halluc.) . {cl['accuracy_on_gold_hallucinated']}")
    print(f"  macro F1 ........... {cl['macro_f1']}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

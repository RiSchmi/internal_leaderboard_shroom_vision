"""SHROOM-Vis internal leaderboard (Streamlit).

Run locally:
    streamlit run internal_leaderboard/app.py

Deploy: Streamlit Community Cloud, main file = internal_leaderboard/app.py.
Secrets required — see .streamlit/secrets.toml.example.
"""

from __future__ import annotations

import hmac
import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.scoring import score_submission
from lib.storage import StorageError, github_config, load_submissions, save_submission
from lib.validation import LANGUAGES, ValidationError, collect_uploads

st.set_page_config(page_title="SHROOM-Vis Internal Leaderboard", page_icon="🍄", layout="wide")

# --------------------------------------------------------------------------- #
# Metric definitions (shown as ⓘ tooltips and in the reference expander)
# --------------------------------------------------------------------------- #
METRICS = {
    "mean_iou": (
        "mean IoU",
        "Intersection-over-Union per sample, then averaged (presence-based, "
        "prob > 0). Both gold and prediction empty ⇒ 1.0. Primary ranking "
        "metric; every response weighted equally, inflated by all-clean responses.",
    ),
    "micro_iou": (
        "micro IoU",
        "IoU pooled over ALL characters: TP / (TP+FP+FN). Equals F1/(2−F1), so "
        "it tracks precision/recall directly. Honest span-localization number.",
    ),
    "precision": (
        "precision",
        "TP / (TP+FP) over characters: of the characters you flagged, the "
        "fraction that are truly hallucinated. Low = over-flagging.",
    ),
    "recall": (
        "recall",
        "TP / (TP+FN) over characters: of the truly hallucinated characters, "
        "the fraction you caught. Low = missing hallucinations.",
    ),
    "f1": ("F1", "Harmonic mean of character-level precision and recall."),
    "mean_spearman": (
        "mean Spearman",
        "Mean of per-sentence Spearman between predicted and gold per-character "
        "probabilities (scorer.py style; constant sentences fall back to 0/1 "
        "exact match). Equal weight per response.",
    ),
    "micro_spearman": (
        "micro Spearman",
        "One global rank correlation over every character in the corpus. No "
        "per-sentence fallback; dominated by long responses.",
    ),
    "mae": (
        "MAE",
        "Mean absolute error |pred − gold| per character between predicted and "
        "gold empirical probability. Lower is better.",
    ),
    "acc_intersection": (
        "acc (intersection)",
        "Among characters BOTH gold and prediction mark as hallucinated, the "
        "fraction with the correct category. Pure labeling skill.",
    ),
    "acc_gold": (
        "acc (gold halluc.)",
        "Among ALL gold-hallucinated characters, the fraction labeled correctly "
        "(a missed character counts as wrong). Stricter; ≤ intersection accuracy.",
    ),
    "macro_f1": (
        "macro F1",
        "Per-category F1 averaged equally over the 5 hallucination categories "
        "(invention, mischaracterization, ocr, miscounting, other).",
    ),
}


def flatten_scores(scores: dict) -> dict:
    si = scores.get("span_identification", {})
    cc = scores.get("confidence_calibration", {})
    cl = scores.get("classification", {})
    return {
        "mean_iou": si.get("mean_iou"),
        "micro_iou": si.get("micro_iou"),
        "precision": si.get("precision"),
        "recall": si.get("recall"),
        "f1": si.get("f1"),
        "mean_spearman": cc.get("mean_spearman"),
        "micro_spearman": cc.get("micro_spearman"),
        "mae": cc.get("mae"),
        "acc_intersection": cl.get("accuracy_on_intersection"),
        "acc_gold": cl.get("accuracy_on_gold_hallucinated"),
        "macro_f1": cl.get("macro_f1"),
    }


# --------------------------------------------------------------------------- #
# Password gate
# --------------------------------------------------------------------------- #
def check_password() -> bool:
    try:
        expected = st.secrets["auth"]["password"]
    except Exception:
        st.error(
            "No password configured. Add `[auth] password = \"...\"` to "
            ".streamlit/secrets.toml (locally) or the app secrets (Streamlit Cloud)."
        )
        return False

    if st.session_state.get("authenticated"):
        return True

    st.title("🍄 SHROOM-Vis Internal Leaderboard")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hmac.compare_digest(password, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            time.sleep(1)  # crude brute-force damper
            st.error("Wrong password.")
    return False


if not check_password():
    st.stop()

# --------------------------------------------------------------------------- #
# Cached leaderboard loading
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=120, show_spinner="Loading leaderboard from GitHub ...")
def cached_submissions() -> list[dict]:
    return load_submissions(st.secrets)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("🍄 SHROOM-Vis Internal Leaderboard")
if github_config(st.secrets) is None:
    st.warning(
        "GitHub persistence is **not configured** — submissions are stored on "
        "the local disk only (fine for development, data loss on Streamlit Cloud). "
        "Add a `[github]` section to the secrets to enable it.",
        icon="⚠️",
    )

tab_upload, tab_board = st.tabs(["📤 Upload submission", "🏆 Leaderboard"])

# --------------------------------------------------------------------------- #
# Upload tab
# --------------------------------------------------------------------------- #
with tab_upload:
    st.markdown(
        f"""
Upload per-language prediction files (`*.<lang>.jsonl`, languages:
`{"`, `".join(LANGUAGES)}`) **or one zip** containing them. Each line must be

```json
{{"id": "...", "labels": [{{"start": 0, "end": 5, "prob": 0.9, "label": "invention"}}]}}
```

Files with any other structure are rejected. Missing test ids / languages are
scored as *empty* predictions so all submissions are comparable.
"""
    )

    with st.form("upload_form", clear_on_submit=False):
        name = st.text_input("Submission name *", max_chars=80)
        short_desc = st.text_input(
            "Architecture — short description *", max_chars=140,
            placeholder="e.g. HalluShift++ (LLaVA-1.5-7B hidden-state shifts + RF)",
        )
        long_desc = st.text_area(
            "Architecture — long description *",
            placeholder="Model, features, training data/splits, decoding, anything "
            "needed to reproduce the run.",
            height=150,
        )
        uploads = st.file_uploader(
            "Prediction files (.jsonl) or a .zip",
            type=["jsonl", "zip"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Validate, score & publish", type="primary")

    if submitted:
        if not name.strip() or not short_desc.strip() or not long_desc.strip():
            st.error("Name and both architecture descriptions are required.")
        elif not uploads:
            st.error("Please upload at least one prediction file.")
        else:
            try:
                with st.spinner("Validating upload ..."):
                    per_language = collect_uploads(uploads)
                st.success(
                    "Valid upload for language(s): "
                    + ", ".join(f"`{l}` ({len(r)} records)" for l, r in sorted(per_language.items()))
                )
                with st.spinner("Scoring against the gold test split ..."):
                    scores = score_submission(per_language)
                metadata = {
                    "name": name.strip(),
                    "short_description": short_desc.strip(),
                    "long_description": long_desc.strip(),
                    "languages": scores["languages"],
                    "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                }
                with st.spinner("Publishing to GitHub ..."):
                    sub_id = save_submission(st.secrets, metadata, scores, per_language)
                cached_submissions.clear()
                st.success(f"Submission **{sub_id}** published. 🎉")
                flat = flatten_scores(scores)
                st.metric("mean IoU (ranking metric)", f"{flat['mean_iou']:.4f}")
                st.json(flat, expanded=False)
            except (ValidationError, StorageError) as exc:
                st.error(f"❌ {exc}")

# --------------------------------------------------------------------------- #
# Leaderboard tab
# --------------------------------------------------------------------------- #
with tab_board:
    col_refresh, _ = st.columns([1, 5])
    if col_refresh.button("🔄 Refresh"):
        cached_submissions.clear()
        st.rerun()

    entries = cached_submissions()
    if not entries:
        st.info("No submissions yet — be the first!")
    else:
        rows = []
        for e in entries:
            flat = flatten_scores(e["scores"])
            rows.append(
                {
                    "name": e["metadata"].get("name", e["id"]),
                    "architecture": e["metadata"].get("short_description", ""),
                    "languages": ", ".join(e["metadata"].get("languages", [])),
                    "submitted": e["metadata"].get("submitted_at", ""),
                    **flat,
                    "_id": e["id"],
                }
            )
        df = pd.DataFrame(rows).sort_values("mean_iou", ascending=False, na_position="last")
        df.insert(0, "rank", range(1, len(df) + 1))

        column_config = {
            "rank": st.column_config.NumberColumn("#", width="small"),
            "name": st.column_config.TextColumn("Name"),
            "architecture": st.column_config.TextColumn("Architecture"),
            "languages": st.column_config.TextColumn("Langs", width="small"),
            "submitted": st.column_config.TextColumn("Submitted", width="small"),
            "_id": None,  # hidden
        }
        for key, (label, help_text) in METRICS.items():
            column_config[key] = st.column_config.NumberColumn(
                label, help=help_text, format="%.4f"
            )

        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            column_config=column_config,
        )

        with st.expander("ⓘ Metric definitions"):
            for key, (label, help_text) in METRICS.items():
                st.markdown(f"**{label}** — {help_text}")

        st.subheader("Details & per-language breakdown")
        by_id = {e["id"]: e for e in entries}
        for _, row in df.iterrows():
            e = by_id[row["_id"]]
            flat = flatten_scores(e["scores"])
            header = (
                f"#{row['rank']}  {row['name']}  —  mean IoU "
                f"{flat['mean_iou']:.4f}" if flat["mean_iou"] is not None else row["name"]
            )
            with st.expander(header):
                meta = e["metadata"]
                st.markdown(f"**Architecture:** {meta.get('short_description', '')}")
                if meta.get("long_description"):
                    st.markdown(meta["long_description"])
                st.caption(
                    f"Submitted {meta.get('submitted_at', '?')} · id `{e['id']}` · "
                    f"{e['scores'].get('num_samples_scored', '?')} samples · "
                    f"{e['scores'].get('num_characters', '?')} characters"
                )

                per_language = e["scores"].get("per_language", {})
                if per_language:
                    lang_df = pd.DataFrame(
                        [
                            {"language": lang, **vals}
                            for lang, vals in sorted(per_language.items())
                        ]
                    )
                    st.dataframe(
                        lang_df,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "language": st.column_config.TextColumn("Language"),
                            "num_samples": st.column_config.NumberColumn("Samples"),
                            "mean_iou": st.column_config.NumberColumn(
                                "mean IoU", help=METRICS["mean_iou"][1], format="%.4f"
                            ),
                            "mean_spearman": st.column_config.NumberColumn(
                                "mean Spearman", help=METRICS["mean_spearman"][1], format="%.4f"
                            ),
                            "micro_spearman": st.column_config.NumberColumn(
                                "micro Spearman", help=METRICS["micro_spearman"][1], format="%.4f"
                            ),
                        },
                    )

                coverage = e["scores"].get("coverage")
                if coverage:
                    missing = {l: c["missing"] for l, c in coverage.items() if c["missing"]}
                    if missing:
                        st.warning(
                            "Missing predictions (scored as empty): "
                            + ", ".join(f"{l}: {n}" for l, n in sorted(missing.items()))
                        )

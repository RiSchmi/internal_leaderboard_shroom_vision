# SHROOM-Vis Internal Leaderboard

Streamlit app for the internal SHROOM-Vis leaderboard: upload per-language
prediction files, score them against the gold **test** split with
[`evaluation/validate_shroom.py`](../evaluation/validate_shroom.py), and
publish results to this GitHub repository (single source of truth).

## Layout

```
internal_leaderboard/
├── app.py                     # Streamlit entry point (password gate, upload, board)
├── lib/
│   ├── validation.py          # strict format checks for .jsonl / .zip uploads
│   ├── scoring.py             # runs validate_shroom.score() vs data/gold
│   ├── storage.py             # GitHub Contents API read/write (local fallback)
│   └── validate_shroom.py     # vendored copy of ../evaluation/validate_shroom.py
├── data/gold/                 # test gold copies (participant_kit data is gitignored)
├── submissions/               # committed submissions (metadata, scores, predictions)
├── requirements.txt
└── .streamlit/secrets.toml.example
```

> **Self-contained on purpose.** When the app is deployed as its *own* repo on
> Streamlit Cloud (repo root = this folder), `../evaluation/` and
> `../participant_kit/` do not exist. `lib/validate_shroom.py` and `data/gold/`
> are vendored copies so scoring works standalone. Inside the monorepo the
> canonical `evaluation/validate_shroom.py` is used automatically (it takes
> precedence). **If you change the canonical scorer, re-copy it:**
> `cp evaluation/validate_shroom.py internal_leaderboard/lib/validate_shroom.py`.


## Run locally

```bash
pip install -r internal_leaderboard/requirements.txt
cp internal_leaderboard/.streamlit/secrets.toml.example internal_leaderboard/.streamlit/secrets.toml
# edit secrets.toml (at minimum set [auth] password)
streamlit run internal_leaderboard/app.py
```

Without a `[github]` section, submissions are stored under
`internal_leaderboard/submissions/` on the local disk.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub. **The repository must be PRIVATE** —
   `internal_leaderboard/data/gold/` contains the test-set labels.
2. Create a **fine-grained PAT**: repository access = this repo only,
   permissions = *Contents: Read and write*. Nothing else.
3. New app → pick the repo/branch → main file `internal_leaderboard/app.py`.
4. App → Settings → Secrets → paste the contents of your `secrets.toml`
   (`[auth] password` and `[github] token/repo/branch`).

Every upload is committed as
`internal_leaderboard/submissions/<timestamp>_<name>/{metadata.json,scores.json,predictions/*.jsonl}`.
The leaderboard reads submissions back through the GitHub API (cached 120 s),
so the app survives restarts and multiple viewers see the same state.

## Scoring conventions

- Submissions are always scored on **all 4 languages × full test set**;
  missing ids/languages count as empty predictions, so numbers are comparable
  across submissions (missing coverage is displayed on the details row).
- Ranking metric: **mean IoU** (per-sample, presence-based). See the ⓘ
  tooltips / “Metric definitions” expander in the app, or
  [`evaluation/validation_tool.md`](../evaluation/validation_tool.md).

## Security notes

- Single shared password (compared with `hmac.compare_digest`, 1 s failure
  delay). This is a *speed bump*, not real auth — do not expose gold data or
  tokens beyond this trust level.
- The GitHub token lives only in Streamlit secrets, never in the repo.

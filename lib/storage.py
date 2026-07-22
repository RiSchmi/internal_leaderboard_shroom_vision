"""Persistence: GitHub is the single source of truth for the leaderboard.

Streamlit Community Cloud has an ephemeral filesystem, so every submission
(metadata + scores + raw predictions) is committed to the GitHub repository
via the Contents API, and the leaderboard is read back through the same API.

Secrets (``.streamlit/secrets.toml`` locally, App settings -> Secrets on
Streamlit Cloud)::

    [github]
    token = "github_pat_..."   # fine-grained PAT, Contents: read+write, this repo only
    repo = "your-org/ACL-August-8-2026"
    branch = "main"

When no token is configured (local development) the module falls back to
reading/writing ``internal_leaderboard/submissions/`` on the local disk.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import unicodedata

import requests

from .scoring import APP_DIR

SUBMISSIONS_REL_DIR = "internal_leaderboard/submissions"
LOCAL_SUBMISSIONS_DIR = os.path.join(APP_DIR, "submissions")

_API = "https://api.github.com"


class StorageError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def github_config(secrets) -> dict | None:
    """Return {'token','repo','branch'} or None when GitHub is not configured."""
    try:
        gh = secrets["github"]
        token = gh["token"]
        repo = gh["repo"]
    except Exception:
        return None
    if not token or not repo:
        return None
    return {"token": token, "repo": repo, "branch": gh.get("branch", "main")}


def _headers(cfg):
    return {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def slugify(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return name or "submission"


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
def _gh_put_file(cfg, path: str, content: bytes, message: str) -> None:
    url = f"{_API}/repos/{cfg['repo']}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content).decode(),
        "branch": cfg["branch"],
    }
    resp = requests.put(url, headers=_headers(cfg), json=payload, timeout=30)
    if resp.status_code == 422 and "sha" in resp.text:
        # File exists -> fetch sha and update.
        get = requests.get(
            url, headers=_headers(cfg), params={"ref": cfg["branch"]}, timeout=30
        )
        if get.ok:
            payload["sha"] = get.json()["sha"]
            resp = requests.put(url, headers=_headers(cfg), json=payload, timeout=30)
    if not resp.ok:
        raise StorageError(f"GitHub write failed ({resp.status_code}): {resp.text[:300]}")


def save_submission(
    secrets,
    metadata: dict,
    scores: dict,
    per_language_records: dict[str, list[dict]],
) -> str:
    """Persist one submission; returns its id (folder name)."""
    sub_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{slugify(metadata['name'])}"
    files = {
        f"{sub_id}/metadata.json": json.dumps(metadata, indent=2, ensure_ascii=False).encode(),
        f"{sub_id}/scores.json": json.dumps(scores, indent=2, ensure_ascii=False).encode(),
    }
    for lang, records in per_language_records.items():
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        files[f"{sub_id}/predictions/{lang}.jsonl"] = lines.encode()

    cfg = github_config(secrets)
    if cfg:
        for rel, content in files.items():
            _gh_put_file(
                cfg,
                f"{SUBMISSIONS_REL_DIR}/{rel}",
                content,
                f"leaderboard: add submission {sub_id}",
            )
    else:
        for rel, content in files.items():
            path = os.path.join(LOCAL_SUBMISSIONS_DIR, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(content)
    return sub_id


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def _gh_get_json(cfg, url: str, params=None):
    resp = requests.get(url, headers=_headers(cfg), params=params or {}, timeout=30)
    if resp.status_code == 404:
        return None
    if not resp.ok:
        raise StorageError(f"GitHub read failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def _gh_read_file(cfg, path: str):
    data = _gh_get_json(
        cfg,
        f"{_API}/repos/{cfg['repo']}/contents/{path}",
        params={"ref": cfg["branch"]},
    )
    if data is None:
        return None
    return base64.b64decode(data["content"])


def load_submissions(secrets) -> list[dict]:
    """Return [{'id', 'metadata', 'scores'}, ...] for every stored submission."""
    cfg = github_config(secrets)
    entries = []
    if cfg:
        listing = _gh_get_json(
            cfg,
            f"{_API}/repos/{cfg['repo']}/contents/{SUBMISSIONS_REL_DIR}",
            params={"ref": cfg["branch"]},
        )
        if not listing:
            return []
        for item in listing:
            if item.get("type") != "dir":
                continue
            meta_raw = _gh_read_file(cfg, f"{SUBMISSIONS_REL_DIR}/{item['name']}/metadata.json")
            scores_raw = _gh_read_file(cfg, f"{SUBMISSIONS_REL_DIR}/{item['name']}/scores.json")
            if meta_raw is None or scores_raw is None:
                continue
            entries.append(
                {
                    "id": item["name"],
                    "metadata": json.loads(meta_raw),
                    "scores": json.loads(scores_raw),
                }
            )
    else:
        if not os.path.isdir(LOCAL_SUBMISSIONS_DIR):
            return []
        for name in sorted(os.listdir(LOCAL_SUBMISSIONS_DIR)):
            folder = os.path.join(LOCAL_SUBMISSIONS_DIR, name)
            meta_p = os.path.join(folder, "metadata.json")
            scores_p = os.path.join(folder, "scores.json")
            if not (os.path.isfile(meta_p) and os.path.isfile(scores_p)):
                continue
            with open(meta_p, encoding="utf-8") as fh:
                metadata = json.load(fh)
            with open(scores_p, encoding="utf-8") as fh:
                scores = json.load(fh)
            entries.append({"id": name, "metadata": metadata, "scores": scores})
    return entries

def _gh_list_files_recursive(cfg, path: str) -> list[dict]:
    """Return flat list of {'path', 'sha'} for every file under path (recursing into dirs)."""
    listing = _gh_get_json(
        cfg,
        f"{_API}/repos/{cfg['repo']}/contents/{path}",
        params={"ref": cfg["branch"]},
    )
    if not listing:
        return []
    files = []
    for item in listing:
        if item["type"] == "file":
            files.append({"path": item["path"], "sha": item["sha"]})
        elif item["type"] == "dir":
            files.extend(_gh_list_files_recursive(cfg, item["path"]))
    return files


def _gh_delete_file(cfg, path: str, sha: str, message: str) -> None:
    url = f"{_API}/repos/{cfg['repo']}/contents/{path}"
    payload = {"message": message, "sha": sha, "branch": cfg["branch"]}
    resp = requests.delete(url, headers=_headers(cfg), json=payload, timeout=30)
    if not resp.ok:
        raise StorageError(f"GitHub delete failed ({resp.status_code}): {resp.text[:300]}")


def delete_submission(secrets, submission_id: str) -> None:
    """Remove a submission (metadata, scores, and all prediction files)."""
    cfg = github_config(secrets)
    if cfg:
        rel_dir = f"{SUBMISSIONS_REL_DIR}/{submission_id}"
        files = _gh_list_files_recursive(cfg, rel_dir)
        if not files:
            raise StorageError(f"Submission {submission_id!r} not found on GitHub.")
        for f in files:
            _gh_delete_file(cfg, f["path"], f["sha"], f"leaderboard: remove submission {submission_id}")
    else:
        import shutil
        folder = os.path.join(LOCAL_SUBMISSIONS_DIR, submission_id)
        if not os.path.isdir(folder):
            raise StorageError(f"Submission {submission_id!r} not found locally.")
        shutil.rmtree(folder)

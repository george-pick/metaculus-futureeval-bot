"""Build local question fixtures from Wayback Machine snapshots.

The Metaculus API now requires an account token for ALL endpoints (including
listing). Until the bot account exists, we test the full pipeline on REAL,
currently-open Metaculus questions by extracting the post JSON embedded in
recently-archived question pages (Next.js flight data).

Usage:
    uv run python scripts/build_fixtures_from_wayback.py <wayback_url> [...]
    uv run python scripts/build_fixtures_from_wayback.py --default

Writes runs/fixtures/questions.json — consumable via `main.py --questions-file`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

BOT_DIR = Path(__file__).resolve().parent.parent
OUT = BOT_DIR / "runs" / "fixtures" / "questions.json"

DEFAULT_SNAPSHOTS = [
    # (snapshot ts, question page) — gathered from the CDX index, June 2026
    "https://web.archive.org/web/20260608071442/https://www.metaculus.com/questions/43841/will-openai-file-an-s-1-with-the-sec-to-launch-an-ipo-before-september-1-2026/",
    "https://web.archive.org/web/20260608071442/https://www.metaculus.com/questions/43525/will-atlas-browser-be-released-for-windows-before-september-1-2026/",
    "https://web.archive.org/web/20260608071442/https://www.metaculus.com/questions/43679/nvda-vs-msft-in-q2-2026/",
    "https://web.archive.org/web/20260608071442/https://www.metaculus.com/questions/43794/how-many-software-development-jobs-will-be-posted-on-upwork-on-july-1-2026/",
    "https://web.archive.org/web/20260608071442/https://www.metaculus.com/questions/43311/verdict-in-musk-v-altman-et-al/",
    "https://web.archive.org/web/20260608071442/https://www.metaculus.com/questions/43494/useu-restriction-on-data-center-grid-before-sep-2026/",
]

_FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,\s*"((?:[^"\\]|\\.)*)"\]\)')


def extract_post(html: str) -> dict | None:
    """Extract the embedded `post` object from Next.js flight data."""
    blob = "".join(json.loads('"' + c + '"') for c in _FLIGHT_RE.findall(html))
    dec = json.JSONDecoder()
    best: dict | None = None
    for m in re.finditer(r'"post":\s*\{', blob):
        start = m.end() - 1
        try:
            obj, _ = dec.raw_decode(blob[start:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("question"), dict)
            and "resolution_criteria" in obj["question"]
        ):
            if best is None or len(json.dumps(obj)) > len(json.dumps(best)):
                best = obj
    return best


def slim(post: dict) -> dict:
    """Keep only the fields the bot consumes (API-schema-compatible)."""
    q = post["question"]
    slim_q = {
        k: q.get(k)
        for k in (
            "id",
            "title",
            "type",
            "status",
            "description",
            "resolution_criteria",
            "fine_print",
            "unit",
            "options",
            "scaling",
            "open_lower_bound",
            "open_upper_bound",
            "scheduled_close_time",
            "scheduled_resolve_time",
        )
    }
    return {"id": post["id"], "title": post["title"], "question": slim_q}


def main() -> None:
    urls = sys.argv[1:]
    if not urls or urls == ["--default"]:
        urls = DEFAULT_SNAPSHOTS
    posts = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for url in urls:
            print(f"fetching {url}")
            r = client.get(url)
            if r.status_code != 200:
                print(f"  skip (HTTP {r.status_code})")
                continue
            post = extract_post(r.text)
            if not post:
                print("  skip (no embedded post JSON found)")
                continue
            q = post["question"]
            print(
                f"  -> post {post['id']} | {q.get('type')} | status={q.get('status')} | "
                f"{post['title'][:70]}"
            )
            if q.get("status") != "open":
                print("  skip (question not open)")
                continue
            posts.append(slim(post))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(posts, indent=2))
    print(f"\nWrote {len(posts)} open question fixture(s) to {OUT}")


if __name__ == "__main__":
    main()

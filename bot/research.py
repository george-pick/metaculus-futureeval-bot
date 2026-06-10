"""Research stage: gather news/evidence for a question.

Provider priority:
  1. AskNews (if ASKNEWS_CLIENT_ID / ASKNEWS_SECRET are set — tournament
     credits; implemented against their REST API so no SDK is needed)
  2. DuckDuckGo news + web search via `ddgs` (keyless)
  3. Google News RSS (keyless fallback)

Results are cached on disk per question so re-runs on unchanged questions
spend nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

from . import config

log = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    text: str
    providers: list[str] = field(default_factory=list)
    cached: bool = False


# --------------------------------------------------------------- providers
async def _asknews_search(query: str) -> str | None:
    """AskNews REST search (slot-in once tournament credits arrive)."""
    if not (config.ASKNEWS_CLIENT_ID and config.ASKNEWS_SECRET):
        return None
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            tok = await client.post(
                "https://auth.asknews.app/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": config.ASKNEWS_CLIENT_ID,
                    "client_secret": config.ASKNEWS_SECRET,
                    "scope": "news",
                },
            )
            tok.raise_for_status()
            access = tok.json()["access_token"]
            headers = {"Authorization": f"Bearer {access}"}
            chunks = []
            for strategy, n in (("latest news", 6), ("news knowledge", 10)):
                r = await client.get(
                    "https://api.asknews.app/v1/news/search",
                    params={
                        "query": query,
                        "n_articles": n,
                        "return_type": "string",
                        "strategy": strategy,
                    },
                    headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                as_string = data.get("as_string") or ""
                if as_string:
                    chunks.append(f"### AskNews ({strategy})\n{as_string}")
            return "\n\n".join(chunks) if chunks else None
    except Exception as exc:  # noqa: BLE001 — research is best-effort
        log.warning("AskNews provider failed: %s", exc)
        return None


def _ddg_search_sync(query: str, max_items: int) -> str | None:
    try:
        from ddgs import DDGS

        lines: list[str] = []
        with DDGS() as ddgs:
            try:
                news = list(ddgs.news(query, max_results=max_items))
            except Exception:
                news = []
            for item in news:
                lines.append(
                    f"- [{item.get('date', '?')}] {item.get('title', '')} "
                    f"({item.get('source', '?')}): {item.get('body', '')[:400]}"
                )
            if len(lines) < 3:
                try:
                    web = list(ddgs.text(query, max_results=max_items))
                except Exception:
                    web = []
                for item in web:
                    lines.append(
                        f"- {item.get('title', '')}: {item.get('body', '')[:400]}"
                    )
        if not lines:
            return None
        return "### DuckDuckGo search results\n" + "\n".join(lines[:max_items])
    except Exception as exc:  # noqa: BLE001
        log.warning("DuckDuckGo provider failed: %s", exc)
        return None


async def _ddg_search(query: str, max_items: int) -> str | None:
    return await asyncio.to_thread(_ddg_search_sync, query, max_items)


async def _google_news_rss(query: str, max_items: int) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                headers={"User-Agent": "Mozilla/5.0 (metaculus-bot research)"},
            )
            r.raise_for_status()
        root = ET.fromstring(r.text)
        items = root.findall(".//item")[:max_items]
        if not items:
            return None
        lines = []
        for it in items:
            title = (it.findtext("title") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            source = it.find("{https://news.google.com/rss}source")
            src = source.text if source is not None else ""
            lines.append(f"- [{pub}] {title} ({src})")
        return "### Google News headlines\n" + "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        log.warning("Google News RSS provider failed: %s", exc)
        return None


# ----------------------------------------------------------------- caching
def _cache_key(question_id: int, title: str, resolution_criteria: str) -> str:
    digest = hashlib.sha256(
        f"{title}\n{resolution_criteria}".encode()
    ).hexdigest()[:16]
    return f"{question_id}_{digest}"


def _cache_read(key: str) -> str | None:
    path = config.CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    age_h = (time.time() - data.get("ts", 0)) / 3600
    if age_h > config.RESEARCH_CACHE_TTL_HOURS:
        return None
    return data.get("text")


def _cache_write(key: str, text: str, providers: list[str]) -> None:
    path = config.CACHE_DIR / f"{key}.json"
    path.write_text(
        json.dumps({"ts": time.time(), "providers": providers, "text": text})
    )


# -------------------------------------------------------------------- main
async def gather_research(
    question_id: int, title: str, resolution_criteria: str = ""
) -> ResearchResult:
    key = _cache_key(question_id, title, resolution_criteria)
    cached = _cache_read(key)
    if cached is not None:
        return ResearchResult(text=cached, providers=["cache"], cached=True)

    providers_used: list[str] = []
    sections: list[str] = []

    asknews = await _asknews_search(title)
    if asknews:
        sections.append(asknews)
        providers_used.append("asknews")

    if not asknews:  # keyless fallbacks; run both for coverage
        ddg, gnews = await asyncio.gather(
            _ddg_search(title, config.MAX_NEWS_ITEMS),
            _google_news_rss(title, config.MAX_NEWS_ITEMS),
        )
        if ddg:
            sections.append(ddg)
            providers_used.append("ddgs")
        if gnews:
            sections.append(gnews)
            providers_used.append("google_news_rss")

    if not sections:
        text = "No research results were found. Rely on base rates and background knowledge."
    else:
        text = "\n\n".join(sections)

    _cache_write(key, text, providers_used)
    return ResearchResult(text=text, providers=providers_used, cached=False)

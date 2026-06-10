"""Thin client for the Metaculus API.

Endpoint shapes adapted from the official template:
https://github.com/Metaculus/metac-bot-template (main_with_no_framework.py)

Listing open questions requires no auth; submitting forecasts requires
METACULUS_TOKEN.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from . import config

log = logging.getLogger(__name__)

QUESTION_TYPES = ["binary", "multiple_choice", "numeric", "discrete"]


def _headers(require_token: bool = False) -> dict[str, str]:
    headers = {"User-Agent": "metaculus-futureeval-bot/0.1"}
    if config.METACULUS_TOKEN:
        headers["Authorization"] = f"Token {config.METACULUS_TOKEN}"
    elif require_token:
        raise RuntimeError(
            "METACULUS_TOKEN is not set. Submission requires a Metaculus bot "
            "account token (see README: 'Registering the bot account')."
        )
    return headers


def forecasting_access() -> str:
    """Return this account's api_forecasting_access state ('enabled',
    'pending', etc.), or 'no_token' if unauthenticated. Cheap single GET —
    used as a preflight so we don't spend LLM credits when we can't submit."""
    if not config.METACULUS_TOKEN:
        return "no_token"
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{config.METACULUS_API_BASE}/users/me/", headers=_headers(True)
            )
            resp.raise_for_status()
            me = resp.json()
            log.info(
                "Account '%s' (id=%s): api_forecasting_access=%s is_bot=%s",
                me.get("username"), me.get("id"),
                me.get("api_forecasting_access"), me.get("is_bot"),
            )
            return me.get("api_forecasting_access") or "unknown"
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not check forecasting access: %s", exc)
        return "unknown"


def list_open_questions(
    tournament_id: int | str = config.TOURNAMENT_ID,
    max_questions: int = 200,
) -> list[dict[str, Any]]:
    """Return open question posts in a tournament (paginated).

    Each returned item is a post dict containing a single `question` dict.
    """
    results: list[dict[str, Any]] = []
    offset = 0
    page = 100
    with httpx.Client(timeout=45) as client:
        while len(results) < max_questions:
            resp = client.get(
                f"{config.METACULUS_API_BASE}/posts/",
                params={
                    "limit": page,
                    "offset": offset,
                    "order_by": "-hotness",
                    "forecast_type": ",".join(QUESTION_TYPES),
                    "tournaments": [tournament_id],
                    "statuses": "open",
                    "include_description": "true",
                },
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("results", [])
            for post in batch:
                question = post.get("question")
                if question and question.get("status") == "open":
                    results.append(post)
            if not data.get("next") or not batch:
                break
            offset += page
    return results[:max_questions]


def get_post_detail(post_id: int) -> dict[str, Any]:
    with httpx.Client(timeout=45) as client:
        resp = client.get(
            f"{config.METACULUS_API_BASE}/posts/{post_id}/", headers=_headers()
        )
        resp.raise_for_status()
        return resp.json()


def already_forecasted(post: dict[str, Any]) -> bool:
    """True if this (token-authed) account already has a standing forecast."""
    try:
        values = post["question"]["my_forecasts"]["latest"]["forecast_values"]
        return values is not None
    except (KeyError, TypeError):
        return False


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict[str, Any]:
    """Build the per-question payload for POST /questions/forecast/."""
    if question_type == "binary":
        assert isinstance(forecast, float)
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        assert isinstance(forecast, dict)
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric / discrete: 201-point (or inbound_outcome_count+1) CDF
    assert isinstance(forecast, list)
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def submit_forecast(question_id: int, payload: dict[str, Any]) -> None:
    """POST a forecast. Requires METACULUS_TOKEN."""
    with httpx.Client(timeout=45) as client:
        resp = client.post(
            f"{config.METACULUS_API_BASE}/questions/forecast/",
            json=[{"question": question_id, "source": "api", **payload}],
            headers=_headers(require_token=True),
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Forecast submission failed ({resp.status_code}): {resp.text[:500]}"
            )
    log.info("Submitted forecast for question %s", question_id)


def post_comment(post_id: int, text: str, private: bool = True) -> None:
    """Attach the bot's reasoning as a (private) comment. Requires token."""
    with httpx.Client(timeout=45) as client:
        resp = client.post(
            f"{config.METACULUS_API_BASE}/comments/create/",
            json={
                "text": text,
                "parent": None,
                "included_forecast": True,
                "is_private": private,
                "on_post": post_id,
            },
            headers=_headers(require_token=True),
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Comment post failed ({resp.status_code}): {resp.text[:500]}"
            )

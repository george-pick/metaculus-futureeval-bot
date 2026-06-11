"""Reasoning stage: claude-opus-4-6 ensemble forecasts per question type."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

from . import aggregate, config, prompts
from .cdf import percentiles_to_cdf

log = logging.getLogger(__name__)

_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
_semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)

# Cumulative usage for cost reporting
USAGE = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
PRICE_PER_MTOK = {"input": 5.0, "output": 25.0}  # claude-opus-4-6


def estimated_cost_usd() -> float:
    return (
        USAGE["input_tokens"] / 1e6 * PRICE_PER_MTOK["input"]
        + USAGE["output_tokens"] / 1e6 * PRICE_PER_MTOK["output"]
    )


async def _call_claude(user_prompt: str) -> str:
    async with _semaphore:
        for attempt in range(4):
            try:
                msg = await _client.messages.create(
                    model=config.ANTHROPIC_MODEL,
                    max_tokens=config.ANTHROPIC_MAX_TOKENS,
                    system=prompts.SYSTEM_PROMPT,
                    thinking={"type": "adaptive"},
                    output_config={"effort": config.ANTHROPIC_EFFORT},
                    messages=[{"role": "user", "content": user_prompt}],
                )
                USAGE["input_tokens"] += msg.usage.input_tokens
                USAGE["output_tokens"] += msg.usage.output_tokens
                USAGE["calls"] += 1
                return "".join(
                    b.text for b in msg.content if b.type == "text"
                )
            except anthropic.APIConnectionError as exc:
                wait = 5 * (2**attempt)
                log.warning("Claude connection error (%s); retry in %ss", exc, wait)
                await asyncio.sleep(wait)
            except anthropic.APIStatusError as exc:
                # Retry transient server/rate statuses (429, 500, 502, 503, 529
                # "overloaded"); re-raise client errors (400/401/404) immediately.
                if exc.status_code not in (429, 500, 502, 503, 529):
                    raise
                wait = 5 * (2**attempt)
                log.warning(
                    "Claude transient %s; retry in %ss", exc.status_code, wait
                )
                await asyncio.sleep(wait)
        raise RuntimeError("Claude call failed after retries")


# ------------------------------------------------------------------ parsing
def parse_binary(text: str) -> float:
    """Extract 'Probability: ZZ%' (percent) -> decimal."""
    matches = re.findall(r"[Pp]robability:?\s*([\d.]+)\s*%", text)
    if not matches:
        matches = re.findall(r"([\d.]+)\s*%", text)
    if not matches:
        raise ValueError(f"No probability found in response tail: {text[-300:]!r}")
    value = float(matches[-1])
    return min(max(value, 0.5), 99.5) / 100.0


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float]:
    """Extract per-option probabilities from the final lines."""
    probs: dict[str, float] = {}
    lines = text.strip().splitlines()
    for opt in options:
        pattern = re.compile(
            re.escape(opt.strip()) + r"\s*:?\s*([\d.]+)\s*%?\s*$", re.IGNORECASE
        )
        for line in reversed(lines):
            m = pattern.search(line.strip().strip("*").strip())
            if m:
                probs[opt] = float(m.group(1))
                break
    if len(probs) != len(options):
        missing = [o for o in options if o not in probs]
        raise ValueError(f"Missing probabilities for options {missing}")
    # Accept either percents or decimals; normalize to sum 1.
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Option probabilities sum to zero")
    return {k: v / total for k, v in probs.items()}


def parse_percentiles(text: str) -> dict[float, float]:
    """Extract percentile lines -> {percentile(0-100): value}.

    Tolerant of the several formats the model drifts into:
      'Percentile 10: 1,234'  ·  'P10: 1234'  ·  '10th percentile: 1234'
      '10%: 1234'             ·  '- Percentile 90 -> $1.2M'
    Values are coerced to a strictly increasing sequence (ordered by
    percentile) so a minor monotonicity glitch in one sample doesn't
    discard an otherwise-good forecast.
    """
    val_re = r"\$?\s*(-?[\d,]+(?:\.\d+)?)\s*([KkMmBb])?"
    patterns = [
        rf"[Pp]ercentile\s+([\d.]+)\s*\**\s*[:\-=>]+\s*\**\s*{val_re}",
        rf"\bP\s*([\d.]+)\s*[:\-=>]+\s*{val_re}",
        rf"([\d.]+)\s*(?:th|st|nd|rd)\s+percentile\s*[:\-=>]+\s*{val_re}",
        rf"([\d.]+)\s*%\s*[:\-=>]+\s*{val_re}",
    ]
    mult = {"k": 1e3, "m": 1e6, "b": 1e9}
    out: dict[float, float] = {}
    for pat in patterns:
        for m in re.finditer(pat, text):
            pct = float(m.group(1))
            if not 0 <= pct <= 100:
                continue
            val = float(m.group(2).replace(",", ""))
            suffix = m.group(3)
            if suffix:
                val *= mult[suffix.lower()]
            out.setdefault(pct, val)  # first (primary) format wins per pct
        if len(out) >= 2:
            break
    if len(out) < 2:
        raise ValueError(f"Could not parse percentiles from tail: {text[-400:]!r}")
    return _enforce_increasing_values(out)


def _enforce_increasing_values(pcts: dict[float, float]) -> dict[float, float]:
    """Force values strictly increasing in percentile order via tiny nudges."""
    items = sorted(pcts.items())
    span = abs(items[-1][1] - items[0][1]) or abs(items[-1][1]) or 1.0
    eps = span * 1e-6
    fixed: dict[float, float] = {}
    prev: float | None = None
    for pct, val in items:
        if prev is not None and val <= prev:
            val = prev + eps
        fixed[pct] = val
        prev = val
    return fixed


# --------------------------------------------------------------- forecasting
@dataclass
class ForecastOutput:
    forecast: Any  # float | dict[str,float] | list[float]
    samples: list[Any]
    rationales: list[str]
    n_failed: int = 0
    summary: str = ""
    extra: dict = field(default_factory=dict)


def _question_fields(q: dict[str, Any]) -> dict[str, str]:
    return {
        "today": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "title": q.get("title", ""),
        "background": q.get("description") or "(none provided)",
        "resolution_criteria": q.get("resolution_criteria") or "(none provided)",
        "fine_print": q.get("fine_print") or "(none)",
        "close_time": q.get("scheduled_close_time") or "unknown",
        "resolve_time": q.get("scheduled_resolve_time") or "unknown",
    }


async def _run_ensemble(prompt: str, parse_fn, n: int) -> tuple[list[Any], list[str], int]:
    async def one() -> tuple[Any, str] | None:
        try:
            text = await _call_claude(prompt)
            return parse_fn(text), text
        except Exception as exc:  # noqa: BLE001
            log.warning("Ensemble sample failed: %s", exc)
            return None

    results = await asyncio.gather(*[one() for _ in range(n)])
    ok = [r for r in results if r is not None]
    if len(ok) < max(2, n // 2):
        raise RuntimeError(f"Too few successful ensemble samples ({len(ok)}/{n})")
    samples = [r[0] for r in ok]
    rationales = [r[1] for r in ok]
    return samples, rationales, n - len(ok)


async def forecast_binary(q: dict[str, Any], research: str) -> ForecastOutput:
    prompt = prompts.BINARY_PROMPT.format(**_question_fields(q), research=research)
    samples, rationales, n_failed = await _run_ensemble(
        prompt, parse_binary, config.ENSEMBLE_SIZE
    )
    p = aggregate.aggregate_binary(samples)
    return ForecastOutput(
        forecast=p,
        samples=samples,
        rationales=rationales,
        n_failed=n_failed,
        summary=f"P(yes)={p:.3f} from samples {[round(s, 3) for s in samples]}",
    )


async def forecast_multiple_choice(q: dict[str, Any], research: str) -> ForecastOutput:
    options: list[str] = q["options"]
    option_format_lines = "\n".join(f"{o}: XX%" for o in options)
    prompt = prompts.MULTIPLE_CHOICE_PROMPT.format(
        **_question_fields(q),
        research=research,
        options=options,
        option_format_lines=option_format_lines,
    )
    samples, rationales, n_failed = await _run_ensemble(
        prompt, lambda t: parse_multiple_choice(t, options), config.ENSEMBLE_SIZE
    )
    dist = aggregate.aggregate_multiple_choice(samples, options)
    return ForecastOutput(
        forecast=dist,
        samples=samples,
        rationales=rationales,
        n_failed=n_failed,
        summary="; ".join(f"{k}={v:.3f}" for k, v in dist.items()),
    )


async def forecast_numeric(q: dict[str, Any], research: str) -> ForecastOutput:
    scaling = q["scaling"]
    open_upper = q["open_upper_bound"]
    open_lower = q["open_lower_bound"]
    upper = scaling["range_max"]
    lower = scaling["range_min"]
    zero_point = scaling.get("zero_point")
    if q["type"] == "discrete":
        cdf_size = scaling["inbound_outcome_count"] + 1
    else:
        cdf_size = 201
    lower_msg, upper_msg = prompts.bound_messages(open_lower, open_upper, lower, upper)
    units = q.get("unit") or "not stated (infer from the question)"
    prompt = prompts.NUMERIC_PROMPT.format(
        **_question_fields(q),
        research=research,
        kind=q["type"],
        units=units,
        lower_bound_message=lower_msg,
        upper_bound_message=upper_msg,
    )

    def to_cdf(text: str) -> list[float]:
        percentiles = parse_percentiles(text)
        return percentiles_to_cdf(
            percentiles,
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
            zero_point=zero_point,
            cdf_size=cdf_size,
        )

    samples, rationales, n_failed = await _run_ensemble(
        prompt, to_cdf, config.ENSEMBLE_SIZE
    )
    cdf = aggregate.aggregate_cdfs(samples)
    return ForecastOutput(
        forecast=cdf,
        samples=samples,
        rationales=rationales,
        n_failed=n_failed,
        summary=f"CDF[{cdf_size}] head={', '.join(f'{v:.4f}' for v in cdf[:3])} "
        f"tail={', '.join(f'{v:.4f}' for v in cdf[-3:])}",
        extra={"declared_percentiles_per_sample": "see rationales"},
    )


async def forecast_question(q: dict[str, Any], research: str) -> ForecastOutput:
    qtype = q["type"]
    if qtype == "binary":
        return await forecast_binary(q, research)
    if qtype == "multiple_choice":
        return await forecast_multiple_choice(q, research)
    if qtype in ("numeric", "discrete"):
        return await forecast_numeric(q, research)
    raise ValueError(f"Unsupported question type: {qtype}")

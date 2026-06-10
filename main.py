"""Metaculus FutureEval forecasting bot — entry point.

Default mode is DRY RUN: forecasts every open question, writes JSONL logs,
submits nothing. Pass --submit (requires METACULUS_TOKEN) to publish.

Examples:
    uv run python main.py --limit 4                       # dry-run, summer tournament
    uv run python main.py --tournament bot-testing-area   # dry-run on test area
    uv run python main.py --submit                        # real submission run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from bot import config, forecaster, metaculus_api, research, storage

log = logging.getLogger("main")


async def process_question(
    post: dict[str, Any], submit: bool, comment: bool
) -> dict[str, Any]:
    q = post["question"]
    qid, post_id = q["id"], post["id"]
    title = q["title"]
    url = f"https://www.metaculus.com/questions/{post_id}/"

    res = await research.gather_research(
        qid, title, q.get("resolution_criteria") or ""
    )
    output = await forecaster.forecast_question(q, res.text)

    payload = metaculus_api.create_forecast_payload(output.forecast, q["type"])
    record = {
        "question_id": qid,
        "post_id": post_id,
        "url": url,
        "title": title,
        "type": q["type"],
        "model": config.ANTHROPIC_MODEL,
        "ensemble_size": config.ENSEMBLE_SIZE,
        "research_providers": res.providers,
        "research_cached": res.cached,
        "research_text": res.text,
        "samples": output.samples,
        "rationales": output.rationales,
        "n_failed_samples": output.n_failed,
        "forecast": output.forecast,
        "summary": output.summary,
        "submitted": False,
    }

    if submit:
        metaculus_api.submit_forecast(qid, payload)
        record["submitted"] = True
        if comment:
            try:
                best_rationale = max(output.rationales, key=len)
                metaculus_api.post_comment(
                    post_id,
                    f"Ensemble of {len(output.samples)} {config.ANTHROPIC_MODEL} "
                    f"samples. {output.summary}\n\n---\n{best_rationale[:6000]}",
                )
            except Exception as exc:  # noqa: BLE001 — comments are best-effort
                log.warning("Comment failed for post %s: %s", post_id, exc)

    log_path = storage.log_forecast(record)
    mode = "SUBMITTED" if record["submitted"] else "dry-run"
    print(f"[{mode}] {title}\n  -> {output.summary}\n  log: {log_path}\n")
    return record


async def run(args: argparse.Namespace) -> int:
    # Preflight: don't spend LLM credits forecasting if we can't submit.
    if args.submit:
        access = metaculus_api.forecasting_access()
        if access != "enabled":
            print(
                f"ERROR: account api_forecasting_access='{access}' (need 'enabled').\n"
                "Enable it at https://www.metaculus.com/accounts/settings/account/"
                "#api-forecasting-access and confirm this is a bot/automated account.\n"
                "Aborting before spending API credits.",
                file=sys.stderr,
            )
            return 3

    if args.questions_file:
        import json

        with open(args.questions_file) as f:
            posts = json.load(f)
        print(f"Loaded {len(posts)} question(s) from {args.questions_file}.")
        if args.submit:
            print("Refusing to --submit from a local questions file.", file=sys.stderr)
            return 2
    else:
        posts = metaculus_api.list_open_questions(
            args.tournament, max_questions=args.limit or 500
        )
        print(f"Found {len(posts)} open question(s) in tournament '{args.tournament}'.")

    if args.skip_forecasted and config.METACULUS_TOKEN:
        before = len(posts)
        posts = [p for p in posts if not metaculus_api.already_forecasted(p)]
        print(f"Skipping {before - len(posts)} already-forecasted question(s).")

    if args.limit:
        posts = posts[: args.limit]
    if not posts:
        print("Nothing to do.")
        return 0

    q_sem = asyncio.Semaphore(args.question_concurrency)

    async def guarded(post: dict[str, Any]):
        async with q_sem:
            return await process_question(post, args.submit, args.comment)

    results = await asyncio.gather(*[guarded(p) for p in posts], return_exceptions=True)

    n_err = 0
    for post, r in zip(posts, results):
        if isinstance(r, Exception):
            n_err += 1
            log.error(
                "FAILED %s (%s): %r", post["question"]["title"], post["id"], r
            )

    print(
        f"Done: {len(posts) - n_err} ok, {n_err} failed. "
        f"LLM usage: {forecaster.USAGE['calls']} calls, "
        f"{forecaster.USAGE['input_tokens']} in / "
        f"{forecaster.USAGE['output_tokens']} out tokens, "
        f"~${forecaster.estimated_cost_usd():.2f}"
    )
    return 1 if n_err else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tournament",
        default=config.TOURNAMENT_ID,
        help="Tournament ID or slug (default: %(default)s). "
        "Use 'bot-testing-area' or 'minibench' for testing.",
    )
    p.add_argument("--limit", type=int, default=None, help="Max questions this run")
    p.add_argument(
        "--submit",
        action="store_true",
        help="Publish forecasts to Metaculus (requires METACULUS_TOKEN). "
        "Default is dry-run.",
    )
    p.add_argument(
        "--comment",
        action="store_true",
        help="Also post reasoning as a private comment when submitting.",
    )
    p.add_argument(
        "--skip-forecasted",
        action="store_true",
        default=True,
        help="Skip questions this account already forecasted (token required).",
    )
    p.add_argument(
        "--no-skip-forecasted",
        dest="skip_forecasted",
        action="store_false",
        help="Re-forecast even if a forecast already stands (use to refresh).",
    )
    p.add_argument("--question-concurrency", type=int, default=3)
    p.add_argument(
        "--questions-file",
        default=None,
        help="Forecast questions from a local JSON fixture instead of the API "
        "(dry-run only; see scripts/build_fixtures_from_wayback.py).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    if args.submit and not config.METACULUS_TOKEN:
        print(
            "ERROR: --submit requires METACULUS_TOKEN. The bot account does not "
            "exist yet — see README 'Registering the bot account'. "
            "Run without --submit for a dry run.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not config.ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not found in environment/.env", file=sys.stderr)
        sys.exit(2)

    mode = "SUBMIT" if args.submit else "DRY RUN (nothing will be posted)"
    print(f"Mode: {mode} | model={config.ANTHROPIC_MODEL} | "
          f"ensemble N={config.ENSEMBLE_SIZE}\n")
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()

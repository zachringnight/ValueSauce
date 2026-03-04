#!/usr/bin/env python3
"""Unified data fetch — runs all ingestion clients once, populates shared data/.

All engines then read from the same cached data instead of making
their own API calls. Run this before any engine pipeline.

Usage:
    python fetch_all.py --season 2025-26 --date 2026-03-04
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from config.settings import get_settings
from ingestion import (
    NBAStatsClient,
    OddsAPIClient,
    SGOClient,
    SportradarClient,
    SQLiteCache,
    TheRundownClient,
)
from ingestion.injury_client import InjuryClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def save_json(data: object, path: Path) -> None:
    """Write JSON to data/ directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, default=str, indent=2)
    logger.info("Saved %s", path)


def fetch_nba_stats(
    client: NBAStatsClient,
    cache: SQLiteCache,
    season: str,
    target_date: str,
) -> dict:
    """Fetch player/team game logs and tracking data from stats.nba.com."""
    results = {"player_games": 0, "team_games": 0, "tracking": 0, "opponent": 0}

    # Player game logs
    cache_key = f"nba_stats_player_logs_{season}_{target_date}"
    hit = cache.get("player_game_logs", {"season": season, "date": target_date})
    if hit:
        logger.info("Player game logs cache hit, skipping API call.")
    else:
        logs = client.get_player_game_logs(
            season=season,
            date_from=target_date.replace("-", "/"),
            date_to=target_date.replace("-", "/"),
        )
        cache.set("player_game_logs", {"season": season, "date": target_date}, logs)
        save_json(logs, Path(f"data/player_logs/{target_date}.json"))
        results["player_games"] = len(logs)

    # Team game logs
    hit = cache.get("team_game_logs", {"season": season})
    if not hit:
        team_logs = client.get_team_game_logs(season=season)
        cache.set("team_game_logs", {"season": season}, team_logs)
        save_json(team_logs, Path(f"data/games/{target_date}_teams.json"))
        results["team_games"] = len(team_logs)

    # Tracking data
    hit = cache.get("tracking_catch_shoot", {"season": season})
    if not hit:
        try:
            cs = client.get_tracking_catch_and_shoot(season)
            pu = client.get_tracking_pull_up(season)
            touches = client.get_tracking_touches(season)
            cache.set("tracking_catch_shoot", {"season": season}, cs)
            cache.set("tracking_pull_up", {"season": season}, pu)
            cache.set("tracking_touches", {"season": season}, touches)
            save_json({"catch_shoot": cs, "pull_up": pu, "touches": touches},
                      Path(f"data/tracking/{target_date}.json"))
            results["tracking"] = len(cs) + len(pu) + len(touches)
        except Exception as exc:
            logger.warning("Tracking data not available: %s", exc)

    # Opponent shooting
    hit = cache.get("opponent_shooting", {"season": season})
    if not hit:
        opp = client.get_opponent_shooting(season)
        cache.set("opponent_shooting", {"season": season}, opp)
        save_json(opp, Path(f"data/tracking/{target_date}_opponent.json"))
        results["opponent"] = len(opp)

    return results


def fetch_odds(
    settings,
    cache: SQLiteCache,
    target_date: str,
) -> dict:
    """Fetch odds from all configured odds providers."""
    results = {"sgo": 0, "odds_api": 0, "therundown": 0}

    # SportsGameOdds
    if settings.sportsgameodds_api_key:
        hit = cache.get("sgo_events", {"date": target_date})
        if not hit:
            sgo = SGOClient(api_key=settings.sportsgameodds_api_key)
            if sgo.authenticate():
                events = sgo.fetch_nba_events()
                cache.set("sgo_events", {"date": target_date}, events)
                save_json(events, Path(f"data/odds/sgo_{target_date}.json"))
                results["sgo"] = len(events)
                cache.record_endpoint_call("sgo_events", ok=True)
            else:
                cache.record_endpoint_call("sgo_events", ok=False, error="Auth failed")
        else:
            logger.info("SGO cache hit, skipping.")
    else:
        logger.info("SGO API key not configured, skipping.")

    # The Odds API
    if settings.the_odds_api_key:
        hit = cache.get("odds_api_events", {"date": target_date})
        if not hit:
            odds_client = OddsAPIClient(api_key=settings.the_odds_api_key)
            events = odds_client.get_upcoming_events()
            today_events = [
                e for e in events
                if e.get("commence_time", "").startswith(target_date)
            ]
            all_props = []
            for event in today_events:
                try:
                    props = odds_client.get_player_prop_snapshots(event["id"])
                    all_props.extend(props)
                except Exception as exc:
                    logger.warning("Odds API fetch failed for %s: %s", event.get("id"), exc)
            cache.set("odds_api_events", {"date": target_date}, all_props)
            save_json(all_props, Path(f"data/odds/theoddsapi_{target_date}.json"))
            results["odds_api"] = len(all_props)
            cache.record_endpoint_call("odds_api_events", ok=True)
        else:
            logger.info("Odds API cache hit, skipping.")
    else:
        logger.info("The Odds API key not configured, skipping.")

    # TheRundown
    if settings.therundown_api_key:
        hit = cache.get("therundown_events", {"date": target_date})
        if not hit:
            rundown = TheRundownClient(api_key=settings.therundown_api_key)
            events = rundown.get_nba_events_with_props(date=target_date)
            cache.set("therundown_events", {"date": target_date}, events)
            save_json(events, Path(f"data/odds/therundown_{target_date}.json"))
            results["therundown"] = len(events)
            cache.record_endpoint_call("therundown_events", ok=True)
        else:
            logger.info("TheRundown cache hit, skipping.")
    else:
        logger.info("TheRundown API key not configured, skipping.")

    return results


def fetch_injuries(
    cache: SQLiteCache,
    target_date: str,
) -> dict:
    """Fetch injury reports from all sources."""
    hit = cache.get("injuries", {"date": target_date})
    if hit:
        logger.info("Injury cache hit, skipping.")
        return {"injuries": 0}

    client = InjuryClient()
    reports = client.fetch_injuries(mode="auto")
    cache.set("injuries", {"date": target_date}, reports)
    save_json(reports, Path(f"data/injuries/{target_date}.json"))
    cache.record_endpoint_call("injuries", ok=True)
    return {"injuries": len(reports)}


def fetch_sportradar(
    settings,
    cache: SQLiteCache,
    target_date: str,
) -> dict:
    """Fetch schedule from Sportradar if configured."""
    if not settings.sportradar_api_key:
        logger.info("Sportradar API key not configured, skipping.")
        return {"sportradar": 0}

    hit = cache.get("sportradar_schedule", {"date": target_date})
    if hit:
        logger.info("Sportradar cache hit, skipping.")
        return {"sportradar": 0}

    client = SportradarClient(api_key=settings.sportradar_api_key)
    games = client.get_daily_schedule(target_date)
    cache.set("sportradar_schedule", {"date": target_date}, games)
    save_json(games, Path(f"data/games/sportradar_{target_date}.json"))
    cache.record_endpoint_call("sportradar_schedule", ok=True)
    return {"sportradar": len(games)}


def main():
    parser = argparse.ArgumentParser(description="Unified data fetch for all engines")
    parser.add_argument("--season", default="2025-26", help="NBA season (e.g., 2025-26)")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()
    settings = get_settings()
    cache = SQLiteCache(path=settings.cache_db_path)

    logger.info("=== Unified fetch for %s (season %s) ===", target_date, args.season)

    nba_client = NBAStatsClient()

    # Run all fetches
    stats_results = fetch_nba_stats(nba_client, cache, args.season, target_date)
    odds_results = fetch_odds(settings, cache, target_date)
    injury_results = fetch_injuries(cache, target_date)
    sr_results = fetch_sportradar(settings, cache, target_date)

    # Summary
    all_results = {**stats_results, **odds_results, **injury_results, **sr_results}
    logger.info("=== Fetch complete ===")
    for k, v in all_results.items():
        logger.info("  %s: %d", k, v)


if __name__ == "__main__":
    main()

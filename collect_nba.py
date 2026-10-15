#!/usr/bin/env python3
"""
NBA Betting Data Collector
==========================
Idempotent pipeline run by GitHub Actions:

  postgame  Refresh the schedule, store box scores for finished games, rebuild
            the pre-game cumulative stats (including rows for today's games).
  morning   Everything postgame does, plus a morning odds + Kalshi snapshot.
  closing   Everything postgame does, plus a near-tipoff odds + Kalshi snapshot.
  validate  One-off check that NBA CDN box score fields match the source the
            historical CSV was built from.

Re-running a mode on the same day is safe: snapshots already taken are skipped.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from nba_pipeline import features, kalshi, merge, nba_cdn, odds
from nba_pipeline.config import (EASTERN, GAMES_FILE, HISTORICAL_FILE, KALSHI_FILE, MERGED_FILE,
                                 ODDS_FILE, TEAM_LOGS_FILE)

LOG = logging.getLogger("nba_collector")


# ─── CSV storage ──────────────────────────────────────────────────────────
def read_csv(path, columns=None) -> pd.DataFrame:
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path, dtype={"game_id": str})
    return pd.DataFrame(columns=columns or [])


def write_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def append_csv(df: pd.DataFrame, path, columns) -> None:
    if df.empty:
        return
    existing = read_csv(path, columns)
    frames = [f for f in (existing, df) if not f.empty]
    write_csv(pd.concat(frames, ignore_index=True)[columns], path)


# ─── Steps ────────────────────────────────────────────────────────────────
def refresh_games(season: str, schedule: list[dict], logs: pd.DataFrame) -> pd.DataFrame:
    """Replace this season's schedule rows; box scores are authoritative for finals."""
    games = pd.DataFrame(schedule, columns=nba_cdn.GAME_COLUMNS)
    finals = logs[logs["is_home"].astype(str).str.lower() == "true"].set_index("game_id")
    done = games["game_id"].isin(finals.index)
    games.loc[done, "game_status"] = 3
    games.loc[done, "home_points"] = games.loc[done, "game_id"].map(finals["pts"]).astype(int).values
    games.loc[done, "away_points"] = games.loc[done, "game_id"].map(finals["opp_pts"]).astype(int).values

    existing = read_csv(GAMES_FILE, nba_cdn.GAME_COLUMNS)
    other = existing[existing["season"] != season] if not existing.empty else existing
    frames = [f for f in (other, games) if not f.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else games
    combined = combined.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    for col in ("game_status", "home_points", "away_points"):
        combined[col] = pd.to_numeric(combined[col]).astype("Int64")
    write_csv(combined, GAMES_FILE)
    return combined


def ingest_boxscores(schedule: list[dict], season: str, today: str) -> tuple[pd.DataFrame, int]:
    """Fetch box scores for finished games not yet in the team game logs."""
    logs = read_csv(TEAM_LOGS_FILE, nba_cdn.LOG_COLUMNS)
    have = set(logs["game_id"]) if not logs.empty else set()
    todo = [g for g in schedule
            if g["game_id"] not in have
            and (g["game_date"] < today or (g["game_date"] == today and g["game_status"] == 3))
            and not (g["postponed"] and g["game_status"] != 3)]
    new_rows, pending = [], 0
    for g in todo:
        box = nba_cdn.fetch_boxscore(g["game_id"])
        rows = nba_cdn.parse_boxscore(box, season) if box else None
        if rows is None:
            pending += 1
            LOG.warning("Box score not available yet for %s (%s %s@%s)", g["game_id"], g["game_date"],
                        g["away_team_abbreviation"], g["home_team_abbreviation"])
            continue
        new_rows.extend(rows)
    if new_rows:
        append_csv(pd.DataFrame(new_rows, columns=nba_cdn.LOG_COLUMNS), TEAM_LOGS_FILE, nba_cdn.LOG_COLUMNS)
        logs = read_csv(TEAM_LOGS_FILE, nba_cdn.LOG_COLUMNS)
    LOG.info("Box scores: %d new games stored, %d still pending", len(new_rows) // 2, pending)
    return logs, len(new_rows) // 2


def rebuild_history(games: pd.DataFrame, logs: pd.DataFrame, season: str, today: str) -> pd.DataFrame:
    season_games = games[games["season"] == season]
    season_logs = logs[logs["season"] == season] if not logs.empty else logs
    history = read_csv(HISTORICAL_FILE, features.COLUMNS)
    if "home_OREB/G" in history:
        # Seasons from before this pipeline have stats but no game logs to rebuild from.
        old = history[(history["game_date"].map(features.season_from_date) == season)
                      & (history["home_games"] > 0) & history["home_OREB/G"].isna()]
        if not old.empty:
            raise RuntimeError(f"{season} rows predate this pipeline's game logs; refusing to overwrite them")
    rebuilt = features.build_rows(season_games, season_logs, today)
    combined = features.merge_into_history(history, rebuilt)
    write_csv(combined, HISTORICAL_FILE)
    LOG.info("Historical stats: %d rows (%d for %s)", len(combined), len(rebuilt), season)
    return combined


def already_snapped(path, columns, label: str, game_date: str) -> bool:
    df = read_csv(path, columns)
    return not df.empty and ((df["snapshot_label"] == label) & (df["game_date"] == game_date)).any()


def take_snapshots(label: str, games: pd.DataFrame, today: str, now: datetime, force: bool) -> dict:
    result = {"odds_rows": 0, "kalshi_rows": 0, "odds_error": None, "kalshi_error": None, "credits_left": None}
    upcoming = games[(games["game_date"] == today) & (games["game_status"] == 1)]
    upcoming = upcoming[pd.to_datetime(upcoming["tipoff_utc"], utc=True) > pd.Timestamp(now)]
    result["upcoming"] = len(upcoming)
    if upcoming.empty:
        LOG.info("No games left to tip off today; skipping %s snapshots", label)
        return result

    if not force and already_snapped(ODDS_FILE, odds.COLUMNS, label, today):
        LOG.info("Odds '%s' snapshot for %s already taken; skipping", label, today)
        result["odds_skipped"] = True
    else:
        try:
            df, quota = odds.fetch_snapshot(label, today, games, now)
            append_csv(df, ODDS_FILE, odds.COLUMNS)
            result["odds_rows"] = len(df)
            result["credits_left"] = quota.get("remaining")
        except Exception as e:  # keep going so the stats and Kalshi still get saved
            LOG.error("Odds snapshot failed: %s", e)
            result["odds_error"] = str(e)

    if not force and already_snapped(KALSHI_FILE, kalshi.COLUMNS, label, today):
        LOG.info("Kalshi '%s' snapshot for %s already taken; skipping", label, today)
    else:
        try:
            df = kalshi.fetch_snapshot(label, today, games, now)
            append_csv(df, KALSHI_FILE, kalshi.COLUMNS)
            result["kalshi_rows"] = len(df)
        except Exception as e:
            LOG.error("Kalshi snapshot failed: %s", e)
            result["kalshi_error"] = str(e)
    return result


def rebuild_merged(history: pd.DataFrame, games: pd.DataFrame) -> int:
    seasons = set(games["season"].dropna())
    merged = merge.build_merged(history, read_csv(ODDS_FILE, odds.COLUMNS),
                                read_csv(KALSHI_FILE, kalshi.COLUMNS), seasons)
    write_csv(merged, MERGED_FILE)
    return len(merged)


# ─── Validation ───────────────────────────────────────────────────────────
VALIDATE_CANDIDATES = {
    "reb": {"reboundsTotal": ["reboundsTotal"], "reboundsPersonal": ["reboundsPersonal"]},
    "dreb": {"reboundsDefensive": ["reboundsDefensive"],
             "reboundsDefensive+reboundsTeamDefensive": ["reboundsDefensive", "reboundsTeamDefensive"]},
    "tov": {"turnoversTotal": ["turnoversTotal"], "turnovers": ["turnovers"],
            "turnovers+turnoversTeam": ["turnovers", "turnoversTeam"]},
    "pf": {"foulsPersonal": ["foulsPersonal"], "foulsTeam": ["foulsTeam"]},
    "blk": {"blocks": ["blocks"]}, "stl": {"steals": ["steals"]}, "ast": {"assists": ["assists"]},
    "fgm": {"fieldGoalsMade": ["fieldGoalsMade"]}, "fga": {"fieldGoalsAttempted": ["fieldGoalsAttempted"]},
    "fg3m": {"threePointersMade": ["threePointersMade"]},
    "fg3a": {"threePointersAttempted": ["threePointersAttempted"]},
    "ftm": {"freeThrowsMade": ["freeThrowsMade"]}, "fta": {"freeThrowsAttempted": ["freeThrowsAttempted"]},
}


def run_validate(sample: int) -> bool:
    """Compare CDN box scores with per-game stats recovered from the historical CSV."""
    from nba_pipeline.legacy import recover_game_logs

    history = read_csv(HISTORICAL_FILE, features.COLUMNS)
    if "season" in history:
        legacy = history[history["home_REB/G"].notna() & history["home_OREB/G"].isna()]
    else:
        legacy = history
    recovered = recover_game_logs(legacy).dropna(subset=["reb"])
    ids = sorted(recovered["game_id"].unique())
    step = max(len(ids) // sample, 1)
    picks = ids[::step][:sample]

    counts = {stat: {name: 0 for name in cands} for stat, cands in VALIDATE_CANDIDATES.items()}
    parser_hits = {stat: 0 for stat in VALIDATE_CANDIDATES}
    compared, misses = 0, 0
    for gid in picks:
        box = nba_cdn.fetch_boxscore(gid)
        if not box:
            LOG.warning("Validate: no box score for %s", gid)
            misses += 1
            if misses >= 3 and compared == 0:
                LOG.error("Validate: first %d box scores all failed; stopping (run mode 'probe' to diagnose)", misses)
                break
            continue
        parsed = {r["team"]: r for r in (nba_cdn.parse_boxscore(box, "validate") or [])}
        for side in ("homeTeam", "awayTeam"):
            team = box[side]
            abbr = nba_cdn.to_abbr(team.get("teamTricode"))
            want = recovered[(recovered["game_id"] == gid) & (recovered["team"] == abbr)]
            if want.empty:
                continue
            want = want.iloc[0]
            stats = team.get("statistics") or {}
            compared += 1
            for stat, cands in VALIDATE_CANDIDATES.items():
                for name, keys in cands.items():
                    if all(stats.get(k) is not None for k in keys):
                        if abs(sum(float(stats[k]) for k in keys) - want[stat]) < 1e-6:
                            counts[stat][name] += 1
                if abbr in parsed and parsed[abbr][stat] is not None and abs(parsed[abbr][stat] - want[stat]) < 1e-6:
                    parser_hits[stat] += 1

    lines = [f"## NBA CDN validation: {compared} team-games compared", "",
             "| Stat | Pipeline match | Candidate field matches |", "|---|---|---|"]
    ok = compared > 0
    for stat, cands in VALIDATE_CANDIDATES.items():
        rate = parser_hits[stat] / compared if compared else 0
        ok = ok and rate >= 0.95
        detail = ", ".join(f"`{n}` {c}/{compared}" for n, c in counts[stat].items())
        lines.append(f"| {stat} | {parser_hits[stat]}/{compared} ({rate:.0%}) | {detail} |")
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    return ok


# ─── Probe ────────────────────────────────────────────────────────────────
PROBE_URLS = {
    "NBA schedule": nba_cdn.SCHEDULE_URL,
    "NBA today's scoreboard": "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
    "NBA box score 2025-26": nba_cdn.BOXSCORE_URL.format(game_id="0022500001"),
    "NBA box score 2022-23": nba_cdn.BOXSCORE_URL.format(game_id="0022200720"),
    "NBA box score 2026-27 preseason": nba_cdn.BOXSCORE_URL.format(game_id="0012600001"),
    "ESPN scoreboard": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20260211",
    "Kalshi markets": f"{kalshi.API}/markets?series_ticker={kalshi.SERIES}&limit=1",
}


def run_probe() -> None:
    """Report which data sources answer from this machine, and with which headers."""
    import requests
    from nba_pipeline.http import BROWSER_UA, DEFAULT_HEADERS, NBA_CDN_HEADERS

    variants = {
        "pipeline": None,
        "nba_api": {**NBA_CDN_HEADERS, "Host": "cdn.nba.com"},
        "minimal": {"User-Agent": BROWSER_UA, "Referer": "https://www.nba.com/"},
        "bare": {},
    }
    lines = ["## Data source probe", "", "| Source | Headers | Result |", "|---|---|---|"]
    for name, url in PROBE_URLS.items():
        for label, headers in variants.items():
            if label != "pipeline" and "nba.com" not in url:
                continue
            if label == "nba_api" and "cdn.nba.com" not in url:
                continue
            sent = headers if headers is not None else (
                NBA_CDN_HEADERS if "nba.com" in url else DEFAULT_HEADERS)
            try:
                r = requests.get(url, headers=sent, timeout=20)
                kind = r.headers.get("Content-Type", "?").split(";")[0]
                ok = "OK" if r.status_code == 200 and "json" in kind else "FAIL"
                result = f"{ok} {r.status_code} {kind}, {len(r.content)} bytes, server {r.headers.get('Server', '?')}"
            except requests.RequestException as e:
                result = f"FAIL {type(e).__name__}"
            lines.append(f"| {name} | {label} | {result} |")
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────
def resolve_mode(mode: str, now: datetime) -> str:
    if mode != "auto":
        return mode
    hour = now.astimezone(timezone.utc).hour
    if hour < 12:
        return "postgame"
    return "morning" if hour < 20 else "closing"


def set_output(**kwargs) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        for k, v in kwargs.items():
            f.write(f"{k}={v}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="auto", choices=["auto", "postgame", "morning", "closing", "validate", "probe"])
    parser.add_argument("--date", help="Override today's date (ET), YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Take snapshots even if already taken today")
    parser.add_argument("--sample", type=int, default=40, help="Games to check in validate mode")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    now = datetime.now(timezone.utc)
    today = args.date or now.astimezone(EASTERN).strftime("%Y-%m-%d")
    mode = resolve_mode(args.mode, now)
    LOG.info("Mode %s for %s", mode, today)

    if mode == "probe":
        run_probe()
        set_output(collected="false", mode=mode, date=today)
        return 0

    if mode == "validate":
        ok = run_validate(args.sample)
        set_output(collected="false", mode=mode, date=today)
        return 0 if ok else 1

    season, schedule = nba_cdn.fetch_schedule()
    logs, new_games = ingest_boxscores(schedule, season, today)
    games = refresh_games(season, schedule, logs)
    history = rebuild_history(games, logs, season, today)

    snap = {}
    if mode in ("morning", "closing"):
        snap = take_snapshots(mode, games, today, now, args.force)
    merged_rows = rebuild_merged(history, games)

    todays = int((games["game_date"] == today).sum())
    stats = (f"{todays} games today; {new_games} new box scores; "
             f"odds rows {snap.get('odds_rows', 0)}; kalshi rows {snap.get('kalshi_rows', 0)}; "
             f"Odds API credits left {snap.get('credits_left') or 'n/a'}; merged rows {merged_rows}")
    LOG.info(stats)
    set_output(collected="true", mode=mode, date=today, season=season, stats=stats)

    failed = False
    if snap.get("odds_error"):
        print(f"::error::Odds snapshot failed: {snap['odds_error']}")
        failed = True
    elif snap.get("upcoming") and not snap.get("odds_rows") and not snap.get("odds_skipped"):
        print(f"::error::{snap['upcoming']} games still to tip off today but the odds snapshot is empty")
        failed = True
    if snap.get("kalshi_error"):
        print(f"::warning::Kalshi snapshot failed: {snap['kalshi_error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

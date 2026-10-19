#!/usr/bin/env python3
"""
NBA Betting Data Collector
==========================
Idempotent pipeline run by GitHub Actions:

  postgame  Refresh the schedule, store box scores for finished games, rebuild
            the pre-game cumulative stats (including rows for today's games).
  morning   Everything postgame does, plus a morning odds + Kalshi snapshot.
  closing   Everything postgame does, plus a near-tipoff odds + Kalshi snapshot.
  validate  One-off check that ESPN box score fields match the source the
            historical CSV was built from.

Re-running a mode on the same day is safe: snapshots already taken are skipped.
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from nba_pipeline import espn, features, kalshi, merge, odds
from nba_pipeline.config import (EASTERN, GAMES_FILE, HISTORICAL_FILE, KALSHI_FILE, KALSHI_SPREAD_FILE,
                                 MERGED_FILE, ODDS_FILE, TEAM_LOGS_FILE, season_label)

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
def update_games(season: str, today: str, logs: pd.DataFrame) -> pd.DataFrame:
    """Fetch the schedule window that can still change and upsert it into games.csv.

    The first run of a season pulls everything from October 1 through tomorrow; later
    runs re-pull from the earliest past game not yet final (or three days back).
    Box scores are authoritative for final scores.
    """
    existing = read_csv(GAMES_FILE, espn.GAME_COLUMNS)
    day = date.fromisoformat(today)
    season_start = date(espn.season_start_year(day), 10, 1)
    mine = existing[existing["season"] == season] if not existing.empty else existing
    if mine.empty:
        start = season_start
    else:
        postponed = mine["postponed"].astype(str).str.lower() == "true"
        open_past = mine[(mine["game_date"] < today) & (pd.to_numeric(mine["game_status"]) != 3) & ~postponed]
        start = min([day - timedelta(days=3)] + [date.fromisoformat(d) for d in open_past["game_date"]])
        start = max(start, season_start)
    fresh = pd.DataFrame(espn.fetch_games(start, day + timedelta(days=1)), columns=espn.GAME_COLUMNS)

    frames = [f for f in (existing, fresh) if not f.empty]
    games = pd.concat(frames, ignore_index=True) if frames else fresh
    games["game_id"] = games["game_id"].astype(str)
    games = games.drop_duplicates("game_id", keep="last")
    return apply_finals(games, logs)


def apply_finals(games: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Take final scores from stored box scores, then save games.csv."""
    games = games.copy()
    finals = logs[logs["is_home"].astype(str).str.lower() == "true"].set_index("game_id")
    done = games["game_id"].isin(finals.index)
    games.loc[done, "game_status"] = 3
    games.loc[done, "home_points"] = games.loc[done, "game_id"].map(finals["pts"]).values
    games.loc[done, "away_points"] = games.loc[done, "game_id"].map(finals["opp_pts"]).values

    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    for col in ("game_status", "home_points", "away_points"):
        games[col] = pd.to_numeric(games[col]).round().astype("Int64")
    games["postponed"] = games["postponed"].astype(str).str.lower() == "true"
    write_csv(games, GAMES_FILE)
    return games


def ingest_boxscores(games: pd.DataFrame, season: str, today: str) -> tuple[pd.DataFrame, int]:
    """Fetch box scores for finished games not yet in the team game logs."""
    logs = read_csv(TEAM_LOGS_FILE, espn.LOG_COLUMNS)
    have = set(logs["game_id"].astype(str)) if not logs.empty else set()
    status = pd.to_numeric(games["game_status"])
    todo = games[(games["season"] == season) & ~games["game_id"].isin(have)
                 & ((games["game_date"] < today) | ((games["game_date"] == today) & (status == 3)))
                 & ~(games["postponed"] & (status != 3))]
    new_rows, pending = [], 0
    for g in todo.itertuples(index=False):
        box = espn.fetch_boxscore(g.game_id)
        rows = espn.parse_boxscore(box, season) if box else None
        if rows is None:
            pending += 1
            LOG.warning("Box score not available yet for %s (%s %s@%s)", g.game_id, g.game_date,
                        g.away_team_abbreviation, g.home_team_abbreviation)
            continue
        new_rows.extend(rows)
    if new_rows:
        append_csv(pd.DataFrame(new_rows, columns=espn.LOG_COLUMNS), TEAM_LOGS_FILE, espn.LOG_COLUMNS)
        logs = read_csv(TEAM_LOGS_FILE, espn.LOG_COLUMNS)
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
    result = {"odds_rows": 0, "kalshi_rows": 0, "kalshi_spread_rows": 0, "odds_error": None, "kalshi_error": None,
              "credits_left": None}
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

    errors = []
    for series, path, columns, key in ((kalshi.SERIES, KALSHI_FILE, kalshi.COLUMNS, "kalshi_rows"),
                                       (kalshi.SPREAD_SERIES, KALSHI_SPREAD_FILE, kalshi.SPREAD_COLUMNS,
                                        "kalshi_spread_rows")):
        if not force and already_snapped(path, columns, label, today):
            LOG.info("Kalshi %s '%s' snapshot for %s already taken; skipping", series, label, today)
            continue
        try:
            df = kalshi.fetch_snapshot(label, today, games, now, series)
            append_csv(df, path, columns)
            result[key] = len(df)
        except Exception as e:
            LOG.error("Kalshi %s snapshot failed: %s", series, e)
            errors.append(f"{series}: {e}")
    result["kalshi_error"] = "; ".join(errors) or None
    return result


def rebuild_merged(history: pd.DataFrame, games: pd.DataFrame) -> int:
    seasons = set(games["season"].dropna())
    merged = merge.build_merged(history, read_csv(ODDS_FILE, odds.COLUMNS),
                                read_csv(KALSHI_FILE, kalshi.COLUMNS), seasons,
                                read_csv(KALSHI_SPREAD_FILE, kalshi.SPREAD_COLUMNS))
    write_csv(merged, MERGED_FILE)
    return len(merged)


# ─── Validation ───────────────────────────────────────────────────────────
# Stat -> alternative ESPN fields reported alongside the pipeline's choice.
VALIDATE_ALTERNATIVES = {
    "reb": {"offensiveRebounds+defensiveRebounds": ["offensiveRebounds", "defensiveRebounds"]},
    "tov": {"totalTurnovers": ["totalTurnovers"]},
}
VALIDATE_STATS = ["reb", "dreb", "tov", "pf", "blk", "stl", "ast", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]


def run_validate(sample: int) -> bool:
    """Compare ESPN box scores with per-game stats recovered from the historical CSV."""
    from nba_pipeline.legacy import recover_game_logs

    history = read_csv(HISTORICAL_FILE, features.COLUMNS)
    legacy = history[history["home_REB/G"].notna() & history["home_OREB/G"].isna()]
    recovered = recover_game_logs(legacy).dropna(subset=["reb"])
    keys = recovered[["game_date", "game_id"]].drop_duplicates().sort_values("game_id")
    step = max(len(keys) // sample, 1)
    picks = keys.iloc[::step].head(sample)

    hits = {stat: 0 for stat in VALIDATE_STATS}
    alt_hits = {stat: {name: 0 for name in alts} for stat, alts in VALIDATE_ALTERNATIVES.items()}
    compared, misses = 0, 0
    for game_date, gid in picks.itertuples(index=False):
        want = recovered[recovered["game_id"] == gid].set_index("team")
        teams = set(want.index) | set(want["opponent"])
        day = date.fromisoformat(game_date)
        try:
            events = [g for g in espn.fetch_games(day, day)
                      if {g["home_team_abbreviation"], g["away_team_abbreviation"]} == teams]
        except RuntimeError as e:
            events = []
            LOG.warning("Validate: %s", e)
        summary = espn.fetch_boxscore(events[0]["game_id"]) if events else None
        if not summary:
            LOG.warning("Validate: no ESPN box score for %s (%s)", gid, game_date)
            misses += 1
            if misses >= 3 and compared == 0:
                LOG.error("Validate: first %d games all failed; stopping (run mode 'probe' to diagnose)", misses)
                break
            continue
        raw = {t["team"]["abbreviation"]: {x["name"]: x.get("displayValue") for x in t.get("statistics") or []}
               for t in summary["boxscore"]["teams"]}
        for row in espn.parse_boxscore(summary, "validate") or []:
            if row["team"] not in want.index:
                continue
            target = want.loc[row["team"]]
            compared += 1
            for stat in VALIDATE_STATS:
                if row[stat] is not None and abs(row[stat] - target[stat]) < 1e-6:
                    hits[stat] += 1
            stats = next((v for k, v in raw.items() if espn.to_abbr(k) == row["team"]), {})
            for stat, alts in VALIDATE_ALTERNATIVES.items():
                for name, fields in alts.items():
                    if all(stats.get(f) not in (None, "") for f in fields):
                        if abs(sum(float(stats[f]) for f in fields) - target[stat]) < 1e-6:
                            alt_hits[stat][name] += 1

    lines = [f"## ESPN validation: {compared} team-games compared", "",
             "| Stat | Pipeline match | Alternative fields |", "|---|---|---|"]
    ok = compared > 0
    for stat in VALIDATE_STATS:
        rate = hits[stat] / compared if compared else 0
        ok = ok and rate >= 0.95
        alts = ", ".join(f"`{n}` {c}/{compared}" for n, c in alt_hits.get(stat, {}).items())
        lines.append(f"| {stat} | {hits[stat]}/{compared} ({rate:.0%}) | {alts} |")
    report = "\n".join(lines)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    return ok


# ─── Probe ────────────────────────────────────────────────────────────────
PROBE_URLS = {
    "ESPN scoreboard": f"{espn.SCOREBOARD_URL}?dates=20260211",
    "ESPN box score": f"{espn.SUMMARY_URL}?event=401809243",
    "Kalshi markets": f"{kalshi.API}/markets?series_ticker={kalshi.SERIES}&limit=1",
    "Kalshi spread markets": f"{kalshi.API}/markets?series_ticker={kalshi.SPREAD_SERIES}&limit=1",
    "NBA.com CDN schedule (not used; blocked from GitHub)":
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
}


def run_probe() -> None:
    """Report which data sources answer from this machine."""
    import requests
    from nba_pipeline.http import headers_for

    lines = ["## Data source probe", "", "| Source | Result |", "|---|---|"]
    for name, url in PROBE_URLS.items():
        try:
            r = requests.get(url, headers=headers_for(url), timeout=20)
            kind = r.headers.get("Content-Type", "?").split(";")[0]
            ok = "OK" if r.status_code == 200 and "json" in kind else "FAIL"
            result = f"{ok} {r.status_code} {kind}, {len(r.content)} bytes"
        except requests.RequestException as e:
            result = f"FAIL {type(e).__name__}"
        lines.append(f"| {name} | {result} |")
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

    season = season_label(espn.season_start_year(date.fromisoformat(today)))
    games = update_games(season, today, read_csv(TEAM_LOGS_FILE, espn.LOG_COLUMNS))
    logs, new_games = ingest_boxscores(games, season, today)
    if new_games:
        games = apply_finals(games, logs)
    history = rebuild_history(games, logs, season, today)

    snap = {}
    if mode in ("morning", "closing"):
        snap = take_snapshots(mode, games, today, now, args.force)
    merged_rows = rebuild_merged(history, games)

    todays = int((games["game_date"] == today).sum())
    stats = (f"{todays} games today; {new_games} new box scores; "
             f"odds rows {snap.get('odds_rows', 0)}; kalshi rows {snap.get('kalshi_rows', 0)} totals, "
             f"{snap.get('kalshi_spread_rows', 0)} spreads; "
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

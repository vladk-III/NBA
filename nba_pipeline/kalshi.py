"""Kalshi NBA game-total markets (series KXNBATOTAL): every line's price and volume,
plus open-vs-current movement for each game's highest-volume line. Public, no key."""

import logging
import re
from datetime import datetime, timezone

import pandas as pd

from .config import TEAMS, to_abbr
from .http import get_json

LOG = logging.getLogger("nba_collector")

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNBATOTAL"
MAX_TRADE_PAGES = 30

COLUMNS = [
    "snapshot_utc", "snapshot_label", "game_date", "game_id",
    "home_team_abbreviation", "away_team_abbreviation", "event_ticker", "ticker", "line",
    "yes_bid", "yes_ask", "last_price", "previous_price", "volume", "volume_24h",
    "open_interest", "is_top_volume", "open_price", "price_change", "trade_count", "open_price_complete",
]


def date_code(game_date: str) -> str:
    """2026-10-21 -> 26OCT21, the date segment of Kalshi event tickers."""
    return datetime.fromisoformat(game_date).strftime("%y%b%d").upper()


def split_teams(blob: str) -> tuple[str, str] | None:
    """'LALGSW' -> ('LAL', 'GSW') as (away, home); tolerates 2-4 letter provider codes."""
    for i in range(2, min(len(blob) - 1, 5)):
        away, home = to_abbr(blob[:i]), to_abbr(blob[i:])
        if away in TEAMS and home in TEAMS:
            return away, home
    return None


def _cents(m: dict, key: str) -> float | None:
    if m.get(key) is not None:
        return float(m[key])
    dollars = m.get(f"{key}_dollars")
    if dollars not in (None, ""):
        return round(float(dollars) * 100, 2)
    return None


def _count(m: dict, key: str) -> float | None:
    for k in (key, f"{key}_fp"):
        if m.get(k) not in (None, ""):
            return float(m[k])
    return None


def _line(m: dict) -> float | None:
    if m.get("floor_strike") is not None:
        return float(m["floor_strike"])
    match = re.search(r"-(\d+(?:\.\d+)?)$", m.get("ticker", ""))
    return float(match.group(1)) if match else None


def open_markets() -> list[dict]:
    markets, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data, _ = get_json(f"{API}/markets", params=params, attempts=3)
        if data is None:
            raise RuntimeError("Kalshi markets request failed")
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor or not data.get("markets"):
            return markets


def trade_history(ticker: str) -> tuple[float | None, int, bool]:
    """(first trade price in cents, trades seen, whether the full history was read).

    Trades come back newest first, so the opening price is on the last page.
    """
    oldest, count, cursor = None, 0, None
    for _ in range(MAX_TRADE_PAGES):
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        data, _ = get_json(f"{API}/markets/trades", params=params, attempts=3)
        if data is None:
            return oldest, count, False
        trades = data.get("trades", [])
        count += len(trades)
        if trades:
            first = min(trades, key=lambda t: t.get("created_time", ""))
            if oldest is None or first.get("created_time", "") <= oldest[0]:
                oldest = (first.get("created_time", ""), _cents(first, "yes_price"))
        cursor = data.get("cursor")
        if not cursor or not trades:
            return (oldest[1] if oldest else None), count, True
    return (oldest[1] if oldest else None), count, False


def fetch_snapshot(label: str, game_date: str, games: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """All open total lines for game_date's games that have not tipped off."""
    code = date_code(game_date)
    todays = games[games["game_date"] == game_date]
    by_teams = {(r.away_team_abbreviation, r.home_team_abbreviation): r for r in todays.itertuples(index=False)}

    rows = []
    for m in open_markets():
        event = m.get("event_ticker") or m.get("ticker", "").rsplit("-", 1)[0]
        seg = event.split("-", 1)[1] if "-" in event else ""
        if not seg.startswith(code):
            continue
        teams = split_teams(seg[len(code):])
        if teams is None:
            LOG.warning("Kalshi event %s: could not parse teams", event)
            continue
        game = by_teams.get(teams) or by_teams.get(teams[::-1])
        if game is None:
            LOG.warning("Kalshi event %s has no NBA schedule match", event)
            continue
        tip = pd.Timestamp(game.tipoff_utc) if pd.notna(game.tipoff_utc) else None
        if tip is not None and tip <= pd.Timestamp(now):
            continue
        rows.append({
            "snapshot_utc": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "snapshot_label": label, "game_date": game_date, "game_id": game.game_id,
            "home_team_abbreviation": game.home_team_abbreviation,
            "away_team_abbreviation": game.away_team_abbreviation,
            "event_ticker": event, "ticker": m.get("ticker"), "line": _line(m),
            "yes_bid": _cents(m, "yes_bid"), "yes_ask": _cents(m, "yes_ask"),
            "last_price": _cents(m, "last_price"), "previous_price": _cents(m, "previous_price"),
            "volume": _count(m, "volume"), "volume_24h": _count(m, "volume_24h"),
            "open_interest": _count(m, "open_interest"), "is_top_volume": False,
        })

    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        LOG.info("Kalshi snapshot '%s': no open markets for %s", label, code)
        return df
    for col in ("open_price", "price_change", "trade_count", "open_price_complete", "is_top_volume"):
        df[col] = df[col].astype(object)
    top = df.sort_values("volume", ascending=False, na_position="last").groupby("game_id").head(1).index
    df.loc[top, "is_top_volume"] = True
    for idx in top:
        open_price, n, complete = trade_history(df.at[idx, "ticker"])
        df.at[idx, "open_price"] = open_price
        df.at[idx, "trade_count"] = n
        df.at[idx, "open_price_complete"] = complete
        if open_price is not None and pd.notna(df.at[idx, "last_price"]):
            df.at[idx, "price_change"] = df.at[idx, "last_price"] - open_price
    LOG.info("Kalshi snapshot '%s': %d lines across %d games", label, len(df), df["game_id"].nunique())
    return df

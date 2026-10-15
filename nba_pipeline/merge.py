"""Model-ready table: pre-game stats + morning/closing lines + Kalshi + results."""

import numpy as np
import pandas as pd

# Books given their own wide columns; "Consensus" is the median across every US book.
MERGE_BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM"}

MARKETS = {
    "Moneyline Home": "ml_home", "Moneyline Away": "ml_away",
    "Point Spread Home": "spread_home", "Point Spread Home Odds": "spread_home_price",
    "Point Spread Away": "spread_away", "Point Spread Away Odds": "spread_away_price",
    "Over/Under Total": "total", "Over Odds": "over_price", "Under Odds": "under_price",
}

KALSHI = {"Kalshi Line": "line", "Kalshi Open Price": "open_price", "Kalshi Current Price": "last_price",
          "Kalshi Price Change": "price_change", "Kalshi Volume": "volume"}

# snapshot label -> column prefix. Morning columns keep the original names.
LABELS = {"morning": "", "closing": "Close "}


def _latest(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df[(df["snapshot_label"] == label) & df["game_id"].notna()]
    last = df.groupby("game_id")["snapshot_utc"].transform("max")
    return df[df["snapshot_utc"] == last]


def odds_wide(odds: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, prefix in LABELS.items():
        snap = _latest(odds, label)
        if snap.empty:
            continue
        cols = {}
        for book, title in MERGE_BOOKS.items():
            b = snap[snap["bookmaker"] == book].drop_duplicates("game_id").set_index("game_id")
            for market, src in MARKETS.items():
                cols[f"{prefix}{title} - {market}"] = pd.to_numeric(b[src], errors="coerce")
        cons = snap.assign(**{v: pd.to_numeric(snap[v], errors="coerce") for v in MARKETS.values()})
        cons = cons.groupby("game_id")[list(MARKETS.values())].median()
        for market, src in MARKETS.items():
            cols[f"{prefix}Consensus - {market}"] = cons[src]
        cols[f"{prefix}Books Quoted"] = snap.groupby("game_id")["bookmaker"].nunique()
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def kalshi_wide(kalshi: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, prefix in LABELS.items():
        snap = _latest(kalshi, label)
        snap = snap[snap["is_top_volume"].astype(str).str.lower() == "true"]
        if snap.empty:
            continue
        snap = snap.drop_duplicates("game_id").set_index("game_id")
        frames.append(pd.DataFrame({f"{prefix}{name}": pd.to_numeric(snap[src], errors="coerce")
                                    for name, src in KALSHI.items()}))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _hit(actual: pd.Series, line: pd.Series) -> pd.Series:
    """1 over/cover, 0 under/miss, blank on a push or a missing value."""
    out = np.where(actual > line, 1.0, np.where(actual < line, 0.0, np.nan))
    return pd.Series(out, index=actual.index).where(actual.notna() & line.notna())


def build_merged(history: pd.DataFrame, odds: pd.DataFrame, kalshi: pd.DataFrame,
                 seasons: set[str]) -> pd.DataFrame:
    df = history[history["season"].isin(seasons)].copy()
    df["game_id"] = df["game_id"].astype(str)
    for wide in (odds_wide(odds), kalshi_wide(kalshi)):
        if not wide.empty:
            wide.index = wide.index.astype(str)
            df = df.join(wide, on="game_id")

    home = pd.to_numeric(df["home_points"]).astype(float)
    away = pd.to_numeric(df["away_points"]).astype(float)
    total = home + away
    df["Total Score"] = total
    margin = home - away
    for prefix in LABELS.values():
        for book in ("DraftKings", "Consensus"):
            line_col = f"{prefix}{book} - Over/Under Total"
            if line_col in df:
                df[f"{prefix}Over Hit ({book})"] = _hit(total, df[line_col])
            spread_col = f"{prefix}{book} - Point Spread Home"
            if spread_col in df:
                df[f"{prefix}Home Covered ({book})"] = _hit(margin + df[spread_col], pd.Series(0.0, index=df.index))
    return df.reset_index(drop=True)

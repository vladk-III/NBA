"""Recover per-game team stats from the cumulative historical CSV.

A team's pre-game average after k games is A_k, so its k-th game value is
k * A_k - (k - 1) * A_(k-1). This lets us check that NBA CDN box scores line up
with the source the historical CSV was built from, game by game.
"""

import pandas as pd

# Historical column suffix -> team game log column.
RECOVERABLE = {
    "PTS/G": "pts", "REB/G": "reb", "D_REB/G": "dreb", "TO/G": "tov", "block": "blk",
    "plus": "margin", "steal": "stl", "foul": "pf", "assist": "ast",
    "FTM/G": "ftm", "FTA/G": "fta", "FGM/G": "fgm", "FGA/G": "fga",
    "FG3M/G": "fg3m", "FG3A/G": "fg3a",
}


def team_rows(history: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with that team's pre-game averages."""
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        cols = {f"{side}_{s}": s for s in RECOVERABLE}
        df = history[["game_id", "game_date", f"{side}_team_abbreviation", f"{other}_team_abbreviation",
                      f"{side}_points", f"{other}_points", f"{side}_games", f"{side}_wins_current", *cols]]
        df = df.rename(columns={f"{side}_team_abbreviation": "team", f"{other}_team_abbreviation": "opponent",
                                f"{side}_points": "pts", f"{other}_points": "opp_pts",
                                f"{side}_games": "gp", f"{side}_wins_current": "wins", **cols})
        df["is_home"] = side == "home"
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)


def recover_game_logs(history: pd.DataFrame) -> pd.DataFrame:
    """Per-game stats recovered from consecutive pre-game averages.

    Each team's last game has no following row, so its box score stats are not
    recoverable (points and the result still are, from the score columns).
    """
    rows = team_rows(history)
    out = []
    for _, df in rows.groupby("team"):
        df = df.reset_index(drop=True)
        for i in range(len(df)):
            r = df.iloc[i]
            rec = {"game_id": r["game_id"], "game_date": r["game_date"], "team": r["team"],
                   "opponent": r["opponent"], "is_home": r["is_home"],
                   "pts": float(r["pts"]), "opp_pts": float(r["opp_pts"]),
                   "win": int(r["pts"] > r["opp_pts"])}
            if i + 1 < len(df):
                nxt = df.iloc[i + 1]
                k = int(nxt["gp"])
                for suffix, col in RECOVERABLE.items():
                    prev = float(r[suffix]) if k > 1 else 0.0
                    rec[col] = round(k * float(nxt[suffix]) - (k - 1) * prev, 6)
            out.append(rec)
    logs = pd.DataFrame(out)
    return logs

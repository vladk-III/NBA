# NBA Betting Data Collector

Automated pipeline that collects NBA box scores, pre-game cumulative team stats, multi-sportsbook odds and Kalshi game-total markets, committed as CSVs to this repo by GitHub Actions. It follows the same pattern as the NFL and CFB collectors: idempotent runs, season-aware, with optional email summaries.

```
Postgame (3:17 AM ET)               Pregame (8:47 AM ET)             Closing (optional, manual)
┌───────────────────────────┐       ┌──────────────────────────┐     ┌──────────────────────────┐
│ • NBA schedule (ESPN)     │       │ • Everything postgame does│     │ • Everything postgame does│
│ • Box scores of finals    │  ──▶  │ • Odds API snapshot       │ ──▶ │ • Odds API snapshot       │
│ • Rebuild pre-game stats  │       │ • Kalshi totals snapshot  │     │ • Kalshi totals snapshot  │
│   (incl. today's games)   │       │ • Rebuild merged table    │     │ • Rebuild merged table    │
└───────────────────────────┘       └──────────────────────────┘     └──────────────────────────┘
```

## Schedule

Two runs a day. GitHub delays scheduled runs (they have been starting 2.5 to 4 hours late) and drops some at the top of the hour, so both are on odd minutes with plenty of slack. Cron is in UTC, so Eastern times shift by an hour when daylight saving time ends.

| Cron (UTC)    | EDT / EST      | Mode     | Purpose |
|---------------|----------------|----------|---------|
| `47 12 * * *` | 8:47 / 7:47 AM | morning  | Before the day's games: refresh schedule and stats (today's rows), then the odds and Kalshi snapshot |
| `17 7 * * *`  | 3:17 / 2:17 AM | postgame | After the games: final scores and box scores, rebuild stats |

Snapshots only cover games that have not tipped off yet. On days without games, every run is a no-op. A run fails (and GitHub emails you) if games remain today but the odds snapshot came back empty or errored; stats are still committed. The pregame run also collects any box scores the postgame run missed.

One odds call a day is about 3 Odds API credits, roughly 90 a month. A near-tipoff `closing` snapshot is still available from Actions → Run workflow; without it, the closing line value columns stay blank.

## Setup

1. **Repository secrets** (Settings → Secrets and variables → Actions):
   - `ODDS_API_KEY`: [The Odds API](https://the-odds-api.com) key. The free tier (500 credits/month) is enough: each snapshot is one call costing 3 credits, so about 6 credits per game day.
   - Optional email: `EMAIL_USER`, `EMAIL_APP_PASSWORD`, `EMAIL_TO` (Gmail app password, same as the NFL repo).
2. ESPN and Kalshi need no keys.
3. Scheduled workflows only run from the default branch, so the workflow must be on `main`.
4. **Checks you can run any time** (Actions → NBA Betting Data Collector → Run workflow):
   - `probe` reports which data sources answer from GitHub's servers.
   - `validate` downloads ESPN box scores for ~40 games from the 2025-26 historical CSV and compares each stat with the values recovered from that CSV, writing a table to the run summary. All rows should be at or near 100%.

Manual runs (Actions → NBA Betting Data Collector → Run workflow) accept a mode, a date override and `force` to retake a snapshot.

Locally:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
ODDS_API_KEY=... python collect_nba.py --mode morning
```

## Data files (`data/`)

### `historical_nba_cumulative_stats.csv`
One row per game with each team's **season-to-date averages from before that game**, so a row never contains its own result. This is the original 2025-26 file (811 games, Oct 21 – Feb 11) with this season's games appended.

- The first 50 columns are unchanged from the original file. Shooting percentages are total makes over total attempts. `plus` is the average point margin.
- New columns at the end: `season`, `game_type` (regular / play-in / playoffs), `OREB/G`, `OPP_PTS/G` (points allowed), `POSS/G` (estimated possessions: FGA − OREB + TOV + 0.44·FTA), `rest_days` and `back_to_back`. They are blank for the 2025-26 rows, whose box scores were never collected.
- Today's games are added by the overnight run with blank `home_points`, `away_points` and `outcome`, which are filled in the next morning once the game is final.
- A team's first game of the season has 0 for every average, matching the original file; filter on `home_games`/`away_games` if needed.
- The current season is recomputed from `team_game_logs.csv` on every run, so a late box score or stat correction flows through automatically. Earlier seasons are never modified.

### `team_game_logs.csv`
One row per team per final game: the box score totals everything else is computed from.

### `games.csv`
The season schedule with tip-off times, status and final scores.

### `odds_snapshots.csv`
Append-only. One row per snapshot × game × sportsbook (every US book the Odds API returns), with moneyline, spread and total plus their prices. `snapshot_label` is `morning` or `closing`.

### `kalshi_snapshots.csv`
Append-only. Every open line in Kalshi's NBA game-total markets per snapshot: bid/ask, last price, price 24h ago, volume, open interest. For each game's highest-volume line, `open_price` (first trade) and `price_change` (last − open) are added.

### `nba_merged_data.csv`
The model-ready table for the current season: the historical stats row, the morning lines with the original column names (`DraftKings - Over/Under Total`, …), the same for closing lines prefixed with `Close `, a `Consensus` median across all books, Kalshi's top-volume line, and the results `Total Score`, `Over Hit (...)` and `Home Covered (...)` (blank on a push or before the game is played).

## Scope

- Preseason, All-Star and the NBA Cup final are excluded. The Cup final doesn't count toward team records; the other Cup games are regular-season games and are included.
- Data sources: [ESPN's public site API](https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard) (schedule, box scores), [The Odds API](https://the-odds-api.com), [Kalshi](https://kalshi.com) series `KXNBATOTAL`.
- NBA.com's own feeds (stats.nba.com and cdn.nba.com) refuse requests from GitHub Actions servers with HTTP 403, so they are not used. ESPN's team box scores match the original CSV's source stat for stat (rebounds = `totalRebounds`, turnovers = `turnovers`, fouls = `fouls`).
- `game_id` is ESPN's event id for games collected by this pipeline (e.g. `401809243`); the original 2025-26 rows keep their NBA.com ids (e.g. `0022500001`).

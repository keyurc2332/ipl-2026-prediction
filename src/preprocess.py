"""
preprocess.py
--------------
Reads all 1175 Cricsheet IPL JSON files from data/raw/
Produces:
  • data/processed/matches.csv
  • data/processed/batting.csv
  • data/processed/bowling.csv
  • data/processed/deliveries.csv

Run: python src/preprocess.py
"""

import os, glob, json
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"
LIVE_DIR = "data/2026_live"
os.makedirs(PROC_DIR, exist_ok=True)


# ── Team name normalisation ───────────────────────────────────────────────────
TEAM_MAP = {
    "Royal Challengers Bangalore":  "RCB",
    "Royal Challengers Bengaluru":  "RCB",
    "Chennai Super Kings":          "CSK",
    "Mumbai Indians":               "MI",
    "Kolkata Knight Riders":        "KKR",
    "Sunrisers Hyderabad":          "SRH",
    "Deccan Chargers":              "SRH",
    "Rajasthan Royals":             "RR",
    "Delhi Capitals":               "DC",
    "Delhi Daredevils":             "DC",
    "Punjab Kings":                 "PBKS",
    "Kings XI Punjab":              "PBKS",
    "Gujarat Titans":               "GT",
    "Lucknow Super Giants":         "LSG",
    # Defunct — will be dropped
    "Kochi Tuskers Kerala":         "SKIP",
    "Rising Pune Supergiant":       "SKIP",
    "Rising Pune Supergiants":      "SKIP",
    "Pune Warriors":                "SKIP",
    "Gujarat Lions":                "SKIP",
}

SKIP_TEAMS = {"SKIP"}

def norm(name: str) -> str:
    """Normalise a team name to its short code."""
    return TEAM_MAP.get(str(name).strip(), str(name).strip())


# ── Season string → int ───────────────────────────────────────────────────────
def parse_season(s) -> int:
    s = str(s).strip()
    if "/" in s:
        parts  = s.split("/")
        prefix = parts[0][:2]
        return int(prefix + parts[1])
    try:
        return int(s)
    except Exception:
        return 0


# ── Parse one JSON match file ─────────────────────────────────────────────────
def parse_match(filepath: str, match_id: str):
    with open(filepath, "r") as f:
        data = json.load(f)

    info    = data.get("info", {})
    innings = data.get("innings", [])

    # Normalise team names immediately
    raw_teams = info.get("teams", ["", ""])
    teams     = [norm(t) for t in raw_teams]

    outcome = info.get("outcome", {})
    toss    = info.get("toss", {})
    dates   = info.get("dates", [""])
    season  = parse_season(info.get("season", 0))

    raw_winner = outcome.get("winner", "")
    winner     = norm(raw_winner) if raw_winner else ""
    win_by     = outcome.get("by", {})
    no_result  = not winner

    match_row = {
        "match_id":        match_id,
        "season":          season,
        "date":            dates[0] if dates else "",
        "venue":           info.get("venue", ""),
        "city":            info.get("city", ""),
        "team1":           teams[0] if len(teams) > 0 else "",
        "team2":           teams[1] if len(teams) > 1 else "",
        "toss_winner":     norm(toss.get("winner", "")),
        "toss_decision":   toss.get("decision", ""),
        "winner":          winner if not no_result else "NR",
        "win_by_runs":     win_by.get("runs", 0),
        "win_by_wickets":  win_by.get("wickets", 0),
        "player_of_match": info.get("player_of_match", [None])[0],
        "result":          "no result" if no_result else "normal",
    }

    deliveries = []
    for inn_num, inn in enumerate(innings, start=1):
        batting_team = norm(inn.get("team", ""))
        # bowling team = the other team
        bowling_team = next((t for t in teams if t != batting_team), "")

        for over_data in inn.get("overs", []):
            over_num = over_data.get("over", 0)
            for ball_num, d in enumerate(over_data.get("deliveries", []), start=1):
                runs      = d.get("runs", {})
                extras    = d.get("extras", {})
                wickets   = d.get("wickets", [])
                is_wide   = "wides"   in extras
                is_noball = "noballs" in extras

                delivery = {
                    "match_id":     match_id,
                    "season":       season,
                    "date":         dates[0] if dates else "",
                    "venue":        info.get("venue", ""),
                    "innings":      inn_num,
                    "over":         over_num,
                    "ball":         ball_num,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "batter":       d.get("batter", ""),
                    "bowler":       d.get("bowler", ""),
                    "non_striker":  d.get("non_striker", ""),
                    "runs_batter":  runs.get("batter", 0),
                    "runs_extras":  runs.get("extras", 0),
                    "runs_total":   runs.get("total", 0),
                    "is_wide":      int(is_wide),
                    "is_noball":    int(is_noball),
                    "wides":        extras.get("wides", 0),
                    "noballs":      extras.get("noballs", 0),
                    "byes":         extras.get("byes", 0),
                    "legbyes":      extras.get("legbyes", 0),
                    "wicket":       int(len(wickets) > 0),
                    "wicket_kind":  wickets[0].get("kind", "") if wickets else "",
                    "player_out":   wickets[0].get("player_out", "") if wickets else "",
                }
                deliveries.append(delivery)

    return match_row, deliveries


# ── Load all JSON files ───────────────────────────────────────────────────────
def load_all(raw_dir: str):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    print(f"Found {len(files)} JSON files")

    all_matches    = []
    all_deliveries = []

    for filepath in tqdm(files, desc="Parsing JSONs"):
        match_id = os.path.basename(filepath).replace(".json", "")
        try:
            match_row, deliveries = parse_match(filepath, match_id)
            all_matches.append(match_row)
            all_deliveries.extend(deliveries)
        except Exception as e:
            print(f"  ⚠ Skipped {match_id}: {e}")

    matches_df    = pd.DataFrame(all_matches)
    deliveries_df = pd.DataFrame(all_deliveries)
    return matches_df, deliveries_df


# ── Drop defunct teams ────────────────────────────────────────────────────────
def drop_defunct(matches: pd.DataFrame, deliveries: pd.DataFrame):
    before_m = len(matches)
    before_d = len(deliveries)

    matches = matches[
        ~matches["team1"].isin(SKIP_TEAMS) &
        ~matches["team2"].isin(SKIP_TEAMS)
    ].reset_index(drop=True)

    deliveries = deliveries[
        ~deliveries["batting_team"].isin(SKIP_TEAMS) &
        ~deliveries["bowling_team"].isin(SKIP_TEAMS)
    ].reset_index(drop=True)

    print(f"   Dropped defunct teams → matches: {before_m} → {len(matches)} | deliveries: {before_d} → {len(deliveries)}")
    return matches, deliveries


# ── Batting stats per player per season ───────────────────────────────────────
def build_batting(deliveries: pd.DataFrame) -> pd.DataFrame:
    faced = deliveries[deliveries["is_wide"] == 0].copy()

    dismissals = deliveries[
        (deliveries["wicket"] == 1) &
        (~deliveries["wicket_kind"].isin(["run out", "obstructing the field", "retired hurt"]))
    ].groupby(["season", "player_out", "batting_team"])["match_id"].count().reset_index()
    dismissals.columns = ["season", "player", "team", "dismissals"]

    batting = (
        faced.groupby(["season", "batter", "batting_team"])
        .agg(
            matches     = ("match_id",    "nunique"),
            runs        = ("runs_batter", "sum"),
            balls_faced = ("runs_batter", "count"),
            fours       = ("runs_batter", lambda x: (x == 4).sum()),
            sixes       = ("runs_batter", lambda x: (x == 6).sum()),
        )
        .reset_index()
        .rename(columns={"batter": "player", "batting_team": "team"})
    )

    batting = batting.merge(dismissals, on=["season", "player", "team"], how="left")
    batting["dismissals"]  = batting["dismissals"].fillna(0)
    batting["average"]     = np.where(
        batting["dismissals"] > 0,
        batting["runs"] / batting["dismissals"],
        batting["runs"]
    )
    batting["strike_rate"] = (batting["runs"] / batting["balls_faced"]) * 100

    return batting


# ── Bowling stats per player per season ───────────────────────────────────────
def build_bowling(deliveries: pd.DataFrame) -> pd.DataFrame:
    legal = deliveries[
        (deliveries["is_wide"] == 0) &
        (deliveries["is_noball"] == 0)
    ].copy()

    bowling = (
        legal.groupby(["season", "bowler", "bowling_team"])
        .agg(
            matches       = ("match_id",   "nunique"),
            balls_bowled  = ("runs_total", "count"),
            runs_conceded = ("runs_total", "sum"),
            wickets       = ("wicket",     "sum"),
        )
        .reset_index()
        .rename(columns={"bowler": "player", "bowling_team": "team"})
    )

    # Add extras (wides + noballs) back to runs conceded
    extras = (
        deliveries.groupby(["season", "bowler", "bowling_team"])
        .agg(extra_runs=("runs_extras", "sum"))
        .reset_index()
        .rename(columns={"bowler": "player", "bowling_team": "team"})
    )
    bowling = bowling.merge(extras, on=["season", "player", "team"], how="left")
    bowling["runs_conceded"] += bowling["extra_runs"].fillna(0)

    bowling["overs"]   = bowling["balls_bowled"] / 6
    bowling["economy"] = np.where(
        bowling["balls_bowled"] > 0,
        bowling["runs_conceded"] / (bowling["balls_bowled"] / 6),
        0
    )
    bowling["average"] = np.where(
        bowling["wickets"] > 0,
        bowling["runs_conceded"] / bowling["wickets"],
        np.nan
    )
    bowling["sr"] = np.where(
        bowling["wickets"] > 0,
        bowling["balls_bowled"] / bowling["wickets"],
        np.nan
    )

    return bowling


# ── Append 2026 live match data ───────────────────────────────────────────────
def append_2026(matches: pd.DataFrame) -> pd.DataFrame:
    live_path = os.path.join(LIVE_DIR, "ipl_2026_matches.csv")
    if not os.path.exists(live_path):
        print("⚠  No 2026 live data found in data/2026_live/ — skipping")
        return matches
    live = pd.read_csv(live_path)
    live["season"] = 2026
    combined = pd.concat([matches, live], ignore_index=True)
    print(f"✅ Appended 2026 live data ({len(live)} matches)")
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n🏏 IPL Prediction — Preprocessing\n" + "="*45)

    matches, deliveries = load_all(RAW_DIR)

    print(f"\n✅ Parsed  : {len(matches):,} matches | {len(deliveries):,} deliveries")
    print(f"   Seasons : {sorted(matches['season'].unique())}")

    # Drop defunct/SKIP teams
    matches, deliveries = drop_defunct(matches, deliveries)

    # Verify all 10 current teams present
    all_teams = set(matches["team1"].unique()) | set(matches["team2"].unique())
    print(f"   Teams   : {sorted(all_teams)}")

    batting = build_batting(deliveries)
    bowling = build_bowling(deliveries)
    matches = append_2026(matches)

    matches.to_csv(f"{PROC_DIR}/matches.csv",        index=False)
    batting.to_csv(f"{PROC_DIR}/batting.csv",        index=False)
    bowling.to_csv(f"{PROC_DIR}/bowling.csv",        index=False)
    deliveries.to_csv(f"{PROC_DIR}/deliveries.csv",  index=False)

    print(f"\n📦 Saved to {PROC_DIR}/")
    print(f"   matches.csv    → {len(matches):,} rows")
    print(f"   batting.csv    → {len(batting):,} rows")
    print(f"   bowling.csv    → {len(bowling):,} rows")
    print(f"   deliveries.csv → {len(deliveries):,} rows")


if __name__ == "__main__":
    run()
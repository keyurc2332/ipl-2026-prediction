"""
pitch_features.py
------------------
Reads Kaggle pitch data and builds:
  1. data/processed/pitch_lookup.csv   — per match pitch type + conditions
  2. data/processed/pitch_win_rates.csv — per team win rate per pitch type

Run: python src/pitch_features.py
"""

import os
import pandas as pd
import numpy as np

RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"

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
}

PITCH_CODE = {
    "Batting-friendly": 0,
    "Balanced":         1,
    "Spin-friendly":    2,
    "Sluggish":         3,
}

DEW_CODE = {
    "High":   2,
    "Medium": 1,
    "Low":    0,
}

def norm(name):
    return TEAM_MAP.get(str(name).strip(), str(name).strip())


def build_pitch_lookup():
    print("📖 Reading pitch data...")
    df = pd.read_excel(f"{RAW_DIR}/Ipl match data - enriched.xlsx")

    # One row per match — take first delivery per match (pitch type is match-level)
    match_pitch = df.groupby("match_id").first().reset_index()[[
        "match_id", "date", "pitch_type", "dew_prediction",
        "grass_cover", "moisture", "bounce_and_carry"
    ]]

    # Encode pitch type
    match_pitch["pitch_type_code"] = match_pitch["pitch_type"].map(PITCH_CODE).fillna(1)
    match_pitch["is_spin_pitch"]   = (match_pitch["pitch_type"] == "Spin-friendly").astype(int)
    match_pitch["is_batting_pitch"]= (match_pitch["pitch_type"] == "Batting-friendly").astype(int)
    match_pitch["is_sluggish"]     = (match_pitch["pitch_type"] == "Sluggish").astype(int)
    match_pitch["dew_risk"]        = match_pitch["dew_prediction"].map(DEW_CODE).fillna(0)

    match_pitch["date"] = pd.to_datetime(match_pitch["date"], errors="coerce")
    match_pitch["match_id"] = match_pitch["match_id"].astype(str)

    print(f"   Pitch lookup: {len(match_pitch)} matches")
    print(f"   Pitch types : {match_pitch['pitch_type'].value_counts().to_dict()}")

    match_pitch.to_csv(f"{PROC_DIR}/pitch_lookup.csv", index=False)
    print(f"   ✅ Saved → data/processed/pitch_lookup.csv")
    return match_pitch


def build_pitch_win_rates():
    """
    For each team, compute win rate on each pitch type.
    Uses our matches.csv + pitch_lookup.csv joined on match_id.
    """
    # f"{RAW_DIR}/all_matches_batting_stats.csv"
    # f"{RAW_DIR}/all_matches_bowling_stats.csv"
    print("\n📊 Building team pitch win rates...")

    matches     = pd.read_csv(f"{PROC_DIR}/matches.csv")
    pitch_lookup= pd.read_csv(f"{PROC_DIR}/pitch_lookup.csv")

    matches["match_id"]      = matches["match_id"].astype(str)
    pitch_lookup["match_id"] = pitch_lookup["match_id"].astype(str)

    merged = matches.merge(pitch_lookup[["match_id", "pitch_type"]], on="match_id", how="left")
    merged = merged[merged["pitch_type"].notna()]

    print(f"   Matched {len(merged)} matches with pitch data")

    rows = []
    teams       = set(merged["team1"].unique()) | set(merged["team2"].unique())
    pitch_types = ["Batting-friendly", "Balanced", "Spin-friendly", "Sluggish"]

    for team in teams:
        for pt in pitch_types:
            pt_matches = merged[
                (merged["pitch_type"] == pt) &
                ((merged["team1"] == team) | (merged["team2"] == team))
            ]
            if len(pt_matches) == 0:
                win_rate = 0.5
            else:
                wins = (pt_matches["winner"] == team).sum()
                win_rate = wins / len(pt_matches)

            rows.append({
                "team":       team,
                "pitch_type": pt,
                "win_rate":   round(win_rate, 4),
                "matches":    len(pt_matches),
            })

    pitch_wr = pd.DataFrame(rows)

    # Print summary
    print("\n   Team win rates by pitch type:")
    pivot = pitch_wr.pivot(index="team", columns="pitch_type", values="win_rate").round(3)
    print(pivot.to_string())

    pitch_wr.to_csv(f"{PROC_DIR}/pitch_win_rates.csv", index=False)
    print(f"\n   ✅ Saved → data/processed/pitch_win_rates.csv")
    return pitch_wr


def run():
    print("\n🏏 IPL Prediction — Pitch Feature Builder\n" + "="*45)
    build_pitch_lookup()
    build_pitch_win_rates()
    print("\n✅ Pitch features ready")


if __name__ == "__main__":
    run()
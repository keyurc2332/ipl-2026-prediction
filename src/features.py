"""
features.py  (v5 — pitch features integrated)
-----------------------------------------------
Run: python src/features.py
"""

import os
import pandas as pd
import numpy as np

PROC_DIR = "data/processed"

MATCH_FEATURES = [
    # Elo
    "team1_elo", "team2_elo", "elo_diff",

    # Rolling form
    "team1_win_rate_last10", "team2_win_rate_last10",
    "team1_win_rate_last5",  "team2_win_rate_last5",
    "form_diff",

    # Weighted win rate
    "team1_weighted_win_rate", "team2_weighted_win_rate",

    # Win streak
    "team1_win_streak", "team2_win_streak",

    # H2H
    "h2h_win_rate_t1",

    # Venue
    "venue_win_rate_t1", "venue_win_rate_t2",
    "venue_diff", "venue_avg_score",
    "is_home_t1", "is_home_t2",

    # Phase batting
    "team1_pp_sr", "team2_pp_sr",
    "team1_death_sr", "team2_death_sr",

    # Phase bowling
    "team1_pp_bowling_eco", "team2_pp_bowling_eco",
    "team1_death_bowling_eco", "team2_death_bowling_eco",

    # Rolling run margins
    "team1_avg_runs_scored", "team2_avg_runs_scored",
    "team1_avg_runs_conceded", "team2_avg_runs_conceded",
    "run_scoring_diff", "run_conceding_diff",

    # Overall quality
    "team1_batting_str", "team2_batting_str",
    "team1_bowling_str", "team2_bowling_str",
    "batting_diff", "bowling_diff",

    # Toss
    "toss_win_t1", "toss_field_t1",

    # Pitch
    "pitch_type_code",
    "is_spin_pitch",
    "is_batting_pitch",
    "is_sluggish",
    "dew_risk",
    "team1_pitch_win_rate",
    "team2_pitch_win_rate",
    "pitch_win_rate_diff",
]

HOME_GROUNDS = {
    "MI":   ["Wankhede Stadium"],
    "CSK":  ["MA Chidambaram Stadium"],
    "RCB":  ["M Chinnaswamy Stadium"],
    "KKR":  ["Eden Gardens"],
    "DC":   ["Arun Jaitley Stadium", "Feroz Shah Kotla"],
    "PBKS": ["Punjab Cricket Association IS Bindra Stadium",
             "Maharaja Yadavindra Singh International Cricket Stadium"],
    "RR":   ["Sawai Mansingh Stadium"],
    "SRH":  ["Rajiv Gandhi International Cricket Stadium"],
    "GT":   ["Narendra Modi Stadium"],
    "LSG":  ["BRSABV Ekana Cricket Stadium",
             "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium"],
}


def load_data():
    matches    = pd.read_csv(f"{PROC_DIR}/matches.csv")
    batting    = pd.read_csv(f"{PROC_DIR}/batting.csv")
    bowling    = pd.read_csv(f"{PROC_DIR}/bowling.csv")
    deliveries = pd.read_csv(f"{PROC_DIR}/deliveries.csv")

    for df in [matches, batting, bowling, deliveries]:
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches         = matches.sort_values("date").reset_index(drop=True)

    pitch_lookup    = pd.read_csv(f"{PROC_DIR}/pitch_lookup.csv")
    pitch_win_rates = pd.read_csv(f"{PROC_DIR}/pitch_win_rates.csv")
    pitch_lookup["match_id"] = pitch_lookup["match_id"].astype(str)

    return matches, batting, bowling, deliveries, pitch_lookup, pitch_win_rates


# ── Elo with season decay ─────────────────────────────────────────────────────
def compute_elo(matches: pd.DataFrame, k: int = 32, base: int = 1500, decay: float = 0.2):
    elo        = {t: base for t in set(matches["team1"]) | set(matches["team2"])}
    t1_elos    = []
    t2_elos    = []
    cur_season = None

    for _, row in matches.iterrows():
        t1     = row["team1"]
        t2     = row["team2"]
        season = row["season"]
        winner = str(row.get("winner", ""))

        if season != cur_season:
            cur_season = season
            for team in elo:
                elo[team] = elo[team] * (1 - decay) + base * decay

        t1_elos.append(elo.get(t1, base))
        t2_elos.append(elo.get(t2, base))

        if winner in ["NR", "TIE", "nan", "", "no result"] or pd.isna(winner):
            continue

        r1 = elo[t1]
        r2 = elo[t2]
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        s1 = 1.0 if winner == t1 else 0.0

        elo[t1] = r1 + k * (s1 - e1)
        elo[t2] = r2 + k * ((1 - s1) - (1 - e1))

    matches          = matches.copy()
    matches["team1_elo"] = t1_elos
    matches["team2_elo"] = t2_elos
    matches["elo_diff"]  = matches["team1_elo"] - matches["team2_elo"]

    print("   Elo (with season decay) ✅")
    final = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    for team, rating in final:
        bar = "█" * int((rating - 1400) / 8)
        print(f"      {team:<6} {rating:>7.1f}  {bar}")

    return matches, elo


# ── Phase stats from deliveries ───────────────────────────────────────────────
def compute_phase_stats(deliveries: pd.DataFrame) -> dict:
    d = deliveries[deliveries["innings"].isin([1, 2])].copy()

    pp    = d[d["over"].between(0, 5)]
    death = d[d["over"].between(16, 19)]

    def batting_sr(subset, col_name):
        legal = subset[subset["is_wide"] == 0]
        g = legal.groupby(["season", "batting_team"]).agg(
            runs  = ("runs_batter", "sum"),
            balls = ("runs_batter", "count")
        ).reset_index()
        g[col_name] = (g["runs"] / g["balls"]) * 100
        return g[["season", "batting_team", col_name]].rename(
            columns={"batting_team": "team"}
        )

    def bowling_eco(subset, col_name):
        legal = subset[(subset["is_wide"] == 0) & (subset["is_noball"] == 0)]
        g = legal.groupby(["season", "bowling_team"]).agg(
            runs  = ("runs_total", "sum"),
            balls = ("runs_total", "count")
        ).reset_index()
        g[col_name] = (g["runs"] / g["balls"]) * 6
        return g[["season", "bowling_team", col_name]].rename(
            columns={"bowling_team": "team"}
        )

    pp_bat     = batting_sr(pp,    "pp_sr")
    death_bat  = batting_sr(death, "death_sr")
    pp_bowl    = bowling_eco(pp,   "pp_eco")
    death_bowl = bowling_eco(death,"death_eco")

    phase = pp_bat.merge(death_bat, on=["season", "team"], how="outer") \
                  .merge(pp_bowl,   on=["season", "team"], how="outer") \
                  .merge(death_bowl,on=["season", "team"], how="outer")

    lookup = {}
    for _, row in phase.iterrows():
        key = (row["team"], int(row["season"]))
        lookup[key] = {
            "pp_sr":     row.get("pp_sr",     130.0),
            "death_sr":  row.get("death_sr",  140.0),
            "pp_eco":    row.get("pp_eco",    8.0),
            "death_eco": row.get("death_eco", 9.5),
        }

    print("   Phase stats computed ✅")
    return lookup


def get_phase(lookup: dict, team: str, season: int) -> dict:
    for s in [season - 1, season - 2, season - 3]:
        if (team, s) in lookup:
            return lookup[(team, s)]
    return {"pp_sr": 130.0, "death_sr": 140.0, "pp_eco": 8.0, "death_eco": 9.5}


# ── Venue average score ───────────────────────────────────────────────────────
def build_venue_stats(deliveries: pd.DataFrame, min_matches: int = 8) -> dict:
    inn1         = deliveries[deliveries["innings"] == 1]
    venue_scores = inn1.groupby(["venue", "match_id"])["runs_total"].sum().reset_index()
    venue_avg    = venue_scores.groupby("venue").agg(
        avg_score   = ("runs_total", "mean"),
        match_count = ("match_id",   "nunique")
    ).reset_index()
    venue_avg = venue_avg[venue_avg["match_count"] >= min_matches]
    return venue_avg.set_index("venue")["avg_score"].to_dict()


# ── Match runs lookup (pre-computed ONCE) ─────────────────────────────────────
def build_match_runs_lookup(deliveries: pd.DataFrame) -> dict:
    inn1    = deliveries[deliveries["innings"] == 1]
    grouped = inn1.groupby(["match_id", "batting_team"])["runs_total"].sum()
    lookup  = {}
    for (match_id, team), runs in grouped.items():
        mid = str(match_id)
        if mid not in lookup:
            lookup[mid] = {}
        lookup[mid][team] = runs
    return lookup


# ── Rolling win rate ──────────────────────────────────────────────────────────
def rolling_win_rate(matches, team, before_idx, n):
    past = matches.iloc[:before_idx]
    tm   = past[(past["team1"] == team) | (past["team2"] == team)].tail(n)
    if len(tm) == 0:
        return 0.5
    return (tm["winner"] == team).sum() / len(tm)


# ── Weighted win rate ─────────────────────────────────────────────────────────
def weighted_win_rate(matches: pd.DataFrame, team: str, before_idx: int, n: int = 10) -> float:
    past         = matches.iloc[:before_idx]
    team_matches = past[
        (past["team1"] == team) | (past["team2"] == team)
    ].tail(n)
    if len(team_matches) == 0:
        return 0.5
    weights   = np.linspace(1, 2, len(team_matches))
    win_flags = (team_matches["winner"] == team).astype(float).values
    return float(np.average(win_flags, weights=weights))


# ── Win streak ────────────────────────────────────────────────────────────────
def win_streak(matches: pd.DataFrame, team: str, before_idx: int) -> int:
    past         = matches.iloc[:before_idx]
    team_matches = past[
        (past["team1"] == team) | (past["team2"] == team)
    ].tail(10)
    streak = 0
    for _, row in team_matches.iloc[::-1].iterrows():
        winner = str(row.get("winner", ""))
        if winner == team:
            if streak >= 0:
                streak += 1
            else:
                break
        elif winner not in ["NR", "TIE", "nan", ""]:
            if streak <= 0:
                streak -= 1
            else:
                break
    return streak


# ── Avg runs scored (fast) ────────────────────────────────────────────────────
def avg_runs_scored(match_runs: dict, matches: pd.DataFrame,
                    team: str, before_idx: int, n: int = 5) -> float:
    past_ids = matches.iloc[:before_idx]
    past_ids = past_ids[
        (past_ids["team1"] == team) | (past_ids["team2"] == team)
    ].tail(n)["match_id"].astype(str).tolist()
    scores = [
        match_runs[mid][team]
        for mid in past_ids
        if mid in match_runs and team in match_runs[mid]
    ]
    return float(np.mean(scores)) if scores else 160.0


# ── Avg runs conceded (fast) ──────────────────────────────────────────────────
def avg_runs_conceded(match_runs: dict, matches: pd.DataFrame,
                      team: str, before_idx: int, n: int = 5) -> float:
    past_ids = matches.iloc[:before_idx]
    past_ids = past_ids[
        (past_ids["team1"] == team) | (past_ids["team2"] == team)
    ].tail(n)["match_id"].astype(str).tolist()
    scores = []
    for mid in past_ids:
        if mid not in match_runs:
            continue
        for batting_team, runs in match_runs[mid].items():
            if batting_team != team:
                scores.append(runs)
    return float(np.mean(scores)) if scores else 160.0


# ── H2H win rate ──────────────────────────────────────────────────────────────
def h2h_win_rate(matches, team1, team2, before_idx):
    past = matches.iloc[:before_idx]
    h2h  = past[
        ((past["team1"] == team1) & (past["team2"] == team2)) |
        ((past["team1"] == team2) & (past["team2"] == team1))
    ].tail(10)
    if len(h2h) == 0:
        return 0.5
    return (h2h["winner"] == team1).sum() / len(h2h)


# ── Venue win rate ────────────────────────────────────────────────────────────
def venue_win_rate(matches, team, venue, before_idx, min_matches=5):
    past     = matches.iloc[:before_idx]
    at_venue = past[
        (past["venue"] == venue) &
        ((past["team1"] == team) | (past["team2"] == team))
    ]
    if len(at_venue) < min_matches:
        all_m = past[(past["team1"] == team) | (past["team2"] == team)]
        if len(all_m) == 0:
            return 0.5
        return (all_m["winner"] == team).sum() / len(all_m)
    return (at_venue["winner"] == team).sum() / len(at_venue)


# ── Home ground ───────────────────────────────────────────────────────────────
def is_home(team, venue):
    homes = HOME_GROUNDS.get(team, [])
    return int(any(h.lower() in venue.lower() or venue.lower() in h.lower()
                   for h in homes))


# ── Team batting strength ─────────────────────────────────────────────────────
def team_batting_strength(batting, team, season):
    recent = batting[
        (batting["team"] == team) &
        (batting["season"] >= season - 2) &
        (batting["season"] <  season) &
        (batting["balls_faced"] >= 40)
    ]
    if recent.empty:
        return 130.0
    return float(recent.nlargest(6, "runs")["strike_rate"].mean())


# ── Team bowling strength ─────────────────────────────────────────────────────
def team_bowling_strength(bowling, team, season):
    recent = bowling[
        (bowling["team"] == team) &
        (bowling["season"] >= season - 2) &
        (bowling["season"] <  season) &
        (bowling["balls_bowled"] >= 48) &
        (bowling["wickets"] > 0)
    ]
    if recent.empty:
        return 8.5
    return float(recent.nsmallest(5, "economy")["economy"].mean())


# ── Pitch type for a match ────────────────────────────────────────────────────
def get_pitch_features(pitch_lookup: pd.DataFrame, match_id: str) -> dict:
    row = pitch_lookup[pitch_lookup["match_id"] == str(match_id)]
    if row.empty:
        return {
            "pitch_type_code":  1,
            "is_spin_pitch":    0,
            "is_batting_pitch": 0,
            "is_sluggish":      0,
            "dew_risk":         0,
        }
    r = row.iloc[0]
    return {
        "pitch_type_code":  int(r.get("pitch_type_code",  1)),
        "is_spin_pitch":    int(r.get("is_spin_pitch",    0)),
        "is_batting_pitch": int(r.get("is_batting_pitch", 0)),
        "is_sluggish":      int(r.get("is_sluggish",      0)),
        "dew_risk":         int(r.get("dew_risk",         0)),
    }


# ── Team win rate on a specific pitch type ────────────────────────────────────
def get_team_pitch_win_rate(pitch_win_rates: pd.DataFrame,
                            team: str, pitch_type_code: int) -> float:
    pitch_map  = {0: "Batting-friendly", 1: "Balanced",
                  2: "Spin-friendly",    3: "Sluggish"}
    pitch_name = pitch_map.get(pitch_type_code, "Balanced")
    row = pitch_win_rates[
        (pitch_win_rates["team"]       == team) &
        (pitch_win_rates["pitch_type"] == pitch_name)
    ]
    if row.empty or row.iloc[0]["matches"] < 3:
        return 0.5
    return float(row.iloc[0]["win_rate"])


# ── Build full feature matrix ─────────────────────────────────────────────────
def build_features(matches, batting, bowling, phase_lookup,
                   venue_avg_scores, deliveries, pitch_lookup, pitch_win_rates):
    rows    = []
    skipped = 0

    match_runs = build_match_runs_lookup(deliveries)
    print("   Match runs lookup built ✅")

    for i, row in matches.iterrows():
        t1     = str(row["team1"])
        t2     = str(row["team2"])
        venue  = str(row.get("venue", ""))
        season = int(row.get("season", 0))
        winner = str(row.get("winner", ""))

        if winner in ["NR", "TIE", "nan", "", "no result"] or pd.isna(winner):
            skipped += 1
            continue
        if season < 2009:
            skipped += 1
            continue

        toss_winner   = str(row.get("toss_winner", ""))
        toss_decision = str(row.get("toss_decision", ""))
        p1 = get_phase(phase_lookup, t1, season)
        p2 = get_phase(phase_lookup, t2, season)

        pitch_feats = get_pitch_features(pitch_lookup, str(row.get("match_id", "")))
        pitch_code  = pitch_feats["pitch_type_code"]

        feat = {
            "match_id": row.get("match_id", i),
            "season":   season,
            "date":     row.get("date", ""),
            "team1":    t1,
            "team2":    t2,
            "venue":    venue,

            # Elo
            "team1_elo": float(row.get("team1_elo", 1500)),
            "team2_elo": float(row.get("team2_elo", 1500)),
            "elo_diff":  float(row.get("elo_diff",  0)),

            # Form
            "team1_win_rate_last10": rolling_win_rate(matches, t1, i, 10),
            "team2_win_rate_last10": rolling_win_rate(matches, t2, i, 10),
            "team1_win_rate_last5":  rolling_win_rate(matches, t1, i, 5),
            "team2_win_rate_last5":  rolling_win_rate(matches, t2, i, 5),

            # Weighted win rate
            "team1_weighted_win_rate": weighted_win_rate(matches, t1, i),
            "team2_weighted_win_rate": weighted_win_rate(matches, t2, i),

            # Win streak
            "team1_win_streak": win_streak(matches, t1, i),
            "team2_win_streak": win_streak(matches, t2, i),

            # H2H
            "h2h_win_rate_t1": h2h_win_rate(matches, t1, t2, i),

            # Venue
            "venue_win_rate_t1": venue_win_rate(matches, t1, venue, i),
            "venue_win_rate_t2": venue_win_rate(matches, t2, venue, i),
            "venue_avg_score":   venue_avg_scores.get(venue, 165.0),
            "is_home_t1":        is_home(t1, venue),
            "is_home_t2":        is_home(t2, venue),

            # Phase — batting
            "team1_pp_sr":    p1["pp_sr"],
            "team2_pp_sr":    p2["pp_sr"],
            "team1_death_sr": p1["death_sr"],
            "team2_death_sr": p2["death_sr"],

            # Phase — bowling
            "team1_pp_bowling_eco":    p1["pp_eco"],
            "team2_pp_bowling_eco":    p2["pp_eco"],
            "team1_death_bowling_eco": p1["death_eco"],
            "team2_death_bowling_eco": p2["death_eco"],

            # Rolling run margins
            "team1_avg_runs_scored":   avg_runs_scored(match_runs, matches, t1, i),
            "team2_avg_runs_scored":   avg_runs_scored(match_runs, matches, t2, i),
            "team1_avg_runs_conceded": avg_runs_conceded(match_runs, matches, t1, i),
            "team2_avg_runs_conceded": avg_runs_conceded(match_runs, matches, t2, i),

            # Overall quality
            "team1_batting_str": team_batting_strength(batting, t1, season),
            "team2_batting_str": team_batting_strength(batting, t2, season),
            "team1_bowling_str": team_bowling_strength(bowling, t1, season),
            "team2_bowling_str": team_bowling_strength(bowling, t2, season),

            # Toss
            "toss_win_t1":   int(toss_winner == t1),
            "toss_field_t1": int(toss_winner == t1 and toss_decision == "field"),

            # Pitch
            "pitch_type_code":  pitch_feats["pitch_type_code"],
            "is_spin_pitch":    pitch_feats["is_spin_pitch"],
            "is_batting_pitch": pitch_feats["is_batting_pitch"],
            "is_sluggish":      pitch_feats["is_sluggish"],
            "dew_risk":         pitch_feats["dew_risk"],
            "team1_pitch_win_rate": get_team_pitch_win_rate(pitch_win_rates, t1, pitch_code),
            "team2_pitch_win_rate": get_team_pitch_win_rate(pitch_win_rates, t2, pitch_code),

            # Target
            "winner_binary": 1 if winner == t1 else 0,
        }

        # Differentials
        feat["run_scoring_diff"]   = feat["team1_avg_runs_scored"]    - feat["team2_avg_runs_scored"]
        feat["run_conceding_diff"] = feat["team2_avg_runs_conceded"]   - feat["team1_avg_runs_conceded"]
        feat["form_diff"]          = feat["team1_win_rate_last10"]     - feat["team2_win_rate_last10"]
        feat["venue_diff"]         = feat["venue_win_rate_t1"]         - feat["venue_win_rate_t2"]
        feat["batting_diff"]       = feat["team1_batting_str"]         - feat["team2_batting_str"]
        feat["bowling_diff"]       = feat["team2_bowling_str"]         - feat["team1_bowling_str"]
        feat["pitch_win_rate_diff"]= feat["team1_pitch_win_rate"]      - feat["team2_pitch_win_rate"]

        rows.append(feat)

    df = pd.DataFrame(rows)
    print(f"\n✅ Features built : {len(df):,} rows | Skipped: {skipped}")
    print(f"   Total features : {len(MATCH_FEATURES)}")
    print(f"   Seasons        : {sorted(df['season'].unique())}")
    print(f"   Class balance  : {df['winner_binary'].mean():.2%} team1 wins")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n🏏 IPL Prediction — Feature Engineering v5\n" + "="*48)
    matches, batting, bowling, deliveries, pitch_lookup, pitch_win_rates = load_data()

    print("\n   Computing Elo with season decay...")
    matches, final_elo = compute_elo(matches)

    print("\n   Computing phase stats (powerplay + death overs)...")
    phase_lookup = compute_phase_stats(deliveries)

    print("\n   Computing venue average scores...")
    venue_avg_scores = build_venue_stats(deliveries)
    print(f"   Trusted venues (8+ matches): {len(venue_avg_scores)}")

    features = build_features(matches, batting, bowling,
                              phase_lookup, venue_avg_scores, deliveries,
                              pitch_lookup, pitch_win_rates)

    out = f"{PROC_DIR}/features.csv"
    features.to_csv(out, index=False)
    print(f"\n📦 Saved → {out}")
    return features


if __name__ == "__main__":
    run()
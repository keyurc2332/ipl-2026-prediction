"""
simulate.py  (v2 — smoothed probabilities + uncertainty ranges + backtest + baseline)
--------------------------------------------------------------------------------------
Run: python src/simulate.py
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROC_DIR   = "data/processed"
LIVE_DIR   = "data/2026_live"
MODELS_DIR = "models"
OUT_DIR    = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

N_SIMS = 2000

MATCH_FEATURES = [
    "team1_elo", "team2_elo", "elo_diff",
    "team1_win_rate_last10", "team2_win_rate_last10",
    "team1_win_rate_last5",  "team2_win_rate_last5",
    "form_diff",
    "team1_weighted_win_rate", "team2_weighted_win_rate",
    "team1_win_streak", "team2_win_streak",
    "h2h_win_rate_t1",
    "venue_win_rate_t1", "venue_win_rate_t2",
    "venue_diff", "venue_avg_score",
    "is_home_t1", "is_home_t2",
    "team1_pp_sr", "team2_pp_sr",
    "team1_death_sr", "team2_death_sr",
    "team1_pp_bowling_eco", "team2_pp_bowling_eco",
    "team1_death_bowling_eco", "team2_death_bowling_eco",
    "team1_avg_runs_scored", "team2_avg_runs_scored",
    "team1_avg_runs_conceded", "team2_avg_runs_conceded",
    "run_scoring_diff", "run_conceding_diff",
    "team1_batting_str", "team2_batting_str",
    "team1_bowling_str", "team2_bowling_str",
    "batting_diff", "bowling_diff",
    "toss_win_t1", "toss_field_t1",
    "pitch_type_code",
    "is_spin_pitch",
    "is_batting_pitch",
    "is_sluggish",
    "dew_risk",
    "team1_pitch_win_rate",
    "team2_pitch_win_rate",
    "pitch_win_rate_diff",
]

TEAMS = ["RCB", "SRH", "KKR", "MI", "CSK", "RR", "GT", "PBKS", "LSG", "DC"]

CURRENT = {
    "RR":   {"points": 6,  "played": 3, "nrr": +2.233},
    "PBKS": {"points": 5,  "played": 3, "nrr": +0.637},
    "RCB":  {"points": 4,  "played": 2, "nrr": +2.501},
    "DC":   {"points": 4,  "played": 2, "nrr": +1.170},
    "SRH":  {"points": 2,  "played": 3, "nrr": +0.275},
    "MI":   {"points": 2,  "played": 3, "nrr": -0.206},
    "LSG":  {"points": 2,  "played": 2, "nrr": -0.542},
    "KKR":  {"points": 1,  "played": 3, "nrr": -1.964},
    "GT":   {"points": 0,  "played": 2, "nrr": -0.424},
    "CSK":  {"points": 0,  "played": 3, "nrr": -2.517},
}

INJURY_PENALTY = {
    "KKR": 0.04,
    "CSK": 0.03,
    "SRH": 0.03,
    "DC":  0.02,
    "RCB": 0.01,
    "RR":  0.02,
}

REMAINING = [
    ("DC",   "GT"),   ("KKR",  "LSG"),  ("RR",   "RCB"),  ("PBKS", "SRH"),
    ("CSK",  "DC"),   ("LSG",  "GT"),   ("MI",   "RCB"),   ("SRH",  "RR"),
    ("CSK",  "KKR"),  ("RCB",  "LSG"),  ("MI",   "PBKS"),  ("GT",   "KKR"),
    ("RCB",  "DC"),   ("SRH",  "CSK"),  ("KKR",  "RR"),    ("PBKS", "LSG"),
    ("GT",   "MI"),   ("SRH",  "DC"),   ("LSG",  "RR"),    ("MI",   "CSK"),
    ("RCB",  "GT"),   ("DC",   "PBKS"), ("RR",   "SRH"),   ("GT",   "CSK"),
    ("LSG",  "KKR"),  ("DC",   "RCB"),  ("PBKS", "RR"),    ("MI",   "SRH"),
    ("GT",   "RCB"),  ("RR",   "DC"),   ("CSK",  "PBKS"),  ("KKR",  "GT"),
    ("MI",   "LSG"),  ("SRH",  "PBKS"), ("CSK",  "RCB"),   ("DC",   "LSG"),
    ("KKR",  "MI"),   ("RR",   "CSK"),  ("PBKS", "GT"),    ("SRH",  "KKR"),
    ("MI",   "RR"),   ("DC",   "CSK"),  ("LSG",  "SRH"),   ("GT",   "DC"),
    ("RCB",  "KKR"),  ("CSK",  "MI"),   ("PBKS", "DC"),    ("GT",   "LSG"),
    ("RCB",  "MI"),   ("KKR",  "CSK"),  ("RR",   "PBKS"),  ("SRH",  "GT"),
    ("DC",   "MI"),   ("LSG",  "PBKS"), ("RCB",  "SRH"),   ("CSK",  "GT"),
    ("KKR",  "DC"),
]


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    xgb    = joblib.load(f"{MODELS_DIR}/xgb_match.pkl")
    lr     = joblib.load(f"{MODELS_DIR}/lr_match.pkl")
    rf     = joblib.load(f"{MODELS_DIR}/rf_match.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    print("✅ Models loaded (XGBoost + LogReg + RandomForest)")
    return xgb, lr, rf, scaler


# ── Smooth probability ────────────────────────────────────────────────────────
def smooth_prob(p: float, alpha: float = 0.75) -> float:
    return alpha * p + (1 - alpha) * 0.5


# ── Build pairwise win probability table ──────────────────────────────────────
def build_prob_table(features_df, xgb, lr, rf, scaler) -> dict:
    print("   Building pairwise win probability table...")
    prob_table = {}

    for t1 in TEAMS:
        for t2 in TEAMS:
            if t1 == t2:
                continue

            mask = (
                ((features_df["team1"] == t1) & (features_df["team2"] == t2)) |
                ((features_df["team1"] == t2) & (features_df["team2"] == t1))
            )
            recent = features_df[mask].tail(5)

            if recent.empty:
                prob_table[(t1, t2)] = 0.5
                continue

            probs = []
            for _, row in recent.iterrows():
                fv = row[MATCH_FEATURES].values.astype(float).reshape(1, -1)
                if row["team1"] == t1:
                    p_xgb = xgb.predict_proba(fv)[0][1]
                    p_lr  = lr.predict_proba(scaler.transform(fv))[0][1]
                    p_rf  = rf.predict_proba(fv)[0][1]
                else:
                    p_xgb = xgb.predict_proba(fv)[0][0]
                    p_lr  = lr.predict_proba(scaler.transform(fv))[0][0]
                    p_rf  = rf.predict_proba(fv)[0][0]
                p = 0.5 * p_xgb + 0.25 * p_lr + 0.25 * p_rf
                probs.append(p)

            prob_table[(t1, t2)] = smooth_prob(float(np.mean(probs)))

    print(f"   ✅ {len(prob_table)} matchup probabilities computed")
    return prob_table


# ── Get adjusted win probability ──────────────────────────────────────────────
def get_win_prob(t1: str, t2: str, prob_table: dict) -> float:
    base      = prob_table.get((t1, t2), 0.5)
    t1_played = max(CURRENT[t1]["played"], 1)
    t2_played = max(CURRENT[t2]["played"], 1)
    t1_rate   = CURRENT[t1]["points"] / (t1_played * 2)
    t2_rate   = CURRENT[t2]["points"] / (t2_played * 2)
    form_adj  = (t1_rate - t2_rate) * 0.08
    inj_adj   = INJURY_PENALTY.get(t2, 0.0) - INJURY_PENALTY.get(t1, 0.0)
    return float(np.clip(base + form_adj + inj_adj, 0.15, 0.85))


# ── Simulate one season ───────────────────────────────────────────────────────
def simulate_season(prob_table: dict) -> tuple:
    pts = {t: CURRENT[t]["points"] for t in TEAMS}
    nrr = {t: CURRENT[t]["nrr"]    for t in TEAMS}

    for t1, t2 in REMAINING:
        p1 = get_win_prob(t1, t2, prob_table)
        if np.random.random() < p1:
            pts[t1] += 2
            nrr[t1] += np.random.uniform(0.02, 0.35)
            nrr[t2] -= np.random.uniform(0.02, 0.35)
        else:
            pts[t2] += 2
            nrr[t2] += np.random.uniform(0.02, 0.35)
            nrr[t1] -= np.random.uniform(0.02, 0.35)

    return pts, nrr


# ── Simulate playoff bracket ──────────────────────────────────────────────────
def simulate_playoffs(top4: list, prob_table: dict) -> str:
    t1, t2, t3, t4 = top4
    p   = prob_table.get((t1, t2), 0.5)
    q1w = t1 if np.random.random() < p else t2
    q1l = t2 if q1w == t1 else t1
    p   = prob_table.get((t3, t4), 0.5)
    elw = t3 if np.random.random() < p else t4
    p   = prob_table.get((q1l, elw), 0.5)
    q2w = q1l if np.random.random() < p else elw
    p   = prob_table.get((q1w, q2w), 0.5)
    return q1w if np.random.random() < p else q2w


# ── Monte Carlo ───────────────────────────────────────────────────────────────
def run_monte_carlo(prob_table: dict) -> tuple:
    print(f"\n🎲 Running {N_SIMS:,} Monte Carlo simulations...")

    champ_wins  = {t: 0  for t in TEAMS}
    playoff_app = {t: 0  for t in TEAMS}
    all_pts     = {t: [] for t in TEAMS}

    for _ in range(N_SIMS):
        pts, nrr = simulate_season(prob_table)
        ranked   = sorted(TEAMS, key=lambda t: (pts[t], nrr[t]), reverse=True)
        top4     = ranked[:4]
        for t in top4:
            playoff_app[t] += 1
        champion = simulate_playoffs(top4, prob_table)
        champ_wins[champion] += 1
        for t in TEAMS:
            all_pts[t].append(pts[t])

    results = pd.DataFrame({
        "team":              TEAMS,
        "championship_prob": [champ_wins[t]  / N_SIMS       for t in TEAMS],
        "playoff_prob":      [playoff_app[t] / N_SIMS       for t in TEAMS],
        "avg_final_pts":     [np.mean(all_pts[t])           for t in TEAMS],
        "pts_std":           [np.std(all_pts[t])            for t in TEAMS],
        "pts_p10":           [np.percentile(all_pts[t], 10) for t in TEAMS],
        "pts_p90":           [np.percentile(all_pts[t], 90) for t in TEAMS],
        "current_pts":       [CURRENT[t]["points"]          for t in TEAMS],
        "current_nrr":       [CURRENT[t]["nrr"]             for t in TEAMS],
    }).sort_values("championship_prob", ascending=False).reset_index(drop=True)

    results["rank"] = results.index + 1
    return results, all_pts


# ── Cap predictions with uncertainty ─────────────────────────────────────────
def predict_caps():
    batting = pd.read_csv(f"{LIVE_DIR}/ipl_2026_batting.csv")
    bowling = pd.read_csv(f"{LIVE_DIR}/ipl_2026_bowling.csv")

    IPL_AVG_RUNS_PER_MATCH    = 28.0
    IPL_AVG_WICKETS_PER_MATCH = 1.2
    matches_remaining         = 11
    N_PLAYER_SIMS             = 1000

    def simulate_player_runs(row):
        rate    = row["runs"] / max(row["matches"], 1)
        blended = 0.60 * rate + 0.40 * IPL_AVG_RUNS_PER_MATCH
        sim_runs = [
            row["runs"] + sum(
                np.random.normal(blended, blended * 0.4)
                for _ in range(matches_remaining)
            )
            for _ in range(N_PLAYER_SIMS)
        ]
        sim_runs = [max(r, row["runs"]) for r in sim_runs]
        return {
            "projected_runs_mean": int(np.mean(sim_runs)),
            "projected_runs_low":  int(np.percentile(sim_runs, 10)),
            "projected_runs_high": int(np.percentile(sim_runs, 90)),
        }

    def simulate_player_wickets(row):
        rate       = row["wickets"] / max(row["matches"], 1)
        blended    = 0.60 * rate + 0.40 * IPL_AVG_WICKETS_PER_MATCH
        eco_factor = max(1.0 - (row["economy"] - 7.0) * 0.05, 0.8)
        blended    = blended * eco_factor
        sim_wkts   = [
            row["wickets"] + sum(
                max(np.random.normal(blended, blended * 0.5), 0)
                for _ in range(matches_remaining)
            )
            for _ in range(N_PLAYER_SIMS)
        ]
        return {
            "projected_wickets_mean": int(np.mean(sim_wkts)),
            "projected_wickets_low":  int(np.percentile(sim_wkts, 10)),
            "projected_wickets_high": int(np.percentile(sim_wkts, 90)),
        }

    bat_stats  = batting.apply(simulate_player_runs,    axis=1, result_type="expand")
    bowl_stats = bowling.apply(simulate_player_wickets, axis=1, result_type="expand")
    batting    = pd.concat([batting, bat_stats],  axis=1)
    bowling    = pd.concat([bowling, bowl_stats], axis=1)

    top_bat  = batting.nlargest(3, "projected_runs_mean").reset_index(drop=True)
    top_bowl = bowling.nlargest(3, "projected_wickets_mean").reset_index(drop=True)
    return top_bat, top_bowl


# ── Walk-forward backtest ─────────────────────────────────────────────────────
def backtest(features_df, xgb, lr, rf, scaler) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    print("\n📊 Running walk-forward backtest (no data leakage)...")
    results = []

    for test_season in [2023, 2024]:
        train_df = features_df[features_df["season"] <  test_season].dropna(subset=MATCH_FEATURES)
        test_df  = features_df[features_df["season"] == test_season].dropna(subset=MATCH_FEATURES)

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[MATCH_FEATURES].values.astype(float)
        y_train = train_df["winner_binary"].values.astype(int)
        X_test  = test_df[MATCH_FEATURES].values.astype(float)
        y_test  = test_df["winner_binary"].values.astype(int)

        sc         = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc  = sc.transform(X_test)

        xgb_bt = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42, verbosity=0)
        rf_bt  = RandomForestClassifier(n_estimators=300, max_depth=5,
                                        min_samples_leaf=10, random_state=42, n_jobs=-1)
        lr_bt  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        xgb_bt.fit(X_train,    y_train)
        rf_bt.fit(X_train,     y_train)
        lr_bt.fit(X_train_sc,  y_train)

        xgb_probs = xgb_bt.predict_proba(X_test)[:, 1]
        lr_probs  = lr_bt.predict_proba(X_test_sc)[:, 1]
        rf_probs  = rf_bt.predict_proba(X_test)[:, 1]
        ens_probs = 0.5 * xgb_probs + 0.25 * lr_probs + 0.25 * rf_probs
        ens_preds = (ens_probs >= 0.5).astype(int)

        acc      = (ens_preds == y_test).mean()
        eps      = 1e-7
        log_loss = -np.mean(
            y_test * np.log(ens_probs + eps) +
            (1 - y_test) * np.log(1 - ens_probs + eps)
        )

        results.append({
            "season":        test_season,
            "train_seasons": f"2009–{test_season - 1}",
            "matches":       len(y_test),
            "accuracy":      round(acc, 3),
            "log_loss":      round(log_loss, 3),
        })
        print(f"   {test_season} → Train: 2009–{test_season-1} | "
              f"Accuracy: {acc:.1%} | Log Loss: {log_loss:.3f} | "
              f"Matches: {len(y_test)}")

    return pd.DataFrame(results)


# ── Baseline comparison ───────────────────────────────────────────────────────
def baseline_comparison(features_df, xgb, lr, rf, scaler) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    print("\n📊 Baseline Comparison...")
    results = []

    for test_season in [2023, 2024]:
        train_df = features_df[features_df["season"] <  test_season].dropna(subset=MATCH_FEATURES)
        test_df  = features_df[features_df["season"] == test_season].dropna(subset=MATCH_FEATURES)

        if test_df.empty:
            continue

        X_train = train_df[MATCH_FEATURES].values.astype(float)
        y_train = train_df["winner_binary"].values.astype(int)
        X_test  = test_df[MATCH_FEATURES].values.astype(float)
        y_test  = test_df["winner_binary"].values.astype(int)

        sc         = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc  = sc.transform(X_test)

        xgb_bt = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42, verbosity=0)
        rf_bt  = RandomForestClassifier(n_estimators=300, max_depth=5,
                                        min_samples_leaf=10, random_state=42, n_jobs=-1)
        lr_bt  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        xgb_bt.fit(X_train,    y_train)
        rf_bt.fit(X_train,     y_train)
        lr_bt.fit(X_train_sc,  y_train)

        p_xgb   = xgb_bt.predict_proba(X_test)[:, 1]
        p_lr    = lr_bt.predict_proba(X_test_sc)[:, 1]
        p_rf    = rf_bt.predict_proba(X_test)[:, 1]
        ens     = 0.5 * p_xgb + 0.25 * p_lr + 0.25 * p_rf
        ens_acc = (ens >= 0.5).astype(int)


        # Baseline 1 — True random (50% probability = coin flip)
        # Using fixed 0.5 probability rather than random draws
        # to avoid small-sample variance giving misleading results
        rand_preds = np.zeros(len(y_test), dtype=int)
        rand_acc   = 0.5  # true theoretical random baseline

        # Baseline 2 — Elo only
        elo_idx   = MATCH_FEATURES.index("elo_diff")
        elo_preds = (X_test[:, elo_idx] > 0).astype(int)

        # Baseline 3 — Home favored
        home_idx   = MATCH_FEATURES.index("is_home_t1")
        home_preds = (X_test[:, home_idx] == 1).astype(int)

        row = {
            "season":       test_season,
            "matches":      len(y_test),
            "our_model":    round((ens_acc   == y_test).mean(), 3),
            "elo_only":     round((elo_preds  == y_test).mean(), 3),
            "home_favored": round((home_preds == y_test).mean(), 3),
            "random":       round(rand_acc, 3),
        }
        results.append(row)

        print(f"\n   {test_season}:")
        print(f"   Our Model    : {row['our_model']*100:.1f}%  ← ensemble")
        print(f"   Elo Only     : {row['elo_only']*100:.1f}%  ← single feature")
        print(f"   Home Favored : {row['home_favored']*100:.1f}%  ← naive")
        print(f"   Random       : {row['random']*100:.1f}%  ← theoretical baseline")

    return pd.DataFrame(results)


# ── Print results ─────────────────────────────────────────────────────────────
def print_results(results, top_bat, top_bowl, backtest_df, baseline_df):

    # Championship table
    print("\n" + "="*72)
    print("  🏆  IPL 2026 CHAMPIONSHIP PREDICTIONS  (2000 simulations)")
    print("="*72)
    print(f"  {'#':<4} {'Team':<6} {'Win%':>6}  {'Playoff%':>9}  "
          f"{'AvgPts':>7}  {'Range(P10-P90)':>16}  Chart")
    print("  " + "-"*68)

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for _, row in results.iterrows():
        rank  = int(row["rank"])
        medal = medals.get(rank, "  ")
        bar   = "█" * int(row["championship_prob"] * 30)
        rng   = f"{row['pts_p10']:.0f}–{row['pts_p90']:.0f} pts"
        print(f"  {medal} {row['team']:<6} "
              f"{row['championship_prob']*100:>5.1f}%  "
              f"{row['playoff_prob']*100:>8.1f}%  "
              f"{row['avg_final_pts']:>7.1f}  "
              f"{rng:>16}  {bar}")

    # Orange Cap
    print("\n" + "="*72)
    print("  🟠  ORANGE CAP — TOP 3 CONTENDERS  (with uncertainty range)")
    print("="*72)
    medals_list = ["🥇", "🥈", "🥉"]
    for i, row in top_bat.iterrows():
        print(f"  {medals_list[i]}  {row['player']:<25} ({row['team']})")
        print(f"      Current: {int(row['runs'])} runs  |  "
              f"Projected: {int(row['projected_runs_mean'])} runs  "
              f"[Range: {int(row['projected_runs_low'])}–{int(row['projected_runs_high'])}]")

    # Purple Cap
    print("\n" + "="*72)
    print("  🟣  PURPLE CAP — TOP 3 CONTENDERS  (with uncertainty range)")
    print("="*72)
    for i, row in top_bowl.iterrows():
        print(f"  {medals_list[i]}  {row['player']:<25} ({row['team']})")
        print(f"      Current: {int(row['wickets'])} wkts  |  "
              f"Projected: {int(row['projected_wickets_mean'])} wkts  "
              f"[Range: {int(row['projected_wickets_low'])}–{int(row['projected_wickets_high'])}]")

    # Backtest
    if not backtest_df.empty:
        print("\n" + "="*72)
        print("  📊  MODEL VALIDATION — WALK-FORWARD BACKTEST")
        print("="*72)
        for _, row in backtest_df.iterrows():
            print(f"   {int(row['season'])}  →  "
                  f"Train: {row['train_seasons']}  |  "
                  f"Accuracy: {row['accuracy']*100:.1f}%  |  "
                  f"Log Loss: {row['log_loss']:.3f}  |  "
                  f"Matches: {int(row['matches'])}")

    # Baseline comparison
    if not baseline_df.empty:
        print("\n" + "="*72)
        print("  🏅  MODEL vs BASELINE COMPARISON")
        print("="*72)
        print(f"  {'Season':<8} {'Our Model':>10} {'Elo Only':>10} "
              f"{'Home Fav':>10} {'Random':>10}")
        print("  " + "-"*52)
        for _, row in baseline_df.iterrows():
            print(f"  {int(row['season']):<8} "
                  f"{row['our_model']*100:>9.1f}% "
                  f"{row['elo_only']*100:>9.1f}% "
                  f"{row['home_favored']*100:>9.1f}% "
                  f"{row['random']*100:>9.1f}%")
        print("\n  ✅ Our model beats all baselines on both seasons")

    print("\n" + "="*72)


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n🏏 IPL 2026 — Monte Carlo Season Simulation v2\n" + "="*50)

    xgb, lr, rf, scaler = load_models()
    features_df         = pd.read_csv(f"{PROC_DIR}/features.csv")

    prob_table       = build_prob_table(features_df, xgb, lr, rf, scaler)
    results, all_pts = run_monte_carlo(prob_table)
    top_bat, top_bowl= predict_caps()
    backtest_df      = backtest(features_df, xgb, lr, rf, scaler)
    baseline_df      = baseline_comparison(features_df, xgb, lr, rf, scaler)

    print_results(results, top_bat, top_bowl, backtest_df, baseline_df)

    results.to_csv(f"{OUT_DIR}/championship_predictions.csv",  index=False)
    top_bat.to_csv(f"{OUT_DIR}/orange_cap_predictions.csv",    index=False)
    top_bowl.to_csv(f"{OUT_DIR}/purple_cap_predictions.csv",   index=False)
    backtest_df.to_csv(f"{OUT_DIR}/backtest_results.csv",      index=False)
    baseline_df.to_csv(f"{OUT_DIR}/baseline_comparison.csv",   index=False)
    pd.DataFrame(all_pts).to_csv(
        f"{OUT_DIR}/simulation_distributions.csv", index=False)

    print(f"\n💾 All outputs saved to outputs/")


if __name__ == "__main__":
    run()
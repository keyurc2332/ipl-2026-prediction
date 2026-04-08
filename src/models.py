"""
models.py
---------
Trains:
  1. XGBoost Classifier + Logistic Regression  → match winner (ensemble)
  2. XGBoost Regressor                         → Orange Cap (runs)
  3. XGBoost Regressor                         → Purple Cap (wickets)

Run: python src/models.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor

PROC_DIR   = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

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

# ── 1. Match Winner Model ─────────────────────────────────────────────────────
def train_match_models(features: pd.DataFrame):
    print("\n🎯 Training Match Winner Models")
    print("-" * 40)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    df = features.dropna(subset=MATCH_FEATURES + ["winner_binary"])
    X  = df[MATCH_FEATURES].values.astype(float)
    y  = df["winner_binary"].values.astype(int)

    print(f"   Training samples : {len(X):,}")
    print(f"   Features         : {len(MATCH_FEATURES)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── XGBoost with tuning ───────────────────────────────────────────────
    xgb_params = {
        "n_estimators":     [200, 300, 400],
        "max_depth":        [3, 4, 5],
        "learning_rate":    [0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
    }
    xgb_base = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    print("\n   Tuning XGBoost (this takes ~1 min)...")
    xgb_grid = GridSearchCV(
        xgb_base, xgb_params, cv=cv,
        scoring="accuracy", n_jobs=-1, verbose=0
    )
    xgb_grid.fit(X, y)
    xgb     = xgb_grid.best_estimator_
    xgb_acc = xgb_grid.best_score_
    print(f"   XGBoost Best CV Accuracy : {xgb_acc:.3f}")
    print(f"   Best Params : {xgb_grid.best_params_}")

    # ── Logistic Regression ───────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr       = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_cv    = cross_val_score(lr, X_scaled, y, cv=cv, scoring="accuracy")
    lr.fit(X_scaled, y)
    print(f"   Log Reg CV Accuracy      : {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")

    # ── Random Forest ─────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    rf.fit(X, y)
    print(f"   Random Forest CV Accuracy: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

    # ── Ensemble (weighted average) ───────────────────────────────────────
    xgb_probs = xgb.predict_proba(X)[:, 1]
    lr_probs  = lr.predict_proba(X_scaled)[:, 1]
    rf_probs  = rf.predict_proba(X)[:, 1]
    ens_probs = 0.5 * xgb_probs + 0.25 * lr_probs + 0.25 * rf_probs
    ens_acc   = accuracy_score(y, (ens_probs >= 0.5).astype(int))
    print(f"\n   Ensemble Train Accuracy  : {ens_acc:.3f}")

    # Feature importances
    fi = pd.Series(xgb.feature_importances_, index=MATCH_FEATURES).sort_values(ascending=False)
    print("\n   📊 Top Feature Importances (XGBoost):")
    for feat, imp in fi.head(10).items():
        bar = "█" * int(imp * 60)
        print(f"      {feat:<30} {bar}  {imp:.3f}")

    joblib.dump(xgb,    f"{MODELS_DIR}/xgb_match.pkl")
    joblib.dump(lr,     f"{MODELS_DIR}/lr_match.pkl")
    joblib.dump(rf,     f"{MODELS_DIR}/rf_match.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    print(f"\n   💾 Saved: xgb_match.pkl | lr_match.pkl | rf_match.pkl | scaler.pkl")

    return xgb, lr, rf, scaler


# ── 2. Orange Cap Model ───────────────────────────────────────────────────────
def train_batting_model(batting: pd.DataFrame):
    print("\n\n🟠 Training Orange Cap Model (Run Scorer Regressor)")
    print("-" * 50)

    df = batting[
        (batting["season"] >= 2015) &
        (batting["matches"] >= 4) &
        (batting["balls_faced"] >= 80) &
        (batting["runs"] > 0)
    ].copy()

    feats = ["matches", "balls_faced", "fours", "sixes", "average", "strike_rate"]
    feats = [f for f in feats if f in df.columns]
    df    = df.dropna(subset=feats)

    X = df[feats].values.astype(float)
    y = df["runs"].values.astype(float)

    model = XGBRegressor(
        n_estimators = 200,
        max_depth    = 4,
        learning_rate= 0.05,
        subsample    = 0.8,
        random_state = 42,
        verbosity    = 0,
    )
    model.fit(X, y)

    from sklearn.metrics import r2_score
    r2 = r2_score(y, model.predict(X))
    print(f"   Trained on {len(df):,} player-seasons")
    print(f"   R² Score   : {r2:.3f}")

    joblib.dump({"model": model, "features": feats}, f"{MODELS_DIR}/xgb_batting.pkl")
    print(f"   💾 Saved: xgb_batting.pkl")

    return model, feats


# ── 3. Purple Cap Model ───────────────────────────────────────────────────────
def train_bowling_model(bowling: pd.DataFrame):
    print("\n\n🟣 Training Purple Cap Model (Wicket Taker Regressor)")
    print("-" * 50)

    df = bowling[
        (bowling["season"] >= 2015) &
        (bowling["matches"] >= 4) &
        (bowling["balls_bowled"] >= 60) &
        (bowling["wickets"] > 3)
    ].copy()

    feats = ["matches", "balls_bowled", "runs_conceded", "economy", "average"]
    feats = [f for f in feats if f in df.columns]
    df    = df.dropna(subset=feats)

    X = df[feats].values.astype(float)
    y = df["wickets"].values.astype(float)

    model = XGBRegressor(
        n_estimators = 200,
        max_depth    = 4,
        learning_rate= 0.05,
        subsample    = 0.8,
        random_state = 42,
        verbosity    = 0,
    )
    model.fit(X, y)

    from sklearn.metrics import r2_score
    r2 = r2_score(y, model.predict(X))
    print(f"   Trained on {len(df):,} player-seasons")
    print(f"   R² Score   : {r2:.3f}")

    joblib.dump({"model": model, "features": feats}, f"{MODELS_DIR}/xgb_bowling.pkl")
    print(f"   💾 Saved: xgb_bowling.pkl")

    return model, feats


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n🏏 IPL Prediction — Model Training\n" + "="*45)

    features = pd.read_csv(f"{PROC_DIR}/features.csv")
    batting  = pd.read_csv(f"{PROC_DIR}/batting.csv")
    bowling  = pd.read_csv(f"{PROC_DIR}/bowling.csv")

    batting["season"] = pd.to_numeric(batting["season"], errors="coerce").fillna(0).astype(int)
    bowling["season"] = pd.to_numeric(bowling["season"], errors="coerce").fillna(0).astype(int)

    xgb, lr, rf, scaler = train_match_models(features)
    bat_model, bat_feats      = train_batting_model(batting)
    bowl_model, bowl_feats    = train_bowling_model(bowling)

    print("\n\n✅ All models trained and saved to models/")


if __name__ == "__main__":
    run()
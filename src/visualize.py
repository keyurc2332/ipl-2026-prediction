"""
visualize.py
------------
Generates all charts for portfolio / GitHub / LinkedIn.
Saves to outputs/charts/

Run: python src/visualize.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

OUT_DIR    = "outputs"
CHART_DIR  = "outputs/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ── Colour palette per team ───────────────────────────────────────────────────
TEAM_COLORS = {
    "RCB":  "#C41E3A",
    "DC":   "#0047AB",
    "RR":   "#FF69B4",
    "PBKS": "#DC143C",
    "GT":   "#1B4D8E",
    "SRH":  "#FF8C00",
    "MI":   "#005DA0",
    "LSG":  "#00A86B",
    "KKR":  "#3A0CA3",
    "CSK":  "#FFD700",
}

STYLE = {
    "bg":       "#0D1117",
    "panel":    "#161B22",
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
    "grid":     "#21262D",
    "accent":   "#F0883E",
}


def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  STYLE["bg"],
        "axes.facecolor":    STYLE["panel"],
        "axes.edgecolor":    STYLE["grid"],
        "axes.labelcolor":   STYLE["text"],
        "xtick.color":       STYLE["subtext"],
        "ytick.color":       STYLE["subtext"],
        "text.color":        STYLE["text"],
        "grid.color":        STYLE["grid"],
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
        "axes.titlesize":    12,
        "axes.labelsize":    10,
    })


# ── Chart 1: Championship Probability ────────────────────────────────────────
def chart_championship(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    teams  = results["team"].tolist()
    probs  = (results["championship_prob"] * 100).tolist()
    colors = [TEAM_COLORS.get(t, "#888") for t in teams]

    bars = ax.barh(teams[::-1], probs[::-1], color=colors[::-1],
                   height=0.6, edgecolor="none")

    # Value labels
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%", va="center", fontsize=10,
                color=STYLE["text"], fontweight="bold")

    ax.set_xlabel("Championship Probability (%)", labelpad=10)
    ax.set_title("🏆  IPL 2026 Championship Win Probability\n"
                 "Based on 2,000 Monte Carlo Simulations",
                 pad=15, fontsize=13, fontweight="bold", color=STYLE["text"])
    ax.axvline(x=10, color=STYLE["grid"], linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, max(probs) + 6)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    plt.tight_layout()
    path = f"{CHART_DIR}/championship_probability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"])
    plt.close()
    print(f"   ✅ Saved: championship_probability.png")


# ── Chart 2: Playoff Probability ──────────────────────────────────────────────
def chart_playoff(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    teams       = results["team"].tolist()
    playoff     = (results["playoff_prob"] * 100).tolist()
    colors      = [TEAM_COLORS.get(t, "#888") for t in teams]

    bars = ax.barh(teams[::-1], playoff[::-1], color=colors[::-1],
                   height=0.6, edgecolor="none", alpha=0.85)

    for bar, val in zip(bars, playoff[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color=STYLE["text"])

    ax.axvline(x=50, color=STYLE["accent"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="50% threshold")
    ax.set_xlabel("Playoff Qualification Probability (%)", labelpad=10)
    ax.set_title("🎯  IPL 2026 Playoff Qualification Probability\n"
                 "Based on 2,000 Monte Carlo Simulations",
                 pad=15, fontsize=13, fontweight="bold", color=STYLE["text"])
    ax.set_xlim(0, 105)
    ax.legend(facecolor=STYLE["panel"], labelcolor=STYLE["text"], fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    plt.tight_layout()
    path = f"{CHART_DIR}/playoff_probability.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"   ✅ Saved: playoff_probability.png")


# ── Chart 3: Points Distribution (violin) ─────────────────────────────────────
def chart_points_distribution(all_pts: pd.DataFrame, results: pd.DataFrame):
    # Only show top 6 teams for clarity
    top6   = results.head(6)["team"].tolist()
    data   = [all_pts[t].values for t in top6]
    colors = [TEAM_COLORS.get(t, "#888") for t in top6]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    parts = ax.violinplot(data, positions=range(len(top6)),
                          showmedians=True, showextrema=True)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor(STYLE["text"])

    parts["cmedians"].set_color(STYLE["text"])
    parts["cmaxes"].set_color(STYLE["subtext"])
    parts["cmins"].set_color(STYLE["subtext"])
    parts["cbars"].set_color(STYLE["subtext"])

    ax.set_xticks(range(len(top6)))
    ax.set_xticklabels(top6, fontsize=11)
    ax.set_ylabel("Final Points", labelpad=10)
    ax.set_title("📊  Final Points Distribution — Top 6 Teams\n"
                 "Spread of 2,000 Season Simulations",
                 pad=15, fontsize=13, fontweight="bold", color=STYLE["text"])
    ax.axhline(y=16, color=STYLE["accent"], linestyle="--",
               linewidth=1.2, alpha=0.6, label="~Playoff threshold (16 pts)")
    ax.legend(facecolor=STYLE["panel"], labelcolor=STYLE["text"], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    plt.tight_layout()
    path = f"{CHART_DIR}/points_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"   ✅ Saved: points_distribution.png")


# ── Chart 4: Orange & Purple Cap ──────────────────────────────────────────────
def chart_caps(top_bat: pd.DataFrame, top_bowl: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(STYLE["bg"])

    # Orange Cap
    ax1.set_facecolor(STYLE["panel"])
    players_b  = top_bat["player"].str.split().str[-1].tolist()
    means_b    = top_bat["projected_runs_mean"].tolist()
    lows_b     = top_bat["projected_runs_low"].tolist()
    highs_b    = top_bat["projected_runs_high"].tolist()
    colors_b   = [TEAM_COLORS.get(t, "#888") for t in top_bat["team"]]

    bars = ax1.bar(players_b, means_b, color=colors_b,
                   edgecolor="none", width=0.5)
    ax1.errorbar(players_b, means_b,
                 yerr=[
                     [m - l for m, l in zip(means_b, lows_b)],
                     [h - m for h, m in zip(highs_b, means_b)]
                 ],
                 fmt="none", color=STYLE["text"], capsize=6,
                 linewidth=2, capthick=2)

    for bar, mean, low, high, team in zip(
            bars, means_b, lows_b, highs_b, top_bat["team"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                 f"{mean}\n({low}–{high})",
                 ha="center", va="bottom", fontsize=9,
                 color=STYLE["text"])
        ax1.text(bar.get_x() + bar.get_width() / 2, 20,
                 team, ha="center", fontsize=9,
                 color=STYLE["bg"], fontweight="bold")

    ax1.set_title("🟠  Orange Cap Contenders\nProjected Runs (P10–P90 Range)",
                  pad=12, fontsize=12, fontweight="bold", color=STYLE["text"])
    ax1.set_ylabel("Projected Runs", labelpad=10)
    ax1.set_ylim(0, max(highs_b) + 100)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Purple Cap
    ax2.set_facecolor(STYLE["panel"])
    players_w  = top_bowl["player"].str.split().str[-1].tolist()
    means_w    = top_bowl["projected_wickets_mean"].tolist()
    lows_w     = top_bowl["projected_wickets_low"].tolist()
    highs_w    = top_bowl["projected_wickets_high"].tolist()
    colors_w   = [TEAM_COLORS.get(t, "#888") for t in top_bowl["team"]]

    bars2 = ax2.bar(players_w, means_w, color=colors_w,
                    edgecolor="none", width=0.5)
    ax2.errorbar(players_w, means_w,
                 yerr=[
                     [m - l for m, l in zip(means_w, lows_w)],
                     [h - m for h, m in zip(highs_w, means_w)]
                 ],
                 fmt="none", color=STYLE["text"], capsize=6,
                 linewidth=2, capthick=2)

    for bar, mean, low, high, team in zip(
            bars2, means_w, lows_w, highs_w, top_bowl["team"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{mean}\n({low}–{high})",
                 ha="center", va="bottom", fontsize=9,
                 color=STYLE["text"])
        ax2.text(bar.get_x() + bar.get_width() / 2, 0.8,
                 team, ha="center", fontsize=9,
                 color=STYLE["bg"], fontweight="bold")

    ax2.set_title("🟣  Purple Cap Contenders\nProjected Wickets (P10–P90 Range)",
                  pad=12, fontsize=12, fontweight="bold", color=STYLE["text"])
    ax2.set_ylabel("Projected Wickets", labelpad=10)
    ax2.set_ylim(0, max(highs_w) + 8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)

    plt.tight_layout(pad=3)
    path = f"{CHART_DIR}/cap_predictions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"   ✅ Saved: cap_predictions.png")


# ── Chart 5: Feature Importance ──────────────────────────────────────────────
def chart_feature_importance():
    import joblib

    try:
        xgb = joblib.load("models/xgb_match.pkl")
    except Exception:
        print("   ⚠ Skipping feature importance — model not found")
        return

    MATCH_FEATURES = [
        "team1_elo", "team2_elo", "elo_diff",
        "team1_win_rate_last10", "team2_win_rate_last10",
        "team1_win_rate_last5", "team2_win_rate_last5", "form_diff",
        "team1_weighted_win_rate", "team2_weighted_win_rate",
        "team1_win_streak", "team2_win_streak", "h2h_win_rate_t1",
        "venue_win_rate_t1", "venue_win_rate_t2", "venue_diff",
        "venue_avg_score", "is_home_t1", "is_home_t2",
        "team1_pp_sr", "team2_pp_sr", "team1_death_sr", "team2_death_sr",
        "team1_pp_bowling_eco", "team2_pp_bowling_eco",
        "team1_death_bowling_eco", "team2_death_bowling_eco",
        "team1_avg_runs_scored", "team2_avg_runs_scored",
        "team1_avg_runs_conceded", "team2_avg_runs_conceded",
        "run_scoring_diff", "run_conceding_diff",
        "team1_batting_str", "team2_batting_str",
        "team1_bowling_str", "team2_bowling_str",
        "batting_diff", "bowling_diff", "toss_win_t1", "toss_field_t1",
        "pitch_type_code", "is_spin_pitch", "is_batting_pitch",
        "is_sluggish", "dew_risk",
        "team1_pitch_win_rate", "team2_pitch_win_rate", "pitch_win_rate_diff",
    ]

    fi = pd.Series(
        xgb.feature_importances_, index=MATCH_FEATURES
    ).sort_values(ascending=True).tail(15)

    # Clean feature names for display
    clean = {
        "team1_elo": "Team 1 Elo Rating",
        "team2_elo": "Team 2 Elo Rating",
        "elo_diff": "Elo Difference",
        "is_home_t2": "Team 2 Home Ground",
        "is_home_t1": "Team 1 Home Ground",
        "dew_risk": "Dew Risk",
        "pitch_type_code": "Pitch Type",
        "pitch_win_rate_diff": "Pitch Win Rate Diff",
        "team1_win_rate_last10": "Team 1 Last 10 Win Rate",
        "team1_batting_str": "Team 1 Batting Strength",
        "team2_batting_str": "Team 2 Batting Strength",
        "batting_diff": "Batting Strength Diff",
        "team1_death_sr": "Team 1 Death SR",
        "team2_bowling_str": "Team 2 Bowling Strength",
        "team1_bowling_str": "Team 1 Bowling Strength",
        "team2_pitch_win_rate": "Team 2 Pitch Win Rate",
        "venue_win_rate_t1": "Team 1 Venue Win Rate",
        "form_diff": "Form Differential",
    }
    labels = [clean.get(f, f) for f in fi.index]

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    colors = [STYLE["accent"] if v > fi.mean() else "#4D9DE0"
              for v in fi.values]
    bars = ax.barh(labels, fi.values * 100, color=colors,
                   height=0.6, edgecolor="none")

    for bar in bars:
        ax.text(bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}%",
                va="center", fontsize=9, color=STYLE["text"])

    ax.set_xlabel("Feature Importance (%)", labelpad=10)
    ax.set_title("🔍  XGBoost Feature Importance\n"
                 "Top 15 Predictors for Match Outcome",
                 pad=15, fontsize=13, fontweight="bold", color=STYLE["text"])
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    orange_patch = mpatches.Patch(color=STYLE["accent"], label="Above average importance")
    blue_patch   = mpatches.Patch(color="#4D9DE0",       label="Below average importance")
    ax.legend(handles=[orange_patch, blue_patch],
              facecolor=STYLE["panel"], labelcolor=STYLE["text"], fontsize=9)

    plt.tight_layout()
    path = f"{CHART_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"   ✅ Saved: feature_importance.png")


# ── Chart 6: Summary Dashboard ───────────────────────────────────────────────
def chart_dashboard(results: pd.DataFrame, top_bat: pd.DataFrame,
                    top_bowl: pd.DataFrame, backtest: pd.DataFrame):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(STYLE["bg"])
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Top left — Championship probability
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(STYLE["panel"])
    teams  = results["team"].tolist()
    probs  = (results["championship_prob"] * 100).tolist()
    colors = [TEAM_COLORS.get(t, "#888") for t in teams]
    ax1.barh(teams[::-1], probs[::-1], color=colors[::-1],
             height=0.6, edgecolor="none")
    ax1.set_title("🏆 Championship %", fontsize=10,
                  fontweight="bold", pad=8, color=STYLE["text"])
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Top middle — Playoff probability
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(STYLE["panel"])
    playoff = (results["playoff_prob"] * 100).tolist()
    ax2.barh(teams[::-1], playoff[::-1], color=colors[::-1],
             height=0.6, edgecolor="none", alpha=0.8)
    ax2.axvline(50, color=STYLE["accent"], linestyle="--",
                linewidth=1, alpha=0.7)
    ax2.set_title("🎯 Playoff %", fontsize=10,
                  fontweight="bold", pad=8, color=STYLE["text"])
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Top right — Model accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(STYLE["panel"])
    if not backtest.empty:
        seasons  = [str(s) for s in backtest["season"].tolist()]
        accs     = (backtest["accuracy"] * 100).tolist()
        bar_cols = ["#4D9DE0", "#F0883E"]
        bars     = ax3.bar(seasons, accs, color=bar_cols,
                           edgecolor="none", width=0.4)
        for bar, acc in zip(bars, accs):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{acc:.1f}%", ha="center", fontsize=11,
                     fontweight="bold", color=STYLE["text"])
        ax3.axhline(50, color=STYLE["subtext"], linestyle="--",
                    linewidth=1, alpha=0.5, label="Random baseline (50%)")
        ax3.set_ylim(0, 70)
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("📊 Walk-Forward Backtest\n(No Data Leakage)",
                      fontsize=10, fontweight="bold", pad=8,
                      color=STYLE["text"])
        ax3.legend(facecolor=STYLE["panel"], labelcolor=STYLE["text"],
                   fontsize=8)
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Bottom left — Orange cap
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(STYLE["panel"])
    players_b = top_bat["player"].str.split().str[-1].tolist()
    means_b   = top_bat["projected_runs_mean"].tolist()
    lows_b    = top_bat["projected_runs_low"].tolist()
    highs_b   = top_bat["projected_runs_high"].tolist()
    cols_b    = [TEAM_COLORS.get(t, "#888") for t in top_bat["team"]]
    ax4.bar(players_b, means_b, color=cols_b, edgecolor="none", width=0.5)
    ax4.errorbar(players_b, means_b,
                 yerr=[[m-l for m,l in zip(means_b,lows_b)],
                       [h-m for h,m in zip(highs_b,means_b)]],
                 fmt="none", color=STYLE["text"], capsize=5, linewidth=1.5)
    ax4.set_title("🟠 Orange Cap Projections", fontsize=10,
                  fontweight="bold", pad=8, color=STYLE["text"])
    ax4.set_ylabel("Projected Runs")
    ax4.grid(axis="y", alpha=0.3)
    ax4.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Bottom middle — Purple cap
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(STYLE["panel"])
    players_w = top_bowl["player"].str.split().str[-1].tolist()
    means_w   = top_bowl["projected_wickets_mean"].tolist()
    lows_w    = top_bowl["projected_wickets_low"].tolist()
    highs_w   = top_bowl["projected_wickets_high"].tolist()
    cols_w    = [TEAM_COLORS.get(t, "#888") for t in top_bowl["team"]]
    ax5.bar(players_w, means_w, color=cols_w, edgecolor="none", width=0.5)
    ax5.errorbar(players_w, means_w,
                 yerr=[[m-l for m,l in zip(means_w,lows_w)],
                       [h-m for h,m in zip(highs_w,means_w)]],
                 fmt="none", color=STYLE["text"], capsize=5, linewidth=1.5)
    ax5.set_title("🟣 Purple Cap Projections", fontsize=10,
                  fontweight="bold", pad=8, color=STYLE["text"])
    ax5.set_ylabel("Projected Wickets")
    ax5.grid(axis="y", alpha=0.3)
    ax5.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Bottom right — Current standings
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(STYLE["panel"])
    cur_pts = results.sort_values("current_pts", ascending=False)
    cols_cur = [TEAM_COLORS.get(t, "#888") for t in cur_pts["team"]]
    ax6.barh(cur_pts["team"][::-1],
             cur_pts["current_pts"][::-1],
             color=cols_cur[::-1], height=0.6, edgecolor="none")
    ax6.set_title("📋 Current 2026 Standings",
                  fontsize=10, fontweight="bold", pad=8, color=STYLE["text"])
    ax6.set_xlabel("Points")
    ax6.grid(axis="x", alpha=0.3)
    ax6.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Main title
    fig.suptitle("🏏  IPL 2026 Prediction Dashboard  |  ML + Monte Carlo Simulation",
                 fontsize=15, fontweight="bold", color=STYLE["text"], y=1.01)

    path = f"{CHART_DIR}/dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close()
    print(f"   ✅ Saved: dashboard.png  ← main portfolio image")


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n🎨 IPL 2026 — Generating Visualizations\n" + "="*45)
    set_dark_style()

    results  = pd.read_csv(f"{OUT_DIR}/championship_predictions.csv")
    top_bat  = pd.read_csv(f"{OUT_DIR}/orange_cap_predictions.csv")
    top_bowl = pd.read_csv(f"{OUT_DIR}/purple_cap_predictions.csv")
    all_pts  = pd.read_csv(f"{OUT_DIR}/simulation_distributions.csv")

    try:
        backtest = pd.read_csv(f"{OUT_DIR}/backtest_results.csv")
    except Exception:
        backtest = pd.DataFrame()

    print("\n   Generating charts...")
    chart_championship(results)
    chart_playoff(results)
    chart_points_distribution(all_pts, results)
    chart_caps(top_bat, top_bowl)
    chart_feature_importance()
    chart_dashboard(results, top_bat, top_bowl, backtest)

    print(f"\n✅ All charts saved to {CHART_DIR}/")
    print("\n   Files generated:")
    for f in os.listdir(CHART_DIR):
        size = os.path.getsize(f"{CHART_DIR}/{f}") // 1024
        print(f"   📊 {f:<40} {size} KB")


if __name__ == "__main__":
    run()
"""
app.py — IPL 2026 Prediction Dashboard (Streamlit)
----------------------------------------------------
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL 2026 Predictions",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Team colors ───────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "RCB":  "#C41E3A", "DC":   "#0047AB", "RR":   "#FF69B4",
    "PBKS": "#DC143C", "GT":   "#1B4D8E", "SRH":  "#FF8C00",
    "MI":   "#005DA0", "LSG":  "#00A86B", "KKR":  "#3A0CA3",
    "CSK":  "#FFD700",
}

TEAM_LOGOS = {
    "RCB": "🔴", "DC": "🔵", "RR": "🩷", "PBKS": "🔴",
    "GT":  "🔵", "SRH": "🟠", "MI": "🔵", "LSG": "🟢",
    "KKR": "🟣", "CSK": "🟡",
}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    results  = pd.read_csv("outputs/championship_predictions.csv")
    top_bat  = pd.read_csv("outputs/orange_cap_predictions.csv")
    top_bowl = pd.read_csv("outputs/purple_cap_predictions.csv")
    all_pts  = pd.read_csv("outputs/simulation_distributions.csv")
    backtest = pd.read_csv("outputs/backtest_results.csv")
    baseline = pd.read_csv("outputs/baseline_comparison.csv")
    return results, top_bat, top_bowl, all_pts, backtest, baseline


# ── Custom CSS ────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    .main { background-color: #0D1117; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #F0883E;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8B949E;
        margin-top: 4px;
    }
    .team-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin: 2px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #E6EDF3;
        border-bottom: 2px solid #F0883E;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <h1 style='color:#F0883E; font-size:2.5rem; margin:0;'>
            🏏 IPL 2026 Prediction Dashboard
        </h1>
        <p style='color:#8B949E; font-size:1rem; margin-top:8px;'>
            ML Ensemble (XGBoost + LR + RF) + Monte Carlo Simulation (2,000 runs)
            &nbsp;|&nbsp; Data: 18 seasons, 1,073 matches, 49 features
            &nbsp;|&nbsp; Updated: April 8, 2026 (Match 13 completed)
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


# ── KPI row ───────────────────────────────────────────────────────────────────
def render_kpis(results, backtest):
    leader     = results.iloc[0]
    top_team   = leader["team"]
    top_prob   = leader["championship_prob"] * 100
    avg_acc    = backtest["accuracy"].mean() * 100
    n_features = 49
    n_sims     = 2000

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, f"{TEAM_LOGOS.get(top_team,'')} {top_team}", "🏆 Title Favourite"),
        (c2, f"{top_prob:.1f}%",    "Win Probability"),
        (c3, f"{avg_acc:.1f}%",     "Model Accuracy"),
        (c4, f"{n_features}",       "Features Engineered"),
        (c5, f"{n_sims:,}",         "Simulations Run"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ── Championship probability chart ───────────────────────────────────────────
def chart_championship(results):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=results["championship_prob"] * 100,
        y=results["team"],
        orientation="h",
        marker_color=[TEAM_COLORS.get(t, "#888") for t in results["team"]],
        text=[f"{p:.1f}%" for p in results["championship_prob"] * 100],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Win Probability: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="🏆 Championship Win Probability", font_size=16,
                   font_color="#E6EDF3"),
        xaxis=dict(title="Probability (%)", gridcolor="#21262D",
                   color="#8B949E"),
        yaxis=dict(autorange="reversed", color="#8B949E"),
        plot_bgcolor="#161B22",
        paper_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=420,
        margin=dict(l=20, r=80, t=50, b=20),
    )
    return fig


# ── Playoff probability chart ─────────────────────────────────────────────────
def chart_playoff(results):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=results["playoff_prob"] * 100,
        y=results["team"],
        orientation="h",
        marker_color=[TEAM_COLORS.get(t, "#888") for t in results["team"]],
        opacity=0.85,
        text=[f"{p:.1f}%" for p in results["playoff_prob"] * 100],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Playoff Probability: %{x:.1f}%<extra></extra>",
    ))

    fig.add_vline(x=50, line_dash="dash", line_color="#F0883E",
                  annotation_text="50% threshold",
                  annotation_font_color="#F0883E")

    fig.update_layout(
        title=dict(text="🎯 Playoff Qualification Probability", font_size=16,
                   font_color="#E6EDF3"),
        xaxis=dict(title="Probability (%)", gridcolor="#21262D",
                   color="#8B949E", range=[0, 110]),
        yaxis=dict(autorange="reversed", color="#8B949E"),
        plot_bgcolor="#161B22",
        paper_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=420,
        margin=dict(l=20, r=80, t=50, b=20),
    )
    return fig


# ── Points distribution violin ────────────────────────────────────────────────
def chart_distribution(all_pts, results):
    top6   = results.head(6)["team"].tolist()
    fig    = go.Figure()

    for team in top6:
        fig.add_trace(go.Violin(
            y=all_pts[team],
            name=team,
            box_visible=True,
            meanline_visible=True,
            fillcolor=TEAM_COLORS.get(team, "#888"),
            opacity=0.7,
            line_color=TEAM_COLORS.get(team, "#888"),
            hovertemplate=f"<b>{team}</b><br>Points: %{{y}}<extra></extra>",
        ))

    fig.add_hline(y=16, line_dash="dot", line_color="#F0883E",
                  annotation_text="~Playoff threshold",
                  annotation_font_color="#F0883E")

    fig.update_layout(
        title=dict(text="📊 Final Points Distribution — Top 6 Teams",
                   font_size=16, font_color="#E6EDF3"),
        yaxis=dict(title="Final Points", gridcolor="#21262D", color="#8B949E"),
        xaxis=dict(color="#8B949E"),
        plot_bgcolor="#161B22",
        paper_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=420,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ── Cap predictions chart ─────────────────────────────────────────────────────
def chart_caps(top_bat, top_bowl):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["🟠 Orange Cap — Projected Runs",
                                        "🟣 Purple Cap — Projected Wickets"])

    # Orange Cap
    for i, row in top_bat.iterrows():
        color = TEAM_COLORS.get(row["team"], "#888")
        name  = row["player"].split()[-1]
        fig.add_trace(go.Bar(
            x=[name],
            y=[row["projected_runs_mean"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["projected_runs_high"] - row["projected_runs_mean"]],
                arrayminus=[row["projected_runs_mean"] - row["projected_runs_low"]],
                color="#E6EDF3",
            ),
            marker_color=color,
            name=row["player"],
            text=f"{row['projected_runs_mean']}<br>({row['projected_runs_low']}–{row['projected_runs_high']})",
            textposition="outside",
            hovertemplate=f"<b>{row['player']}</b> ({row['team']})<br>"
                          f"Current: {int(row['runs'])} runs<br>"
                          f"Projected: {int(row['projected_runs_mean'])}<br>"
                          f"Range: {int(row['projected_runs_low'])}–{int(row['projected_runs_high'])}"
                          "<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

    # Purple Cap
    for i, row in top_bowl.iterrows():
        color = TEAM_COLORS.get(row["team"], "#888")
        name  = row["player"].split()[-1]
        fig.add_trace(go.Bar(
            x=[name],
            y=[row["projected_wickets_mean"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["projected_wickets_high"] - row["projected_wickets_mean"]],
                arrayminus=[row["projected_wickets_mean"] - row["projected_wickets_low"]],
                color="#E6EDF3",
            ),
            marker_color=color,
            name=row["player"],
            text=f"{row['projected_wickets_mean']}<br>({row['projected_wickets_low']}–{row['projected_wickets_high']})",
            textposition="outside",
            hovertemplate=(
                f"<b>{row['player']}</b> ({row['team']})<br>"
                f"Current: {int(row['wickets'])} wkts<br>"
                f"Projected: {int(row['projected_wickets_mean'])}<br>"
                f"Range: {int(row['projected_wickets_low'])}–"
                f"{int(row['projected_wickets_high'])}"
                "<extra></extra>"
            ),
        ), row=1, col=2)

    fig.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(color="#8B949E")
    fig.update_yaxes(gridcolor="#21262D", color="#8B949E")
    return fig


# ── Validation chart ──────────────────────────────────────────────────────────
def chart_validation(backtest, baseline):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "📊 Walk-Forward Backtest Accuracy",
            "🏅 Our Model vs Baselines",
        ]
    )

    # Backtest bars
    seasons = [str(s) for s in backtest["season"].tolist()]
    accs    = (backtest["accuracy"] * 100).tolist()
    fig.add_trace(go.Bar(
        x=seasons, y=accs,
        marker_color=["#4D9DE0", "#F0883E"],
        text=[f"{a:.1f}%" for a in accs],
        textposition="outside",
        name="Accuracy",
        showlegend=False,
        hovertemplate="Season %{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=50, line_dash="dash", line_color="#8B949E",
                  annotation_text="Random baseline (50%)",
                  annotation_font_color="#8B949E", row=1, col=1)

    # Baseline comparison grouped bars
    seasons_b = baseline["season"].astype(str).tolist()
    models    = ["our_model", "elo_only", "home_favored", "random"]
    names     = ["Our Model", "Elo Only", "Home Favored", "Random"]
    colors    = ["#F0883E", "#4D9DE0", "#7EE787", "#8B949E"]

    for model, name, color in zip(models, names, colors):
        fig.add_trace(go.Bar(
            x=seasons_b,
            y=(baseline[model] * 100).tolist(),
            name=name,
            marker_color=color,
            hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0D1117",
        font_color="#E6EDF3",
        height=400,
        barmode="group",
        legend=dict(bgcolor="#161B22", font_color="#E6EDF3"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(color="#8B949E")
    fig.update_yaxes(gridcolor="#21262D", color="#8B949E",
                     range=[30, 70], row=1, col=1)
    fig.update_yaxes(gridcolor="#21262D", color="#8B949E",
                     range=[30, 70], row=1, col=2)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(results):
    st.sidebar.markdown("### 🏏 IPL 2026 Predictor")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📅 Data as of:** April 8, 2026")
    st.sidebar.markdown("**🎯 Matches played:** 13 of 70")
    st.sidebar.markdown("**🎲 Simulations:** 2,000")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Standings**")
    for _, row in results.sort_values("current_pts", ascending=False).iterrows():
        bar_len = int(row["current_pts"] / 6 * 10)
        bar     = "█" * bar_len + "░" * (10 - bar_len)
        st.sidebar.markdown(
            f"`{row['team']:<5}` {bar} **{int(row['current_pts'])} pts**"
        )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Tech Stack**")
    st.sidebar.markdown("XGBoost + LR + Random Forest")
    st.sidebar.markdown("Monte Carlo Simulation")
    st.sidebar.markdown("49 engineered features")
    st.sidebar.markdown("Elo ratings + Pitch data")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-View_Code-black?logo=github)]"
        "(https://github.com)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    results, top_bat, top_bowl, all_pts, backtest, baseline = load_data()

    render_header()
    render_sidebar(results)
    render_kpis(results, backtest)

    # Row 1 — Championship + Playoff
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Row 2 — Points distribution
    st.plotly_chart(chart_distribution(all_pts, results),
                    use_container_width=True, config={'displayModeBar': False})

    # Row 3 — Cap predictions
    st.markdown("<div class='section-header'>🏅 Player Projections</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_caps(top_bat, top_bowl),
                    use_container_width=True, config={'displayModeBar': False})

    # Row 4 — Validation
    st.markdown("<div class='section-header'>📊 Model Validation</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_validation(backtest, baseline),
                    use_container_width=True, config={'displayModeBar': False})

    # Row 5 — Raw data tables
    with st.expander("📋 View Raw Prediction Data"):
        tab1, tab2, tab3 = st.tabs(
            ["Championship", "Orange Cap", "Purple Cap"])
        with tab1:
            st.dataframe(results[[
                "team", "championship_prob", "playoff_prob",
                "avg_final_pts", "pts_p10", "pts_p90", "current_pts"
            ]].style.format({
                "championship_prob": "{:.1%}",
                "playoff_prob": "{:.1%}",
                "avg_final_pts": "{:.1f}",
                "pts_p10": "{:.0f}",
                "pts_p90": "{:.0f}",
            }), use_container_width=True, config={'displayModeBar': False})
        with tab2:
            st.dataframe(top_bat[[
                "player", "team", "runs",
                "projected_runs_mean",
                "projected_runs_low", "projected_runs_high"
            ]], use_container_width=True, config={'displayModeBar': False})
        with tab3:
            st.dataframe(top_bowl[[
                "player", "team", "wickets",
                "projected_wickets_mean",
                "projected_wickets_low", "projected_wickets_high"
            ]], use_container_width=True, config={'displayModeBar': False})

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#8B949E; font-size:0.85rem;'>
        Built with Python • XGBoost • Scikit-learn • Monte Carlo Simulation
        &nbsp;|&nbsp; Data: Cricsheet.org + Kaggle
        &nbsp;|&nbsp; 1,073 matches • 279K deliveries • 49 features
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
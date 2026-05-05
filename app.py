"""
NEET 2025 Seat Predictor — Streamlit App
=========================================
Run: streamlit run app.py
"""

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEET 2025 Seat Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load predictor (cached so it only initialises once) ───────────────────────
@st.cache_resource(show_spinner="Loading predictor…")
def load_predictor():
    from model.predict import Predictor
    return Predictor()


ROUND_ORDER_MAP = {"Round 1": 1, "Round 2": 2, "Mop-up Round": 3, "Stray Round": 4}
ROUND_COLORS    = {"R1": "#2196F3", "R2": "#FF9800", "Mop-up": "#9C27B0", "Stray": "#607D8B"}

STATES = ["All States", "All India", "Karnataka", "Gujarat", "Rajasthan", "Tamil Nadu"]
CATEGORIES = ["General", "OBC", "EWS", "SC", "ST", "NRI"]
QUOTAS = ["All", "State Quota", "AIQ", "Management Quota", "NRI"]

DATA_PATH  = Path(__file__).parent / "data" / "processed" / "unified.csv"
MODEL_PATH = Path(__file__).parent / "model" / "checkpoints" / "seat_dnn.pt"


def chance_color(pct: float) -> str:
    if pct >= 70:
        return "background-color: #C8E6C9; color: #1B5E20"   # green
    if pct >= 40:
        return "background-color: #FFF9C4; color: #F57F17"   # yellow
    return "background-color: #FFCDD2; color: #B71C1C"       # red


def style_chance(val):
    try:
        return chance_color(float(val))
    except Exception:
        return ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/"
             "Medical_Caduceus.svg/240px-Medical_Caduceus.svg.png", width=60)
    st.title("NEET 2025\nSeat Predictor")
    st.caption("Based on 2025 counselling closing ranks")

    st.divider()

    rank = st.number_input(
        "Your NEET Rank",
        min_value=1, max_value=800_000,
        value=15_000, step=500,
        help="Enter your All India Rank from the NEET 2025 result",
    )

    category = st.selectbox("Category", CATEGORIES, index=0)

    state = st.selectbox(
        "State / Quota Type",
        STATES,
        index=0,
        help="Select the state counselling or 'All India' for AIQ seats",
    )

    quota = st.selectbox(
        "Quota",
        QUOTAS,
        index=0,
        help="Filter by quota type. Choose 'All' to see everything.",
    )

    gender = st.selectbox(
        "Gender",
        ["All", "Female", "Male"],
        index=0,
        help=(
            "Gender-specific reservations: Rajasthan has 30–33% horizontal "
            "female reservation. Girls-only colleges (Lady Hardinge, BPS GMC, "
            "SVIMS-SPMCW) are included. 'All' shows every seat."
        ),
    )

    st.divider()
    st.subheader("Sort Results By")
    sort_by = st.radio(
        "Sort",
        options=["Chance (High → Low)", "Round (R1 First)"],
        index=0,
        label_visibility="collapsed",
        horizontal=False,
    )

    st.divider()
    predict_btn = st.button("🔍 Find Available Seats", use_container_width=True, type="primary")

    st.divider()
    st.caption("**Data source:** neetugguidance.in, mcc.nic.in")
    if MODEL_PATH.exists():
        st.success("DNN model loaded ✓", icon="🧠")
    elif DATA_PATH.exists():
        st.info("Using lookup mode (run training for DNN)", icon="📋")
    else:
        st.warning("No data yet. Run the scraper first.", icon="⚠️")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🏥 NEET 2025 Seat Predictor")
st.markdown(
    "Find which MBBS seats are likely available for you based on your rank, "
    "category, and state. Predictions are based on 2025 counselling closing ranks."
)

if not DATA_PATH.exists():
    st.error(
        "### No data found\n\n"
        "Run the scraper first to fetch closing rank data:\n\n"
        "```bash\npython -m scraper.scrape\n```",
        icon="⚠️",
    )
    st.stop()

predictor = load_predictor()

# ── Main prediction flow ───────────────────────────────────────────────────────
if predict_btn or "results" in st.session_state:

    if predict_btn:
        with st.spinner("Computing seat availability…"):
            results = predictor.predict(
                student_rank=rank,
                category=category,
                state=state,
                quota=quota if quota != "All" else None,
                gender=gender,
            )
        st.session_state["results"]  = results
        st.session_state["rank"]     = rank
        st.session_state["category"] = category
        st.session_state["state"]    = state
        st.session_state["quota"]    = quota
        st.session_state["gender"]   = gender
    else:
        results  = st.session_state["results"]
        rank     = st.session_state["rank"]
        category = st.session_state["category"]
        state    = st.session_state["state"]
        quota    = st.session_state["quota"]
        gender   = st.session_state.get("gender", "All")

    if results.empty:
        st.warning(
            "No matching colleges found. Try broadening your filters "
            "(e.g. choose 'All' quota or 'All States')."
        )
        st.stop()

    # ── Apply sort ────────────────────────────────────────────────────────
    if sort_by == "Chance (High → Low)":
        results = results.sort_values("chance_pct", ascending=False).reset_index(drop=True)
    else:  # Round (R1 First)
        results = results.sort_values(
            ["round_num", "chance_pct"], ascending=[True, False]
        ).reset_index(drop=True)

    # ── Summary KPIs ──────────────────────────────────────────────────────
    likely   = (results["chance_pct"] >= 70).sum()
    possible = ((results["chance_pct"] >= 40) & (results["chance_pct"] < 70)).sum()
    r1_seats = (results["best_round"] == "R1").sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Colleges Found", len(results["college"].unique()))
    k2.metric("Likely Seats (≥70%)",   likely)
    k3.metric("Possible Seats (≥40%)", possible)
    k4.metric("Available in Round 1",  r1_seats)

    st.divider()

    # ── Results table ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        sort_label = "sorted by chance" if "Chance" in sort_by else "sorted by round"
        st.subheader(f"Results — {sort_label}")

        display = results[[
            "college", "state", "quota", "gender", "round",
            "closing_rank", "chance_pct", "best_round",
        ]].copy()
        display.columns = [
            "College", "State", "Quota", "Gender", "Round",
            "Closing Rank 2025", "Chance %", "Best Round",
        ]
        display.index = range(1, len(display) + 1)

        styled = display.style.applymap(
            style_chance, subset=["Chance %"]
        ).format({"Chance %": "{:.1f}", "Closing Rank 2025": "{:,}"})
        # Gender column styling
        styled = styled.applymap(
            lambda v: "color: #E91E63; font-weight:600" if v == "Female"
            else ("color: #1565C0; font-weight:600" if v == "Male" else ""),
            subset=["Gender"],
        )

        st.dataframe(styled, use_container_width=True, height=520)

        st.caption(
            "🟢 ≥ 70% likely  |  🟡 40–70% possible  |  🔴 < 40% unlikely  |  "
            "Based on 2025 closing ranks with ±12% rank buffer"
        )

    with col_right:
        st.subheader("Survival Curve")
        st.caption("How seat availability changes as rank worsens")

        college_list = results["college"].unique().tolist()
        sel_college  = st.selectbox("Select college", college_list, key="curve_college")

        sel_row = results[results["college"] == sel_college].iloc[0]
        curve_quota = sel_row["quota"]

        curve_data = predictor.survival_curve(
            college  = sel_college,
            category = category,
            quota    = curve_quota,
            state    = sel_row["state"],
        )

        if not curve_data.empty:
            fig = go.Figure()

            for round_name in curve_data["round"].unique():
                rd = curve_data[curve_data["round"] == round_name]
                fig.add_trace(go.Scatter(
                    x=rd["rank"],
                    y=rd["chance_pct"],
                    mode="lines",
                    name=round_name,
                    line=dict(width=2.5),
                ))

            # Student rank vertical line
            fig.add_vline(
                x=rank, line_dash="dash", line_color="#E53935", line_width=2,
                annotation_text=f"Your rank: {rank:,}",
                annotation_position="top right",
                annotation_font_color="#E53935",
            )

            # 50% horizontal guide
            fig.add_hline(
                y=50, line_dash="dot", line_color="grey", line_width=1,
                annotation_text="50%", annotation_position="right",
            )

            fig.update_layout(
                xaxis_title="NEET Rank",
                yaxis_title="Chance of Seat (%)",
                yaxis=dict(range=[0, 105]),
                xaxis=dict(type="log"),
                legend=dict(
                    orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1
                ),
                margin=dict(t=20, b=40, l=40, r=20),
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(245,245,245,0.6)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No curve data available for this selection.")

    # ── Round-wise breakdown ───────────────────────────────────────────────
    st.divider()
    st.subheader("Round-wise Seat Distribution")
    tab1, tab2, tab3, tab4 = st.tabs(["Round 1", "Round 2", "Mop-up Round", "Stray Round"])

    for tab, round_name in zip(
        [tab1, tab2, tab3, tab4],
        ["Round 1", "Round 2", "Mop-up Round", "Stray Round"],
    ):
        with tab:
            r_df = results[results["round"] == round_name][
                ["college", "state", "quota", "gender", "closing_rank", "chance_pct"]
            ].copy()
            r_df.columns = ["College", "State", "Quota", "Gender", "Closing Rank", "Chance %"]
            r_df = r_df.sort_values("Chance %", ascending=False).reset_index(drop=True)
            r_df.index = range(1, len(r_df) + 1)

            if r_df.empty:
                st.info(f"No data for {round_name} with selected filters.")
            else:
                styled_r = (
                    r_df.style
                    .applymap(style_chance, subset=["Chance %"])
                    .applymap(
                        lambda v: "color: #E91E63; font-weight:600" if v == "Female"
                        else ("color: #1565C0; font-weight:600" if v == "Male" else ""),
                        subset=["Gender"],
                    )
                    .format({"Chance %": "{:.1f}", "Closing Rank": "{:,}"})
                )
                st.dataframe(styled_r, use_container_width=True, height=350)

else:
    # ── Welcome screen ────────────────────────────────────────────────────
    st.info(
        "👈 **Enter your rank, category and state in the sidebar, "
        "then click Find Available Seats.**",
        icon="ℹ️",
    )

    with st.expander("How does this work?"):
        st.markdown("""
**Survival Model Logic**

As your rank increases (gets worse), seats at each college progressively
become unavailable — like a survival function where *time* is rank.

For each college × quota × round combination, we know the 2025 closing rank.
We compute:

```
P(seat available) = sigmoid((closing_rank − your_rank) / spread)
```

where `spread ≈ 12%` of the closing rank — this gives a smooth probability
curve rather than a hard cutoff.

Once the DNN model is trained on multi-year data (2020–2025), it learns
*shared patterns* across similar colleges, improving estimates for colleges
with fewer historical data points.

**Rounds explained**
| Round | Seats available |
|-------|----------------|
| Round 1 | Best colleges, competitive |
| Round 2 | Some seats remain after R1 |
| Mop-up | Leftover seats from R1+R2 |
| Stray | Final stray vacancies |
        """)

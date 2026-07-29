"""Streamlit dashboard for optpricing.

Deliberately kept outside the optpricing/ package: this file only ever
imports from the public `optpricing` API (as any other consumer of the
published library would), never from internal modules. That's what keeps it
safe to publish the library on its own later without dragging the dashboard
along, and it doubles as a smoke test that the public API is actually usable
end-to-end.
"""

import plotly.graph_objects as go
import streamlit as st

from optpricing.engines import (
    BlackScholesEngine,
    CrankNicolsonEngine,
    MonteCarloEngine,
    UnsupportedCombination,
)
from optpricing.instruments import American, European, Option
from optpricing.market import MarketData
from optpricing.payoffs import CashOrNothingCall, CashOrNothingPut, Call, Put, UpAndOutCall
from optpricing.processes import GBM

st.set_page_config(page_title="PDE Solver", page_icon=":triangular_ruler:", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.08);
        border: 1px solid rgba(127, 127, 127, 0.2);
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; opacity: 0.7; letter-spacing: 0.04em; }
    .cond-badge {
        display: inline-block; padding: 1px 8px; border-radius: 5px;
        font-size: 0.72rem; font-weight: 600; color: white; margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FRAMEWORKS = ["Vanilla", "Barrier", "Digital", "American"]

OPTION_TYPES = {
    "Vanilla": ["Call", "Put"],
    "American": ["Call", "Put"],
    "Barrier": ["Up-and-Out Call"],
    "Digital": ["Cash-or-Nothing Call", "Cash-or-Nothing Put"],
}


def badge(text: str, color: str) -> str:
    return f'<span class="cond-badge" style="background:{color}">{text}</span>'


def build_option(framework: str, option_type: str, strike: float, maturity: float, barrier: float, cash: float) -> Option:
    exercise = American() if framework == "American" else European()

    if framework in ("Vanilla", "American"):
        payoff = Call(strike=strike) if option_type == "Call" else Put(strike=strike)
    elif framework == "Barrier":
        payoff = UpAndOutCall(strike=strike, barrier=barrier)
    else:  # Digital
        payoff = (
            CashOrNothingCall(strike=strike, cash=cash)
            if option_type == "Cash-or-Nothing Call"
            else CashOrNothingPut(strike=strike, cash=cash)
        )
    return Option(payoff=payoff, expiry=maturity, exercise=exercise)


def conditions_tex(framework: str, option_type: str) -> tuple[str, str]:
    """LaTeX (no $ delimiters) for the terminal and boundary conditions,
    dispatched on the same framework/option_type the sidebar already tracks —
    this only decides what to *display*; CrankNicolsonEngine derives these
    same conditions itself when it actually builds the grid.
    """
    if framework == "Barrier":
        return (
            r"V(S,T)=\max(S-K,0)\ \text{for } S<B",
            r"V(0,t)=0,\quad V(B,t)=0\ \text{(knock-out)}",
        )
    if framework == "Digital":
        if option_type == "Cash-or-Nothing Call":
            return (
                r"V(S,T)=Q\cdot\mathbb{1}\{S>K\}",
                r"V(0,t)=0,\quad V(S_{\max},t)=Qe^{-r(T-t)}",
            )
        return (
            r"V(S,T)=Q\cdot\mathbb{1}\{S<K\}",
            r"V(0,t)=Qe^{-r(T-t)},\quad V(S_{\max},t)=0",
        )
    # Vanilla / American share the same conditions — American differs in the
    # governing equation (a variational inequality), not the boundary data.
    if option_type == "Call":
        return (
            r"V(S,T)=\max(S-K,0)",
            r"V(0,t)=0,\quad V(S_{\max},t)=S_{\max}e^{-q(T-t)}-Ke^{-r(T-t)}",
        )
    return (
        r"V(S,T)=\max(K-S,0)",
        r"V(0,t)=Ke^{-r(T-t)},\quad V(S_{\max},t)=0",
    )


with st.sidebar:
    st.markdown("### :triangular_ruler: PDE Solver")
    st.caption("FINITE DIFFERENCES × MONTE CARLO")

    framework = st.segmented_control("Framework", FRAMEWORKS, default="Vanilla")
    framework = framework or "Vanilla"

    exercise_label = "American" if framework == "American" else "European"
    st.caption(f"Black-Scholes PDE — {framework} {exercise_label}")

    option_type = st.selectbox("Option type", OPTION_TYPES[framework], key=f"option_type_{framework}")

    st.markdown("**Market & Contract**")
    col_a, col_b = st.columns(2)
    spot = col_a.number_input("Spot S₀", value=100.0, min_value=0.01, step=1.0)
    strike = col_b.number_input("Strike K", value=100.0, min_value=0.01, step=1.0)
    rate = col_a.number_input("Rate r", value=0.05, step=0.005, format="%.4f")
    div_yield = col_b.number_input("Div yield q", value=0.0, step=0.005, format="%.4f")
    vol = col_a.number_input("Vol σ", value=0.20, min_value=0.001, step=0.01, format="%.4f")
    maturity = col_b.number_input("Maturity T (yr)", value=1.0, min_value=0.01, step=0.25)

    barrier = None
    if framework == "Barrier":
        barrier = st.number_input("Barrier B", value=max(130.0, spot * 1.1), min_value=spot + 0.01, step=1.0)

    cash = None
    if framework == "Digital":
        cash = st.number_input("Cash payout Q", value=1.0, min_value=0.0, step=0.5)

    st.markdown("**Finite-difference grid**")
    st.selectbox("Scheme", ["Crank-Nicolson (2nd order)"])
    col_c, col_d = st.columns(2)
    s_steps = col_c.number_input("S-steps (M)", value=200, min_value=10, max_value=400, step=10)
    t_steps = col_d.number_input("t-steps (N)", value=200, min_value=10, max_value=400, step=10)

    st.markdown("**Monte-Carlo**")
    col_e, col_f = st.columns(2)
    mc_paths = col_e.number_input("Paths", value=100_000, min_value=1_000, max_value=1_000_000, step=1_000)
    mc_steps = col_f.number_input("Steps", value=252, min_value=1, max_value=1000, step=1)

option = build_option(framework, option_type, strike, maturity, barrier, cash)
market = MarketData(spot=spot, rate=rate, dividend_yield=div_yield)
process = GBM(vol=vol)

fd_result = CrankNicolsonEngine(n_space_steps=int(s_steps), n_time_steps=int(t_steps)).price(option, process, market)

try:
    bs_price = BlackScholesEngine().price(option, process, market).price
except UnsupportedCombination:
    bs_price = None

try:
    mc_result = MonteCarloEngine(n_paths=int(mc_paths), n_steps=int(mc_steps), seed=7).price(option, process, market)
except UnsupportedCombination:
    mc_result = None

st.title(f"Black-Scholes PDE — {framework} {exercise_label}")
st.caption(
    f"{option_type} · FD (crank-nicolson, {s_steps}×{t_steps}) vs "
    f"MC ({mc_paths:,} paths{', not applicable' if mc_result is None else ''})"
)

with st.container(border=True):
    st.markdown("● **Governing equation & conditions**")
    if framework == "American":
        st.latex(
            r"\min\!\left(-\frac{\partial V}{\partial t} - \mathcal{L}V,\; V - \text{payoff}\right) = 0,"
            r"\qquad \mathcal{L}V = \tfrac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}"
            r" + (r-q)S\frac{\partial V}{\partial S} - rV"
        )
    else:
        st.latex(
            r"\frac{\partial V}{\partial t} + \tfrac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}"
            r" + (r-q)S\frac{\partial V}{\partial S} - rV = 0"
        )
    terminal_tex, boundary_tex = conditions_tex(framework, option_type)
    st.markdown(f"{badge('TERMINAL', '#2563eb')} ${terminal_tex}$", unsafe_allow_html=True)
    st.markdown(f"{badge('BOUNDARY', '#7c3aed')} ${boundary_tex}$", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("FINITE DIFFERENCE", f"{fd_result.price:.4f}")
    st.caption(f"crank-nicolson ({s_steps}×{t_steps})")

with c2:
    if mc_result is not None:
        st.metric("MONTE-CARLO", f"{mc_result.price:.4f}")
        lo = mc_result.price - 1.96 * mc_result.std_error
        hi = mc_result.price + 1.96 * mc_result.std_error
        st.caption(f"95% CI [{lo:.4f}, {hi:.4f}]")
    else:
        st.metric("MONTE-CARLO", "N/A")
        st.caption("needs Longstaff-Schwartz — not implemented for American exercise")

with c3:
    if bs_price is not None:
        st.metric("CLOSED FORM", f"{bs_price:.4f}")
        st.caption("analytic")
    else:
        st.metric("CLOSED FORM", "N/A")
        st.caption("no closed form implemented for this combination")

with c4:
    if mc_result is not None:
        diff = fd_result.price - mc_result.price
        rel = diff / mc_result.price * 100 if mc_result.price else float("nan")
        outside_ci = abs(diff) > 1.96 * mc_result.std_error
        st.metric("FD − MC", f"{diff:+.4f}")
        flag = " · **outside CI**" if outside_ci else ""
        st.caption(f"rel {rel:+.3f}%{flag}")
    else:
        st.metric("FD − MC", "—")
        st.caption("MC unavailable for this framework")

st.markdown("● **Price surface V(S, t)**")
view = st.radio("View", ["3D", "Heatmap"], horizontal=True, label_visibility="collapsed")

S_grid = fd_result.diagnostics["S_grid"]
t_grid = fd_result.diagnostics["t_grid"]
V_surface = fd_result.diagnostics["V_surface"]

if view == "3D":
    fig = go.Figure(data=[go.Surface(x=S_grid, y=t_grid, z=V_surface, colorscale="Viridis")])
    fig.update_layout(
        scene=dict(xaxis_title="Spot S", yaxis_title="Time t", zaxis_title="Value V"),
        height=600,
        margin=dict(l=0, r=0, t=10, b=0),
    )
else:
    fig = go.Figure(data=go.Heatmap(x=S_grid, y=t_grid, z=V_surface, colorscale="Viridis"))
    fig.update_layout(xaxis_title="Spot S", yaxis_title="Time t", height=600, margin=dict(l=0, r=0, t=10, b=0))

st.plotly_chart(fig, width="stretch")

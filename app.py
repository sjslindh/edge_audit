import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from edge_audit import (
    expectancy_R,
    kelly_fraction_R,
    profit_factor_estimate_R,
    monte_carlo_equity_R,
)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Edge Audit", layout="wide")

# Dark-mode friendly CSS (Streamlit already supports dark theme, but this tightens it up a bit)
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .stDownloadButton button { width: 100%; }
      .stNumberInput, .stSlider { margin-bottom: 0.5rem; }
      .small-note { color: #A0A0A0; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Edge Audit — Strategy Reality Check")
st.caption("Not financial advice. Educational/statistical analysis based on the inputs you provide.")

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    win_rate_pct = st.number_input("Win rate (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.5)

    st.markdown("**Outcome magnitudes (R-multiples)**")
    avg_win_R = st.number_input("Average win (R)", min_value=0.0, value=1.5, step=0.1)
    avg_loss_R = st.number_input("Average loss (R)", min_value=0.1, value=1.0, step=0.1)

    st.markdown("**Risk sizing**")
    risk_per_trade_pct = st.number_input(
        "Risk per trade (% of account)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    trades_year = st.number_input("Trades per year", min_value=1, value=120, step=10)
    start_capital = st.number_input("Account size ($)", min_value=100.0, value=5000.0, step=100.0)

    st.divider()
    st.markdown("**Simulation**")
    n_sims = st.slider("Monte Carlo simulations", min_value=200, max_value=5000, value=1000, step=100)
    seed = st.number_input("Random seed", min_value=1, value=42, step=1)

# -----------------------------
# Convert inputs
# -----------------------------
win_rate = win_rate_pct / 100.0
risk_per_trade = risk_per_trade_pct / 100.0
n_trades = int(trades_year)

# -----------------------------
# Core metrics
# -----------------------------
e = expectancy_R(win_rate, float(avg_win_R), float(avg_loss_R), float(risk_per_trade))
pf = profit_factor_estimate_R(win_rate, float(avg_win_R), float(avg_loss_R))
k_raw = kelly_fraction_R(win_rate, float(avg_win_R), float(avg_loss_R))

# Recommendations: 1/4 Kelly, capped (professional)
k_quarter = max(0.0, 0.25 * k_raw) if not np.isnan(k_raw) else np.nan
k_reco = min(k_quarter, 0.05) if not np.isnan(k_quarter) else np.nan  # cap at 5% risk/trade

# -----------------------------
# Sanity rails (warnings)
# -----------------------------
if risk_per_trade > 0.03:
    st.warning("Risk per trade above 3% is aggressive and can produce very large drawdowns.")
if avg_win_R > 3.0:
    st.warning("Average win above 3R is uncommon for many liquid strategies. Double-check inputs.")
if n_trades > 300:
    st.warning("Trades/year is very high. Ensure this matches your strategy frequency and holding time.")

# -----------------------------
# Monte Carlo
# -----------------------------
mc = monte_carlo_equity_R(
    win_rate=win_rate,
    avg_win_R=float(avg_win_R),
    avg_loss_R=float(avg_loss_R),
    risk_per_trade=float(risk_per_trade),
    n_trades=n_trades,
    start_capital=float(start_capital),
    n_sims=int(n_sims),
    seed=int(seed),
)

stats = mc["stats"]
equity = mc["equity"]
end_cap = mc["end_cap"]

# -----------------------------
# Layout: metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expectancy / trade", f"{e*100:.2f}%")
col2.metric("Profit Factor (est.)", f"{pf:.2f}" if not np.isnan(pf) else "—")
col3.metric("Kelly (raw)", f"{k_raw*100:.1f}%" if not np.isnan(k_raw) else "—")
col4.metric("Kelly (¼, capped)", f"{k_reco*100:.1f}%" if not np.isnan(k_reco) else "—")

st.markdown('<div class="small-note">Model assumption: each trade risks a fixed % of equity; wins/losses are expressed in R-multiples.</div>', unsafe_allow_html=True)

# -----------------------------
# Results: Monte Carlo summary
# -----------------------------
st.subheader("Monte Carlo results (1 year)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Median ending capital", f"${stats['median_end']:,.0f}")
c2.metric("5th percentile ending", f"${stats['p05_end']:,.0f}")
c3.metric("Prob. -20% year", f"{stats['prob_down_20']*100:.1f}%")
c4.metric("Prob. double", f"{stats['prob_double']*100:.1f}%")

st.subheader("Drawdown risk")
d1, d2, d3 = st.columns(3)
d1.metric("Median max drawdown", f"{stats['median_max_dd']*100:.1f}%")
d2.metric("Prob. DD ≥ 30%", f"{stats['prob_dd_30']*100:.1f}%")
d3.metric("Prob. DD ≥ 50%", f"{stats['prob_dd_50']*100:.1f}%")

# -----------------------------
# Verdict logic (simple but better)
# -----------------------------
st.subheader("Verdict (simple)")

verdict = "Negative Edge"
if e > 0 and (pf < 1.2):
    verdict = "Weak / Fragile Edge"
elif e > 0 and (pf >= 1.2) and (pf < 1.5):
    verdict = "Moderate Edge"
elif e > 0 and (pf >= 1.5):
    verdict = "Strong Edge"

if e > 0 and stats["prob_dd_30"] > 0.50:
    verdict += " (Sizing / Drawdown Risk)"

st.write(f"**{verdict}**")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Charts")

# Equity paths (first 50)
fig1 = plt.figure()
plt.plot(equity[:50].T)
plt.xlabel("Trade #")
plt.ylabel("Equity ($)")
st.pyplot(fig1)

# Ending capital distribution
fig2 = plt.figure()
plt.hist(end_cap, bins=40)
plt.xlabel("Ending capital ($)")
plt.ylabel("Frequency")
st.pyplot(fig2)

# -----------------------------
# Download report (HTML) - fast and reliable
# -----------------------------
report_html = f"""
<h1>Edge Audit Report</h1>

<p><b>Inputs</b></p>
<ul>
  <li>Win rate: {win_rate_pct:.2f}%</li>
  <li>Average win: {avg_win_R:.2f}R</li>
  <li>Average loss: {avg_loss_R:.2f}R</li>
  <li>Risk per trade: {risk_per_trade_pct:.2f}% of account</li>
  <li>Trades/year: {n_trades}</li>
  <li>Account size: ${start_capital:,.0f}</li>
</ul>

<p><b>Core Metrics</b></p>
<ul>
  <li>Expectancy per trade: {e*100:.2f}%</li>
  <li>Profit factor (est): {pf:.2f}</li>
  <li>Kelly (raw): {k_raw*100:.1f}%</li>
  <li>Kelly (¼ Kelly, capped at 5%): {k_reco*100:.1f}%</li>
</ul>

<p><b>Monte Carlo (n={int(n_sims)})</b></p>
<ul>
  <li>Median ending capital: ${stats['median_end']:,.0f}</li>
  <li>5th percentile ending: ${stats['p05_end']:,.0f}</li>
  <li>95th percentile ending: ${stats['p95_end']:,.0f}</li>
  <li>Probability of -20% year: {stats['prob_down_20']*100:.1f}%</li>
  <li>Probability of doubling: {stats['prob_double']*100:.1f}%</li>
</ul>

<p><b>Drawdowns</b></p>
<ul>
  <li>Median max drawdown: {stats['median_max_dd']*100:.1f}%</li>
  <li>Prob DD ≥ 30%: {stats['prob_dd_30']*100:.1f}%</li>
  <li>Prob DD ≥ 50%: {stats['prob_dd_50']*100:.1f}%</li>
</ul>

<h2>Verdict</h2>
<p><b>{verdict}</b></p>

<p style="margin-top:12px;color:#666;font-size:12px;">
Model assumption: each trade risks a fixed % of equity; wins/losses are expressed in R-multiples.
</p>

<p style="margin-top:24px;color:#666;font-size:12px;">
Not financial advice. For educational/statistical purposes only.
</p>
"""

st.download_button(
    label="Download report (HTML)",
    data=report_html,
    file_name="edge_audit_report.html",
    mime="text/html",
)
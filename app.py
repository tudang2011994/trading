"""Streamlit dashboard for bot.py's CSV-formatted trades.log."""

from pathlib import Path

import pandas as pd
import streamlit  as st


LOG_PATH = Path("trades.log")
INITIAL_VALUE = 100000.0

st.set_page_config(page_title="Trading Bot Dashboard", page_icon="📈", layout="wide")
st.title("📈 Automated Trading Bot")
st.caption("Portfolio and daily execution history")


@st.cache_data(ttl=30)
def load_trades() -> pd.DataFrame:
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return pd.DataFrame()
    data = pd.read_csv(LOG_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for column in ("price", "quantity", "cash", "position_shares", "portfolio_value", "prediction"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["timestamp"]).sort_values("timestamp")


trades = load_trades()
if trades.empty:
    st.info("No trades logged yet. Run `python bot.py` once to populate this dashboard.")
    st.stop()

current_value = float(trades.iloc[-1]["portfolio_value"])
closed_sells = trades[trades["action"] == "SELL"].copy()
# Match each long-only exit to the most recent entry to calculate realized wins/losses.
entry_price = None
wins = losses = 0
for trade in trades.itertuples():
    if trade.action == "BUY":
        entry_price = float(trade.price)
    elif trade.action == "SELL" and entry_price is not None:
        if float(trade.price) > entry_price:
            wins += 1
        else:
            losses += 1
        entry_price = None
ratio = f"{wins}/{losses}" if wins + losses else "N/A"

first, second, third = st.columns(3)
first.metric("Current Portfolio Value", f"${current_value:,.2f}", f"${current_value - INITIAL_VALUE:,.2f}")
second.metric("Win/Loss Ratio", ratio)
third.metric("Latest Signal", str(trades.iloc[-1]["action"]))

st.subheader("Portfolio Performance")
performance = trades.set_index("timestamp")[["portfolio_value"]].rename(columns={"portfolio_value": "Portfolio Value"})
st.line_chart(performance)

st.subheader("Daily Historical Trades")
display = trades.copy()
display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
st.dataframe(display, use_container_width=True, hide_index=True)

"""
Streamlit dashboard for the swing scanner.

Tab 1: today's breakout-with-volume signals (from results/signals_today.csv,
       written daily by the GitHub Actions workflow)
Tab 2: backtest results (from results/trades.csv)
Tab 3: run the backtest live with your own settings
"""

import os

import pandas as pd
import streamlit as st

import swing_backtest as sb

st.set_page_config(page_title="Swing Scanner", layout="wide")
st.title("Swing scanner: downtrend structure break + volume")
st.caption("Long only, daily timeframe, time-limited hold. For research, not advice.")

RES = "results"


def read_csv(name):
    path = os.path.join(RES, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def metrics(tr):
    wins, losses = tr[tr.R > 0], tr[tr.R <= 0]
    gl = -losses.R.sum()
    cum = tr.sort_values("exit_date").R.cumsum()
    return {
        "Trades": len(tr),
        "Win rate %": round(len(wins) / len(tr) * 100, 1),
        "Expectancy (R)": round(tr.R.mean(), 3),
        "Profit factor": round(wins.R.sum() / gl, 2) if gl > 0 else float("inf"),
        "Avg return %": round(tr.ret_pct.mean(), 2),
        "Max DD (R)": round((cum - cum.cummax()).min(), 1),
    }


def show_results(tr):
    tr = tr.copy()
    tr["entry_date"] = pd.to_datetime(tr["entry_date"])
    tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    tr = tr.sort_values("exit_date").reset_index(drop=True)

    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics(tr).items()):
        col.metric(k, v)

    st.subheader("Cumulative R")
    st.line_chart(tr.R.cumsum())

    left, right = st.columns(2)
    with left:
        st.subheader("By year")
        by_year = tr.groupby(tr.entry_date.dt.year).R.agg(["count", "mean"]).round(3)
        by_year.columns = ["trades", "expectancy_R"]
        st.dataframe(by_year)
    with right:
        st.subheader("Exit reasons")
        st.dataframe(tr.reason.value_counts().rename("trades"))

    st.subheader("All trades")
    st.dataframe(tr, width="stretch")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_data(ticker, start):
    sb.START = start
    return sb.load_data(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def live_run(start, pivot_n, target_r, max_hold, vol_mult, use_vol):
    sb.PIVOT_N = int(pivot_n)
    sb.TARGET_R = float(target_r)
    sb.MAX_HOLD_DAYS = int(max_hold)
    sb.VOL_MULT = float(vol_mult)
    all_trades, today = [], []
    for tk in sb.TICKERS:
        try:
            df = get_data(tk, start)
            if df is None:
                continue
            tr, td = sb.backtest_stock(tk, df, use_vol)
            all_trades += tr
            today += td
        except Exception:
            continue
    return pd.DataFrame(all_trades), pd.DataFrame(today)


tab1, tab2, tab3 = st.tabs(["Today's signals", "Backtest results", "Run live"])

with tab1:
    sig = read_csv("signals_today.csv")
    if sig is None:
        st.info("No results yet. Run the GitHub Action once so it creates results/.")
    elif sig.empty:
        st.success("No stock broke its lower high on volume in the latest scan.")
    else:
        st.write(f"Breakouts on bar date **{sig['date'].iloc[0]}** "
                 "(entry would be next day's open):")
        st.dataframe(sig.sort_values("vol_ratio", ascending=False),
                     width="stretch")
    log = os.path.join(RES, "run_log.txt")
    if os.path.exists(log):
        with st.expander("Last run log"):
            st.text(open(log).read())

with tab2:
    trades = read_csv("trades.csv")
    if trades is None or trades.empty:
        st.info("No backtest results yet. Run the GitHub Action once.")
    else:
        show_results(trades)

with tab3:
    st.write("Run the backtest here with your own settings. "
             "It downloads data for the whole stock list, so it takes a minute or two.")
    c1, c2, c3, c4, c5 = st.columns(5)
    start = c1.text_input("Start date", sb.START)
    pivot_n = c2.number_input("Pivot bars", 2, 8, int(sb.PIVOT_N))
    target_r = c3.number_input("Target (R)", 1.0, 5.0, float(sb.TARGET_R), 0.5)
    max_hold = c4.number_input("Max hold days", 2, 20, int(sb.MAX_HOLD_DAYS))
    vol_mult = c5.number_input("Volume x average", 1.0, 5.0, float(sb.VOL_MULT), 0.1)
    use_vol = st.checkbox("Require volume on breakout", True)

    if st.button("Run backtest now"):
        with st.spinner("Downloading data and testing..."):
            tr, today = live_run(start, pivot_n, target_r, max_hold, vol_mult, use_vol)
        if tr.empty:
            st.warning("No trades with these settings.")
        else:
            show_results(tr)
        st.subheader("Breakouts on the latest bar")
        if today.empty:
            st.write("None.")
        else:
            st.dataframe(today.sort_values("vol_ratio", ascending=False))

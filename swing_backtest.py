"""
Swing trade backtest: downtrend -> uptrend STRUCTURE BREAK (long only, daily)

Rules
-----
Pivots  : swing high/low = extreme of PIVOT_N bars on each side.
          A pivot is only "known" PIVOT_N bars after it forms (no look-ahead).
Trend   : downtrend = last 2 swing highs are lower AND last 2 swing lows are lower.
Signal  : first daily CLOSE above the most recent (lower) swing high while in downtrend,
          WITH VOLUME: breakout-day volume >= VOL_MULT x average of previous VOL_AVG_DAYS days.
Scanner : the latest bar is also checked, so you get a list of stocks that just
          broke out on volume (candidates for next day's entry).
Entry   : next day's OPEN.
Stop    : most recent swing low (gap below stop at open -> exit at open).
Target  : TARGET_R x risk.
Time    : exit at CLOSE of day MAX_HOLD_DAYS (weekly swing = 5 trading days).
Costs   : COST_PER_SIDE applied on entry and exit (brokerage + slippage).
If stop and target are both touched on the same day, stop is assumed first.

Install : pip install yfinance pandas numpy matplotlib
Run     : python swing_backtest.py
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------- SETTINGS ------------------------------------
START = "2015-01-01"
END = None                    # None = today
PIVOT_N = 3                   # bars each side to confirm a swing point
TARGET_R = 2.0                # profit target in multiples of risk
MAX_HOLD_DAYS = 5             # trading days (5 = about one week)
MAX_RISK_PCT = 0.08           # skip trades where stop is >8% below entry
COST_PER_SIDE = 0.0010        # 0.10% per side
USE_VOLUME_FILTER = True      # breakout day volume must be strong
VOL_AVG_DAYS = 20             # average volume lookback (excludes breakout day)
VOL_MULT = 1.5                # breakout volume >= 1.5 x average volume
SAVE_SIGNALS_CSV = "signals_today.csv"
SAVE_TRADES_CSV = "trades.csv"
SAVE_EQUITY_PNG = "equity_R.png"

TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC",
    "SBIN", "LT", "HINDUNILVR", "KOTAKBANK", "AXISBANK", "BAJFINANCE",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "ASIANPAINT", "NTPC",
    "POWERGRID", "ONGC", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "M&M",
    "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "HCLTECH", "WIPRO", "TECHM",
    "NESTLEIND", "GRASIM", "CIPLA", "DRREDDY", "EICHERMOT", "HEROMOTOCO",
    "BPCL", "HINDALCO", "INDUSINDBK", "APOLLOHOSP", "DIVISLAB", "BRITANNIA",
    "TATACONSUM", "SBILIFE", "HDFCLIFE", "BAJAJ-AUTO",
]
# ---------------------------------------------------------------------------


def load_data(ticker):
    df = yf.download(f"{ticker}.NS", start=START, end=END,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df if len(df) > 100 else None


def find_pivots(high, low, n):
    """Boolean arrays: pivot at bar i (needs n bars on both sides)."""
    L = len(high)
    is_sh = np.zeros(L, dtype=bool)
    is_sl = np.zeros(L, dtype=bool)
    for i in range(n, L - n):
        if high[i] == high[i - n:i + n + 1].max():
            is_sh[i] = True
        if low[i] == low[i - n:i + n + 1].min():
            is_sl[i] = True
    return is_sh, is_sl


def backtest_stock(ticker, df, use_vol=True):
    """Returns (trades, signals_on_latest_bar)."""
    o, h, l, c, v = (df[k].to_numpy(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    avgvol = (pd.Series(v).rolling(VOL_AVG_DAYS).mean().shift(1)
              .fillna(0).to_numpy())          # previous N days only, no look-ahead
    dates = df.index
    L = len(df)
    is_sh, is_sl = find_pivots(h, l, PIVOT_N)

    highs, lows = [], []          # confirmed swing prices, oldest first
    trades, today = [], []
    t = 0
    while t < L:
        # pivot at bar t-N becomes known at the close of bar t
        p = t - PIVOT_N
        if p >= PIVOT_N:
            if is_sh[p]:
                highs.append(h[p])
            if is_sl[p]:
                lows.append(l[p])

        if len(highs) >= 2 and len(lows) >= 2 and t > 0:
            downtrend = highs[-1] < highs[-2] and lows[-1] < lows[-2]
            level = highs[-1]
            first_break = c[t] > level and c[t - 1] <= level
            vol_ratio = v[t] / avgvol[t] if avgvol[t] > 0 else 0.0
            vol_ok = vol_ratio >= VOL_MULT
            if downtrend and first_break and (vol_ok or not use_vol):
                if t == L - 1:          # breakout on the latest bar -> scanner hit
                    stop_now = lows[-1]
                    today.append({
                        "ticker": ticker,
                        "date": dates[t].date(),
                        "close": round(c[t], 2),
                        "broke_level": round(level, 2),
                        "stop": round(stop_now, 2),
                        "risk_pct": round((c[t] - stop_now) / c[t] * 100, 2),
                        "vol_ratio": round(vol_ratio, 2),
                    })
                    t += 1
                    continue
                e = t + 1
                entry = o[e]
                stop = lows[-1]
                risk = entry - stop
                if risk > 0 and risk / entry <= MAX_RISK_PCT:
                    target = entry + TARGET_R * risk
                    last_day = min(e + MAX_HOLD_DAYS - 1, L - 1)
                    exit_px, exit_i, reason = None, None, None
                    for j in range(e, last_day + 1):
                        if j > e and o[j] <= stop:
                            exit_px, exit_i, reason = o[j], j, "gap_stop"
                        elif l[j] <= stop:
                            exit_px, exit_i, reason = stop, j, "stop"
                        elif h[j] >= target:
                            exit_px, exit_i, reason = target, j, "target"
                        elif j == last_day:
                            exit_px, exit_i, reason = c[j], j, "time"
                        if exit_px is not None:
                            break
                    if exit_px is not None and not (e == L - 1 and reason == "time"):
                        net_pnl = exit_px * (1 - COST_PER_SIDE) - entry * (1 + COST_PER_SIDE)
                        trades.append({
                            "ticker": ticker,
                            "signal_date": dates[t],
                            "entry_date": dates[e],
                            "exit_date": dates[exit_i],
                            "entry": round(entry, 2),
                            "stop": round(stop, 2),
                            "target": round(target, 2),
                            "exit": round(exit_px, 2),
                            "reason": reason,
                            "days_held": exit_i - e + 1,
                            "ret_pct": round(net_pnl / entry * 100, 2),
                            "R": round(net_pnl / risk, 2),
                            "vol_ratio": round(vol_ratio, 2),
                        })
                        t = exit_i      # no overlapping trades in the same stock
        t += 1
    return trades, today


def print_scanner(today):
    print("\n===== SCANNER: breakout on latest bar =====")
    if not today:
        print("No stock broke its lower high on the latest bar"
              + (" with volume." if USE_VOLUME_FILTER else "."))
        # overwrite with an empty file so an old day's signals never look current
        pd.DataFrame(columns=["ticker", "date", "close", "broke_level",
                              "stop", "risk_pct", "vol_ratio"]).to_csv(
            SAVE_SIGNALS_CSV, index=False)
        return
    sig = pd.DataFrame(today).sort_values("vol_ratio", ascending=False)
    print(sig.to_string(index=False))
    sig.to_csv(SAVE_SIGNALS_CSV, index=False)
    print(f"\nSaved {SAVE_SIGNALS_CSV}. Entry would be next day's open; "
          f"exit by stop / {TARGET_R}R target / day {MAX_HOLD_DAYS}.")


def summarize(tr, label="ALL"):
    if tr.empty:
        print(f"[{label}] no trades")
        return
    wins, losses = tr[tr.R > 0], tr[tr.R <= 0]
    cum = tr.sort_values("exit_date").R.cumsum()
    max_dd = (cum - cum.cummax()).min()
    gp, gl = wins.R.sum(), -losses.R.sum()
    print(f"\n===== {label} =====")
    print(f"Trades            : {len(tr)}")
    print(f"Win rate          : {len(wins) / len(tr) * 100:.1f}%")
    print(f"Avg win / loss (R): {wins.R.mean() if len(wins) else 0:.2f} / "
          f"{losses.R.mean() if len(losses) else 0:.2f}")
    print(f"Expectancy (R)    : {tr.R.mean():.3f}")
    print(f"Profit factor     : {gp / gl if gl > 0 else float('inf'):.2f}")
    print(f"Avg return/trade  : {tr.ret_pct.mean():.2f}%")
    print(f"Avg days held     : {tr.days_held.mean():.1f}")
    print(f"Max drawdown (R)  : {max_dd:.1f}")
    print("Exit reasons      :", tr.reason.value_counts().to_dict())


def main():
    data = {}
    for tk in TICKERS:
        try:
            df = load_data(tk)
            if df is None:
                print(f"skip {tk}: no data")
                continue
            data[tk] = df
        except Exception as ex:
            print(f"skip {tk}: {ex}")

    # Does the volume filter actually help? Compare against no filter.
    nofilter = pd.DataFrame([x for tk, df in data.items()
                             for x in backtest_stock(tk, df, False)[0]])
    if not nofilter.empty:
        summarize(nofilter.sort_values("exit_date"), "NO VOLUME FILTER (comparison)")

    all_trades, today = [], []
    for tk, df in data.items():
        tr, td = backtest_stock(tk, df, USE_VOLUME_FILTER)
        all_trades += tr
        today += td
        print(f"{tk:12s} trades: {len(tr)}")

    trades = pd.DataFrame(all_trades)
    if trades.empty:
        print("No trades generated.")
        return
    trades = trades.sort_values("exit_date").reset_index(drop=True)
    trades.to_csv(SAVE_TRADES_CSV, index=False)

    summarize(trades, "ALL TRADES")

    # in-sample vs out-of-sample check (guards against curve fitting)
    mid = trades.entry_date.sort_values().iloc[len(trades) // 2]
    summarize(trades[trades.entry_date < mid], f"FIRST HALF (before {mid.date()})")
    summarize(trades[trades.entry_date >= mid], f"SECOND HALF (from {mid.date()})")

    print("\nBy year (expectancy in R):")
    print(trades.groupby(trades.entry_date.dt.year).R.agg(["count", "mean"]).round(3))

    print_scanner(today)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        trades.R.cumsum().plot(figsize=(10, 4), title="Cumulative R (all trades, by exit date)")
        plt.xlabel("Trade #")
        plt.ylabel("R")
        plt.tight_layout()
        plt.savefig(SAVE_EQUITY_PNG, dpi=120)
        print(f"\nSaved {SAVE_TRADES_CSV} and {SAVE_EQUITY_PNG}")
    except ImportError:
        print(f"\nSaved {SAVE_TRADES_CSV} (install matplotlib for the equity chart)")


if __name__ == "__main__":
    main()

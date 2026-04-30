"""
Multi-Signal Smart Broker

Дополнительные сигналы (всё бесплатно через yfinance):
- S&P 500 (SPY) — общий рынок
- VIX — индекс страха
- Золото (GLD)
- Нефть (USO)
- Bitcoin (BTC-USD)
- Доллар (DX-Y.NYB)
- Облигации 10Y (TLT)
- Объём торгов

Логика: модель прогнозирует цену, а мульти-сигналы подтверждают или отменяют сделку.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(
    max_context=1024, max_horizon=256,
    normalize_inputs=True, use_continuous_quantile_head=True,
    force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
))

# --- Загрузка всех данных ---
print("Загрузка данных...")

tickers_trade = ["AAPL", "TSLA", "GOOG"]
tickers_signals = ["SPY", "GLD", "USO", "BTC-USD", "^VIX", "TLT"]
all_tickers = tickers_trade + tickers_signals

data = {}
for t in all_tickers:
    df = yf.download(t, start="2024-01-01", end="2026-02-01", progress=False)
    data[t] = df
    print(f"  {t}: {len(df)} дней")

# Выравниваем по общим датам (торговые дни акций)
common_dates = data["AAPL"].index
for t in ["SPY", "GLD", "USO", "TLT"]:
    common_dates = common_dates.intersection(data[t].index)

print(f"Общих торговых дней: {len(common_dates)}")

# Приводим к общим датам
aligned = {}
for t in all_tickers:
    df = data[t]
    # BTC и VIX торгуются не все дни — forward fill
    df = df.reindex(common_dates, method="ffill")
    aligned[t] = {
        "close": df["Close"].values.flatten(),
        "volume": df["Volume"].values.flatten() if "Volume" in df.columns else np.ones(len(df)),
    }

NUM_DAYS = 60
CONTEXT = min(512, len(common_dates) - NUM_DAYS - 5)
START = 10000
COMMISSION = 0.001

# --- Индикаторы ---
def rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(period)/period, mode="valid")
    avg_loss = np.convolve(losses, np.ones(period)/period, mode="valid")
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)

def momentum(prices, period=5):
    return (prices[-1] - prices[-period]) / prices[-period] * 100


# --- Прогнозы TimesFM для торгуемых акций ---
print("\nГенерация прогнозов...")
predictions = {}
for ticker in tickers_trade:
    series = aligned[ticker]["close"]
    total = CONTEXT + NUM_DAYS + 1
    chunk = series[-total:]
    batch = [chunk[CONTEXT + d - CONTEXT : CONTEXT + d] for d in range(NUM_DAYS)]
    point, _ = model.forecast(horizon=3, inputs=batch)
    predictions[ticker] = point
    print(f"  {ticker}: {NUM_DAYS} прогнозов")


def compute_market_score(day_idx):
    """
    Считает рыночный скор от -3 до +3:
    +1 за каждый бычий сигнал, -1 за медвежий.

    Сигналы:
    1. SPY momentum (5 дней) > 0 -> бычий рынок
    2. VIX < 20 -> низкий страх -> бычий
    3. Золото падает -> деньги идут в акции -> бычий
    4. BTC растёт -> risk-on -> бычий
    5. Нефть стабильна (не крашится) -> бычий
    6. Облигации падают (TLT) -> деньги идут в акции -> бычий
    """
    total = CONTEXT + NUM_DAYS + 1
    score = 0

    # SPY momentum
    spy = aligned["SPY"]["close"][-total:]
    idx = CONTEXT + day_idx
    if idx >= 5:
        spy_mom = momentum(spy[idx-5:idx+1])
        score += 1 if spy_mom > 0.5 else (-1 if spy_mom < -0.5 else 0)

    # VIX level
    vix = aligned["^VIX"]["close"][-total:]
    vix_now = vix[idx]
    score += 1 if vix_now < 18 else (-1 if vix_now > 25 else 0)

    # Gold momentum (обратный — золото растёт = fear)
    gld = aligned["GLD"]["close"][-total:]
    if idx >= 5:
        gld_mom = momentum(gld[idx-5:idx+1])
        score += 1 if gld_mom < -0.3 else (-1 if gld_mom > 1.0 else 0)

    # BTC momentum
    btc = aligned["BTC-USD"]["close"][-total:]
    if idx >= 5:
        btc_mom = momentum(btc[idx-5:idx+1])
        score += 1 if btc_mom > 1 else (-1 if btc_mom < -2 else 0)

    # Oil stability
    uso = aligned["USO"]["close"][-total:]
    if idx >= 5:
        uso_mom = momentum(uso[idx-5:idx+1])
        score += 1 if -2 < uso_mom < 2 else -1

    # Bonds (inverse — bonds down = stocks up)
    tlt = aligned["TLT"]["close"][-total:]
    if idx >= 5:
        tlt_mom = momentum(tlt[idx-5:idx+1])
        score += 1 if tlt_mom < -0.3 else (-1 if tlt_mom > 0.5 else 0)

    return score  # -6 to +6


def run_strategy(ticker, use_signals=True):
    """Запускает стратегию."""
    series = aligned[ticker]["close"]
    volumes = aligned[ticker]["volume"]
    total = CONTEXT + NUM_DAYS + 1
    chunk = series[-total:]
    chunk_vol = volumes[-total:]
    pred = predictions[ticker]

    capital = START
    in_market = False
    shares = 0
    entry_price = 0
    equity = [capital]
    trades = 0
    reasons = []

    for d in range(NUM_DAYS):
        idx = CONTEXT + d
        today = chunk[idx]
        tomorrow = chunk[idx + 1]
        pred_tomorrow = pred[d, 0]
        pred_change = (pred_tomorrow - today) / today

        # RSI
        lookback = chunk[max(0, idx-30):idx+1]
        current_rsi = rsi(lookback)[-1] if len(lookback) > 15 else 50

        # Volume vs average
        vol_lookback = chunk_vol[max(0, idx-20):idx+1]
        avg_vol = np.mean(vol_lookback[:-1]) if len(vol_lookback) > 1 else vol_lookback[-1]
        vol_ratio = vol_lookback[-1] / (avg_vol + 1e-10)

        # Market score
        mkt_score = compute_market_score(d) if use_signals else 0

        # Stop-loss
        if in_market and (today - entry_price) / entry_price <= -0.02:
            capital = shares * today * (1 - COMMISSION)
            shares = 0
            in_market = False
            trades += 1
            reasons.append("stop")
            equity.append(capital)
            continue

        # --- РЕШЕНИЕ О ПОКУПКЕ ---
        if not in_market:
            buy = False

            if use_signals:
                # Нужно: прогноз > 0.5% И рынок >= 0 И RSI < 70
                if pred_change > 0.005 and mkt_score >= 0 and current_rsi < 70:
                    # Бонус: если рынок сильно бычий (score >= 3), покупаем даже при слабом прогнозе
                    buy = True
                elif pred_change > 0.003 and mkt_score >= 3:
                    buy = True
            else:
                if pred_change > 0:
                    buy = True

            if buy:
                shares = capital * (1 - COMMISSION) / today
                capital = 0
                entry_price = today
                in_market = True
                trades += 1
                reasons.append(f"buy(m={mkt_score})")

        # --- РЕШЕНИЕ О ПРОДАЖЕ ---
        elif in_market:
            sell = False

            if use_signals:
                if pred_change < -0.003 or mkt_score <= -2 or current_rsi > 75:
                    sell = True
            else:
                if pred_change < 0:
                    sell = True

            if sell:
                capital = shares * today * (1 - COMMISSION)
                shares = 0
                in_market = False
                trades += 1
                reasons.append(f"sell(m={mkt_score})")

        equity.append(shares * tomorrow if in_market else capital)

    final = shares * chunk[CONTEXT + NUM_DAYS] if in_market else capital
    return final, equity, trades


# --- Запуск ---
print("\n" + "=" * 70)
print(f"{'Ticker':<7} {'Strategy':<20} {'Result':>10} {'Return':>8} {'Trades':>7}")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="white",
                         gridspec_kw={"height_ratios": [2, 1]})
colors = {"AAPL": "#E91E63", "TSLA": "#FF9800", "GOOG": "#9C27B0"}
days = np.arange(NUM_DAYS + 1)
summary = {}

for ticker in tickers_trade:
    series = aligned[ticker]["close"]
    total = CONTEXT + NUM_DAYS + 1
    chunk = series[-total:]
    bh_start = chunk[CONTEXT]
    bh_end = chunk[CONTEXT + NUM_DAYS]
    bh_return = (bh_end / bh_start - 1) * 100
    bh_equity = [START * chunk[CONTEXT + d] / bh_start for d in range(NUM_DAYS + 1)]

    dumb_final, dumb_eq, dumb_trades = run_strategy(ticker, use_signals=False)
    dumb_return = (dumb_final / START - 1) * 100

    smart_final, smart_eq, smart_trades = run_strategy(ticker, use_signals=True)
    smart_return = (smart_final / START - 1) * 100

    summary[ticker] = {
        "bh": bh_return, "dumb": dumb_return, "smart": smart_return,
        "bh_f": START * (1 + bh_return/100), "dumb_f": dumb_final, "smart_f": smart_final,
    }

    print(f"{ticker:<7} {'Buy & Hold':<20} ${START*(1+bh_return/100):>9,.0f} {bh_return:>+7.1f}%")
    print(f"{'':7} {'Model only':<20} ${dumb_final:>9,.0f} {dumb_return:>+7.1f}% {dumb_trades:>5}")
    print(f"{'':7} {'Model + Signals':<20} ${smart_final:>9,.0f} {smart_return:>+7.1f}% {smart_trades:>5}")
    print()

    ax = axes[0]
    ax.plot(days, smart_eq, color=colors[ticker], lw=2.5,
            label=f"{ticker} +signals ({smart_return:+.1f}%)")
    ax.plot(days, dumb_eq, color=colors[ticker], lw=1.2, ls=":",
            label=f"{ticker} model only ({dumb_return:+.1f}%)")
    ax.plot(days, bh_equity, color=colors[ticker], lw=1, ls="--", alpha=0.4,
            label=f"{ticker} hold ({bh_return:+.1f}%)")

ax = axes[0]
ax.axhline(y=START, color="#999", ls=":", lw=1)
ax.set_title("Model + Market Signals vs Model Only vs Buy & Hold", fontsize=14, fontweight="bold")
ax.legend(fontsize=7, loc="upper left", ncol=3)
ax.grid(True, alpha=0.12)
ax.set_ylabel("Portfolio, $", fontsize=11)
ax.set_xlabel("Trading day", fontsize=10)

# Бар-чарт
ax2 = axes[1]
x = np.arange(len(tickers_trade))
w = 0.25
bars1 = ax2.bar(x - w, [summary[t]["bh"] for t in tickers_trade], w, color="#90CAF9", label="Buy & Hold")
bars2 = ax2.bar(x, [summary[t]["dumb"] for t in tickers_trade], w, color="#FFCC80", label="Model only")
bars3 = ax2.bar(x + w, [summary[t]["smart"] for t in tickers_trade], w, color="#CE93D8", label="Model + Signals")

ax2.axhline(y=0, color="#333", lw=1)
ax2.set_xticks(x)
ax2.set_xticklabels(tickers_trade, fontsize=12, fontweight="bold")
ax2.set_ylabel("Return, %", fontsize=11)
ax2.set_title("Return comparison", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.12)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:+.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                     ha="center", va="bottom" if h >= 0 else "top", fontsize=9)

plt.tight_layout()
plt.savefig("multi_signal_backtest.png", dpi=150, bbox_inches="tight")

# Итог
print("=" * 70)
total_s = sum(summary[t]["smart_f"] for t in tickers_trade)
total_d = sum(summary[t]["dumb_f"] for t in tickers_trade)
total_b = sum(summary[t]["bh_f"] for t in tickers_trade)
print(f"ИТОГО ($30k):")
print(f"  Buy & Hold:      ${total_b:,.0f}")
print(f"  Model only:      ${total_d:,.0f}")
print(f"  Model + Signals: ${total_s:,.0f}")
print(f"\n  Signals vs Hold: ${total_s - total_b:+,.0f}")
print(f"  Signals vs Dumb: ${total_s - total_d:+,.0f}")

# Показать рыночные сигналы за последние дни
print(f"\n--- Текущие рыночные сигналы ---")
score = compute_market_score(NUM_DAYS - 1)
print(f"Market score: {score}/6 ({'BULLISH' if score >= 2 else 'BEARISH' if score <= -2 else 'NEUTRAL'})")

print("\nSaved: multi_signal_backtest.png")

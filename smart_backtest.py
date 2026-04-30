"""
Smart Broker Strategy vs Dumb Strategy vs Buy & Hold

Улучшения:
1. Фильтр уверенности — торгуем только когда прогноз > порога
2. Объём как подтверждение — покупаем только если объём выше среднего
3. RSI фильтр — не покупаем перекупленное (RSI > 70), не продаём перепроданное (RSI < 30)
4. MA тренд — торгуем только по тренду (цена > MA20)
5. Stop-loss — выходим при -2% от входа
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

tickers = ["AAPL", "TSLA", "GOOG"]
CONTEXT = 1024
NUM_DAYS = 60
COMMISSION = 0.001
START = 10000

# --- Индикаторы ---
def rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(period)/period, mode="valid")
    avg_loss = np.convolve(losses, np.ones(period)/period, mode="valid")
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)

def ma(prices, period=20):
    return np.convolve(prices, np.ones(period)/period, mode="valid")

def avg_volume(volumes, period=20):
    return np.convolve(volumes, np.ones(period)/period, mode="valid")


# --- Загрузка данных ---
stock_data = {}
for t in tickers:
    df = yf.download(t, start="2000-01-01", end="2026-02-01", progress=False)
    stock_data[t] = {
        "close": df["Close"].values.flatten(),
        "volume": df["Volume"].values.flatten(),
    }

print(f"Smart Broker Backtest: {NUM_DAYS} дней, капитал ${START:,}\n")

# --- Прогнозы ---
all_predictions = {}
for ticker in tickers:
    series = stock_data[ticker]["close"]
    total_needed = CONTEXT + NUM_DAYS + 1
    chunk = series[-total_needed:]
    batch = [chunk[CONTEXT + d - CONTEXT : CONTEXT + d] for d in range(NUM_DAYS)]
    point, _ = model.forecast(horizon=3, inputs=batch)
    all_predictions[ticker] = point


def run_dumb(ticker):
    """Тупая стратегия: торгуем каждый день по прогнозу."""
    series = stock_data[ticker]["close"]
    chunk = series[-(CONTEXT + NUM_DAYS + 1):]
    pred = all_predictions[ticker]

    capital = START
    in_market = False
    shares = 0
    equity = [capital]
    trades = 0

    for d in range(NUM_DAYS):
        today = chunk[CONTEXT + d]
        tomorrow = chunk[CONTEXT + d + 1]
        pred_tomorrow = pred[d, 0]

        if pred_tomorrow > today and not in_market:
            shares = capital * (1 - COMMISSION) / today
            capital = 0
            in_market = True
            trades += 1
        elif pred_tomorrow <= today and in_market:
            capital = shares * today * (1 - COMMISSION)
            shares = 0
            in_market = False
            trades += 1

        equity.append(shares * tomorrow if in_market else capital)

    final = shares * chunk[CONTEXT + NUM_DAYS] if in_market else capital
    return final, equity, trades


def run_smart(ticker):
    """Умная стратегия: фильтры + stop-loss."""
    series = stock_data[ticker]["close"]
    volumes = stock_data[ticker]["volume"]
    chunk_close = series[-(CONTEXT + NUM_DAYS + 1):]
    chunk_vol = volumes[-(CONTEXT + NUM_DAYS + 1):]
    pred = all_predictions[ticker]

    CONFIDENCE_THRESHOLD = 0.005  # прогноз > 0.5% для входа
    STOP_LOSS = -0.02  # -2% стоп-лосс
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    capital = START
    in_market = False
    shares = 0
    entry_price = 0
    equity = [capital]
    trades = 0
    skipped = 0
    stopped = 0

    for d in range(NUM_DAYS):
        today = chunk_close[CONTEXT + d]
        tomorrow = chunk_close[CONTEXT + d + 1]
        pred_tomorrow = pred[d, 0]
        pred_change = (pred_tomorrow - today) / today

        # Индикаторы
        lookback = chunk_close[CONTEXT + d - 30 : CONTEXT + d + 1]
        lookback_vol = chunk_vol[CONTEXT + d - 30 : CONTEXT + d + 1]

        current_rsi = rsi(lookback)[-1] if len(lookback) > 15 else 50
        current_ma = ma(lookback, 20)[-1] if len(lookback) > 20 else today
        current_avg_vol = avg_volume(lookback_vol, 20)[-1] if len(lookback_vol) > 20 else lookback_vol[-1]
        today_vol = lookback_vol[-1]

        # Stop-loss
        if in_market and (today - entry_price) / entry_price <= STOP_LOSS:
            capital = shares * today * (1 - COMMISSION)
            shares = 0
            in_market = False
            trades += 1
            stopped += 1
            equity.append(capital)
            continue

        # ПОКУПКА: все фильтры должны пройти
        if not in_market and pred_change > CONFIDENCE_THRESHOLD:
            # Фильтр 1: RSI не перекуплен
            if current_rsi > RSI_OVERBOUGHT:
                skipped += 1
                equity.append(capital)
                continue
            # Фильтр 2: цена выше MA20 (тренд вверх)
            if today < current_ma * 0.99:
                skipped += 1
                equity.append(capital)
                continue
            # Фильтр 3: объём выше среднего (подтверждение)
            if today_vol < current_avg_vol * 0.8:
                skipped += 1
                equity.append(capital)
                continue

            # Все фильтры прошли — покупаем
            shares = capital * (1 - COMMISSION) / today
            capital = 0
            entry_price = today
            in_market = True
            trades += 1

        # ПРОДАЖА: прогноз вниз или RSI перекуплен
        elif in_market and (pred_change < -CONFIDENCE_THRESHOLD or current_rsi > RSI_OVERBOUGHT):
            capital = shares * today * (1 - COMMISSION)
            shares = 0
            in_market = False
            trades += 1

        equity.append(shares * tomorrow if in_market else capital)

    final = shares * chunk_close[CONTEXT + NUM_DAYS] if in_market else capital
    return final, equity, trades, skipped, stopped


# --- Запуск ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="white",
                         gridspec_kw={"height_ratios": [2, 1]})

colors = {"AAPL": "#E91E63", "TSLA": "#FF9800", "GOOG": "#9C27B0"}
days = np.arange(NUM_DAYS + 1)

print(f"{'Ticker':<7} {'Strategy':<12} {'Result':>10} {'Return':>8} {'Trades':>7} {'Notes'}")
print("=" * 65)

summary = {}
for ticker in tickers:
    series = stock_data[ticker]["close"]
    chunk = series[-(CONTEXT + NUM_DAYS + 1):]
    bh_start = chunk[CONTEXT]
    bh_end = chunk[CONTEXT + NUM_DAYS]
    bh_return = (bh_end / bh_start - 1) * 100
    bh_equity = [START * chunk[CONTEXT + d] / bh_start for d in range(NUM_DAYS + 1)]

    dumb_final, dumb_eq, dumb_trades = run_dumb(ticker)
    dumb_return = (dumb_final / START - 1) * 100

    smart_final, smart_eq, smart_trades, skipped, stopped = run_smart(ticker)
    smart_return = (smart_final / START - 1) * 100

    summary[ticker] = {
        "bh": bh_return, "dumb": dumb_return, "smart": smart_return,
        "smart_final": smart_final, "dumb_final": dumb_final,
        "bh_final": START * bh_end / bh_start,
    }

    print(f"{ticker:<7} {'Buy&Hold':<12} ${START*(1+bh_return/100):>9,.0f} {bh_return:>+7.1f}%")
    print(f"{'':7} {'Dumb':<12} ${dumb_final:>9,.0f} {dumb_return:>+7.1f}% {dumb_trades:>5}")
    print(f"{'':7} {'Smart':<12} ${smart_final:>9,.0f} {smart_return:>+7.1f}% {smart_trades:>5}   skip:{skipped} stop:{stopped}")
    print()

    # Графики
    ax = axes[0]
    ax.plot(days, smart_eq, color=colors[ticker], lw=2.5,
            label=f"{ticker} smart ({smart_return:+.1f}%)")
    ax.plot(days, dumb_eq, color=colors[ticker], lw=1.2, ls=":",
            label=f"{ticker} dumb ({dumb_return:+.1f}%)")
    ax.plot(days, bh_equity, color=colors[ticker], lw=1, ls="--", alpha=0.4,
            label=f"{ticker} hold ({bh_return:+.1f}%)")

ax = axes[0]
ax.axhline(y=START, color="#999", ls=":", lw=1)
ax.set_title("Smart Broker vs Dumb Model vs Buy & Hold", fontsize=14, fontweight="bold")
ax.legend(fontsize=7, loc="upper left", ncol=3)
ax.grid(True, alpha=0.12)
ax.set_ylabel("Portfolio, $", fontsize=11)

# Нижний график — сравнение стратегий (бар)
ax2 = axes[1]
x = np.arange(len(tickers))
w = 0.25
bh_vals = [summary[t]["bh"] for t in tickers]
dumb_vals = [summary[t]["dumb"] for t in tickers]
smart_vals = [summary[t]["smart"] for t in tickers]

bars1 = ax2.bar(x - w, bh_vals, w, color="#90CAF9", label="Buy & Hold")
bars2 = ax2.bar(x, dumb_vals, w, color="#FFCC80", label="Dumb Model")
bars3 = ax2.bar(x + w, smart_vals, w, color="#CE93D8", label="Smart Broker")

ax2.axhline(y=0, color="#333", lw=1)
ax2.set_xticks(x)
ax2.set_xticklabels(tickers, fontsize=12, fontweight="bold")
ax2.set_ylabel("Return, %", fontsize=11)
ax2.set_title("Return comparison", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.12)

# Значения на столбиках
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:+.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                     ha="center", va="bottom" if h >= 0 else "top", fontsize=9)

plt.tight_layout()
plt.savefig("smart_backtest.png", dpi=150, bbox_inches="tight")

# Итог
print("=" * 65)
total_smart = sum(summary[t]["smart_final"] for t in tickers)
total_dumb = sum(summary[t]["dumb_final"] for t in tickers)
total_bh = sum(summary[t]["bh_final"] for t in tickers)
print(f"ИТОГО ($30k start):")
print(f"  Buy & Hold:  ${total_bh:,.0f}")
print(f"  Dumb Model:  ${total_dumb:,.0f}")
print(f"  Smart Broker: ${total_smart:,.0f}")
print(f"  Smart vs BH:  ${total_smart - total_bh:+,.0f}")
print(f"  Smart vs Dumb: ${total_smart - total_dumb:+,.0f}")
print("\nSaved: smart_backtest.png")

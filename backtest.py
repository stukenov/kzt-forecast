"""
Простой бэктест: если каждый день следовать прогнозу модели — заработаем?

Стратегия:
- Модель говорит "завтра вырастет" -> покупаем
- Модель говорит "завтра упадёт" -> не покупаем (или продаём)
- Комиссия 0.1% за сделку
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
NUM_DAYS = 60  # бэктест за 60 торговых дней (~3 месяца)
COMMISSION = 0.001  # 0.1%
START_CAPITAL = 10000  # $10,000

stock_data = {}
for t in tickers:
    df = yf.download(t, start="2000-01-01", end="2026-02-01", progress=False)
    stock_data[t] = df["Close"].values.flatten()

print(f"Backtest: {NUM_DAYS} дней, стартовый капитал ${START_CAPITAL:,}")
print(f"Комиссия: {COMMISSION*100}% за сделку\n")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="white",
                         gridspec_kw={"height_ratios": [2, 1]})

all_results = {}

for ticker in tickers:
    series = stock_data[ticker]
    total_needed = CONTEXT + NUM_DAYS + 1
    chunk = series[-total_needed:]

    # Собираем прогнозы батчем
    batch_inputs = []
    for d in range(NUM_DAYS):
        ctx_end = CONTEXT + d
        batch_inputs.append(chunk[ctx_end - CONTEXT : ctx_end])

    point, _ = model.forecast(horizon=1, inputs=batch_inputs)

    # --- Стратегия 1: Buy & Hold ---
    price_start = chunk[CONTEXT]
    price_end = chunk[CONTEXT + NUM_DAYS]
    bh_return = (price_end / price_start - 1) * 100
    bh_capital = START_CAPITAL * price_end / price_start

    # --- Стратегия 2: Модель (long only) ---
    # Если прогноз > текущей цены -> держим/покупаем, иначе -> в кэше
    capital = START_CAPITAL
    in_market = False
    shares = 0
    model_equity = [capital]
    bh_equity = [START_CAPITAL]
    signals = []  # 1=buy, 0=cash

    correct_direction = 0
    total_predictions = 0

    for d in range(NUM_DAYS):
        today_price = chunk[CONTEXT + d]
        tomorrow_price = chunk[CONTEXT + d + 1]
        predicted_tomorrow = point[d, 0]

        # Модель предсказывает рост?
        pred_up = predicted_tomorrow > today_price
        actual_up = tomorrow_price > today_price
        if pred_up == actual_up:
            correct_direction += 1
        total_predictions += 1

        if pred_up and not in_market:
            # Покупаем
            shares = capital * (1 - COMMISSION) / today_price
            capital = 0
            in_market = True
            signals.append(1)
        elif not pred_up and in_market:
            # Продаём
            capital = shares * today_price * (1 - COMMISSION)
            shares = 0
            in_market = False
            signals.append(0)
        else:
            signals.append(1 if in_market else 0)

        # Считаем equity
        if in_market:
            model_equity.append(shares * tomorrow_price)
        else:
            model_equity.append(capital)

        bh_equity.append(START_CAPITAL * tomorrow_price / chunk[CONTEXT])

    # Финальный капитал
    if in_market:
        final_capital = shares * chunk[CONTEXT + NUM_DAYS]
    else:
        final_capital = capital

    model_return = (final_capital / START_CAPITAL - 1) * 100
    direction_acc = correct_direction / total_predictions * 100
    num_trades = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])

    all_results[ticker] = {
        "model_return": model_return,
        "bh_return": bh_return,
        "final_capital": final_capital,
        "bh_capital": bh_capital,
        "direction_acc": direction_acc,
        "num_trades": num_trades,
        "model_equity": model_equity,
        "bh_equity": bh_equity,
        "signals": signals,
    }

    print(f"{ticker}:")
    print(f"  Direction accuracy: {direction_acc:.1f}% ({correct_direction}/{total_predictions})")
    print(f"  Trades: {num_trades}")
    print(f"  Model:     ${START_CAPITAL:,} -> ${final_capital:,.0f} ({model_return:+.1f}%)")
    print(f"  Buy&Hold:  ${START_CAPITAL:,} -> ${bh_capital:,.0f} ({bh_return:+.1f}%)")
    print(f"  Alpha:     {model_return - bh_return:+.1f}%")
    print()

# --- Графики ---
days = np.arange(NUM_DAYS + 1)

# Верхний: equity curves
ax = axes[0]
colors = {"AAPL": "#E91E63", "TSLA": "#FF9800", "GOOG": "#9C27B0"}
for ticker in tickers:
    r = all_results[ticker]
    ax.plot(days, r["model_equity"], color=colors[ticker], lw=2.5,
            label=f"{ticker} model ({r['model_return']:+.1f}%)")
    ax.plot(days, r["bh_equity"], color=colors[ticker], lw=1.5, ls="--", alpha=0.5,
            label=f"{ticker} buy&hold ({r['bh_return']:+.1f}%)")

ax.axhline(y=START_CAPITAL, color="#999", ls=":", lw=1)
ax.set_title(f"$10,000 invested — Model vs Buy & Hold ({NUM_DAYS} days)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", ncol=2)
ax.grid(True, alpha=0.12)
ax.set_ylabel("Portfolio value, $", fontsize=11)
ax.set_xlabel("Trading day", fontsize=10)

# Нижний: direction accuracy по дням (кумулятивная)
ax2 = axes[1]
for ticker in tickers:
    r = all_results[ticker]
    # Кумулятивная точность направления
    cum_acc = []
    correct = 0
    series_data = stock_data[ticker]
    total_needed = CONTEXT + NUM_DAYS + 1
    chunk = series_data[-total_needed:]
    batch_inputs = []
    for d in range(NUM_DAYS):
        ctx_end = CONTEXT + d
        batch_inputs.append(chunk[ctx_end - CONTEXT : ctx_end])
    point_t, _ = model.forecast(horizon=1, inputs=batch_inputs)
    for d in range(NUM_DAYS):
        today = chunk[CONTEXT + d]
        tomorrow = chunk[CONTEXT + d + 1]
        pred = point_t[d, 0]
        if (pred > today) == (tomorrow > today):
            correct += 1
        cum_acc.append(correct / (d + 1) * 100)

    ax2.plot(range(NUM_DAYS), cum_acc, color=colors[ticker], lw=2.5,
             label=f"{ticker} ({cum_acc[-1]:.0f}%)")

ax2.axhline(y=50, color="#F44336", ls="--", lw=1.5, alpha=0.7, label="50% (coin flip)")
ax2.set_title("Direction accuracy (cumulative) — above 50% = model is useful",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.12)
ax2.set_ylabel("Accuracy, %", fontsize=11)
ax2.set_xlabel("Trading day", fontsize=10)
ax2.set_ylim(35, 75)

plt.tight_layout()
plt.savefig("backtest.png", dpi=150, bbox_inches="tight")

# Итог
print("=" * 55)
total_model = sum(r["final_capital"] for r in all_results.values())
total_bh = sum(r["bh_capital"] for r in all_results.values())
print(f"ИТОГО (3 акции по $10k = $30k):")
print(f"  Model:    ${total_model:,.0f}")
print(f"  Buy&Hold: ${total_bh:,.0f}")
print(f"  Разница:  ${total_model - total_bh:+,.0f}")
print("=" * 55)
print("\nSaved: backtest.png")

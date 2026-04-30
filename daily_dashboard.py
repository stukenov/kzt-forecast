"""
Дневной дашборд: каждый день спрашиваем модель прогноз на 3 дня.
Показываем насколько она ошиблась.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import torch
import timesfm
from datetime import timedelta

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(
    max_context=1024, max_horizon=256,
    normalize_inputs=True, use_continuous_quantile_head=True,
    force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
))

tickers = ["AAPL", "TSLA", "GOOG"]
HORIZON = 3
CONTEXT = 1024
NUM_WINDOWS = 30

stock_data = {}
stock_dates = {}
for t in tickers:
    df = yf.download(t, start="2000-01-01", end="2026-02-01", progress=False)
    stock_data[t] = df["Close"].values.flatten()
    stock_dates[t] = df.index.values

# --- Собираем данные ---
results = {}
for ticker in tickers:
    series = stock_data[ticker]
    dates = stock_dates[ticker]
    total_needed = CONTEXT + NUM_WINDOWS + HORIZON
    chunk = series[-total_needed:]
    chunk_dates = dates[-total_needed:]

    batch_inputs = []
    for w in range(NUM_WINDOWS):
        ctx_end = CONTEXT + w
        batch_inputs.append(chunk[ctx_end - CONTEXT : ctx_end])

    point, _ = model.forecast(horizon=HORIZON, inputs=batch_inputs)

    r = {"dates": [], "actual": [], "pred_d1": [], "pred_d2": [], "pred_d3": [],
         "err_d1": [], "err_d2": [], "err_d3": []}
    for w in range(NUM_WINDOWS):
        ctx_end = CONTEXT + w
        actual_3 = chunk[ctx_end : ctx_end + HORIZON]
        r["dates"].append(chunk_dates[ctx_end])
        r["actual"].append(actual_3[0])
        r["pred_d1"].append(point[w, 0])
        r["pred_d2"].append(point[w, 1])
        r["pred_d3"].append(point[w, 2])
        r["err_d1"].append((point[w, 0] - actual_3[0]) / actual_3[0] * 100)
        r["err_d2"].append((point[w, 1] - actual_3[1]) / actual_3[1] * 100)
        r["err_d3"].append((point[w, 2] - actual_3[2]) / actual_3[2] * 100)

    for k in r:
        r[k] = np.array(r[k])
    results[ticker] = r

# --- График ---
fig, axes = plt.subplots(len(tickers), 2, figsize=(16, 4.5 * len(tickers)),
                         facecolor="white", gridspec_kw={"width_ratios": [2, 1]})

for ti, ticker in enumerate(tickers):
    r = results[ticker]
    days = np.arange(NUM_WINDOWS)

    # Левый график: цена + прогноз day+1
    ax = axes[ti, 0]
    ax.plot(days, r["actual"], color="#333333", lw=2.5, label="Actual price", zorder=3)
    ax.plot(days, r["pred_d1"], color="#E91E63", lw=2, marker="o", ms=5,
            markerfacecolor="white", markeredgewidth=1.5, label="Forecast (day+1)", zorder=4)

    # Соединяем actual и pred стрелками чтобы видеть ошибку
    for d in range(NUM_WINDOWS):
        color = "#4CAF50" if abs(r["err_d1"][d]) < 1 else "#FF9800" if abs(r["err_d1"][d]) < 2 else "#F44336"
        ax.plot([d, d], [r["actual"][d], r["pred_d1"][d]], color=color, lw=1.5, alpha=0.6)

    avg_mape = np.mean(np.abs(r["err_d1"]))
    ax.set_title(f"{ticker} — daily forecast vs actual  |  avg error: {avg_mape:.2f}%",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.12)
    ax.set_ylabel("Price, $", fontsize=11)
    ax.set_xlabel("Day", fontsize=10)

    # Правый график: ошибка в % по дням (day+1, +2, +3)
    ax2 = axes[ti, 1]
    width = 0.25
    ax2.bar(days - width, np.abs(r["err_d1"]), width, color="#E91E63", alpha=0.8, label="Day+1")
    ax2.bar(days, np.abs(r["err_d2"]), width, color="#FF9800", alpha=0.8, label="Day+2")
    ax2.bar(days + width, np.abs(r["err_d3"]), width, color="#9C27B0", alpha=0.8, label="Day+3")

    # Горизонтальные линии
    ax2.axhline(y=1, color="#4CAF50", ls="--", lw=1, alpha=0.7, label="1% error")
    ax2.axhline(y=2, color="#FF9800", ls="--", lw=1, alpha=0.7, label="2% error")

    avg1 = np.mean(np.abs(r["err_d1"]))
    avg2 = np.mean(np.abs(r["err_d2"]))
    avg3 = np.mean(np.abs(r["err_d3"]))
    ax2.set_title(f"Error %  (d1: {avg1:.1f}%  d2: {avg2:.1f}%  d3: {avg3:.1f}%)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.12)
    ax2.set_ylabel("Error, %", fontsize=10)
    ax2.set_xlabel("Day", fontsize=10)

plt.suptitle("Daily 3-day Forecast — How wrong is the model each day?",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("daily_dashboard.png", dpi=150, bbox_inches="tight")

# --- Итоговая таблица ---
print("\n" + "=" * 55)
print(f"{'Ticker':<8} {'Day+1':>8} {'Day+2':>8} {'Day+3':>8} {'Avg':>8}")
print("=" * 55)
for ticker in tickers:
    r = results[ticker]
    m1 = np.mean(np.abs(r["err_d1"]))
    m2 = np.mean(np.abs(r["err_d2"]))
    m3 = np.mean(np.abs(r["err_d3"]))
    avg = (m1 + m2 + m3) / 3
    print(f"{ticker:<8} {m1:>7.2f}% {m2:>7.2f}% {m3:>7.2f}% {avg:>7.2f}%")
print("=" * 55)

print("\nSaved: daily_dashboard.png")

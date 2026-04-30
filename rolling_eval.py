"""
Rolling 3-day forecast: для каждого из 30 дней сдвигаем окно на 1 день,
прогнозируем на 3 дня вперёд, сравниваем с фактом.
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
HORIZON = 3
CONTEXT = 1024
NUM_WINDOWS = 30

stock_data = {}
for t in tickers:
    df = yf.download(t, start="2000-01-01", end="2026-02-01", progress=False)
    stock_data[t] = df["Close"].values.flatten()

print(f"Rolling forecast: {NUM_WINDOWS} окон, горизонт {HORIZON} дня\n")

fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 4 * len(tickers)), facecolor="white")

for ti, ticker in enumerate(tickers):
    series = stock_data[ticker]
    # Берём последние CONTEXT + NUM_WINDOWS + HORIZON дней
    total_needed = CONTEXT + NUM_WINDOWS + HORIZON
    chunk = series[-total_needed:]

    all_preds_d1 = []
    all_preds_d2 = []
    all_preds_d3 = []
    all_actuals_d1 = []
    all_actuals_d2 = []
    all_actuals_d3 = []

    # Собираем все окна в один батч для скорости
    batch_inputs = []
    for w in range(NUM_WINDOWS):
        ctx_end = CONTEXT + w
        ctx = chunk[ctx_end - CONTEXT : ctx_end]
        batch_inputs.append(ctx)

    point, _ = model.forecast(horizon=HORIZON, inputs=batch_inputs)

    for w in range(NUM_WINDOWS):
        ctx_end = CONTEXT + w
        actual_3 = chunk[ctx_end : ctx_end + HORIZON]

        all_actuals_d1.append(actual_3[0])
        all_actuals_d2.append(actual_3[1])
        all_actuals_d3.append(actual_3[2])
        all_preds_d1.append(point[w, 0])
        all_preds_d2.append(point[w, 1])
        all_preds_d3.append(point[w, 2])

    all_actuals_d1 = np.array(all_actuals_d1)
    all_actuals_d2 = np.array(all_actuals_d2)
    all_actuals_d3 = np.array(all_actuals_d3)
    all_preds_d1 = np.array(all_preds_d1)
    all_preds_d2 = np.array(all_preds_d2)
    all_preds_d3 = np.array(all_preds_d3)

    # Метрики
    def mape(a, p):
        return np.mean(np.abs((a - p) / a)) * 100

    def mae(a, p):
        return np.mean(np.abs(a - p))

    mape1 = mape(all_actuals_d1, all_preds_d1)
    mape2 = mape(all_actuals_d2, all_preds_d2)
    mape3 = mape(all_actuals_d3, all_preds_d3)
    mae1 = mae(all_actuals_d1, all_preds_d1)
    mae2 = mae(all_actuals_d2, all_preds_d2)
    mae3 = mae(all_actuals_d3, all_preds_d3)

    mape_avg = (mape1 + mape2 + mape3) / 3

    print(f"{ticker}:")
    print(f"  Day+1: MAE=${mae1:.2f}, MAPE={mape1:.2f}%")
    print(f"  Day+2: MAE=${mae2:.2f}, MAPE={mape2:.2f}%")
    print(f"  Day+3: MAE=${mae3:.2f}, MAPE={mape3:.2f}%")
    print(f"  Average MAPE: {mape_avg:.2f}%")
    print()

    # График
    ax = axes[ti]
    days = np.arange(NUM_WINDOWS)

    ax.plot(days, all_actuals_d1, color="#333333", lw=2, label="Actual (day+1)")
    ax.plot(days, all_preds_d1, color="#E91E63", lw=2, marker="o", ms=4, label=f"Forecast day+1 (MAPE {mape1:.1f}%)")
    ax.plot(days, all_preds_d2, color="#FF9800", lw=1.5, marker="s", ms=3, label=f"Forecast day+2 (MAPE {mape2:.1f}%)")
    ax.plot(days, all_preds_d3, color="#9C27B0", lw=1.5, marker="^", ms=3, label=f"Forecast day+3 (MAPE {mape3:.1f}%)")

    # Ошибка по каждому дню (столбики)
    errors_pct = np.abs((all_actuals_d1 - all_preds_d1) / all_actuals_d1) * 100
    ax2 = ax.twinx()
    ax2.bar(days, errors_pct, alpha=0.15, color="#E91E63", label="Error %")
    ax2.set_ylabel("Error %", fontsize=10, color="#E91E63")
    ax2.set_ylim(0, max(errors_pct) * 3)
    ax2.tick_params(axis="y", labelcolor="#E91E63")

    ax.set_title(f"{ticker} — avg MAPE: {mape_avg:.2f}%", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("Window #")
    ax.set_ylabel("Price, $")

plt.suptitle(f"Rolling {HORIZON}-day Forecast ({NUM_WINDOWS} windows)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("rolling_eval.png", dpi=150, bbox_inches="tight")
print("Graph: rolling_eval.png")

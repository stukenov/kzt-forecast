"""
Прогноз курса USD/KZT (тенге) на 7 дней + rolling evaluation.
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

# Загрузка
df = yf.download("USDKZT=X", start="2020-01-01", end="2026-02-01", progress=False)
close = df["Close"].values.flatten()
dates = df.index

print(f"USD/KZT: {len(close)} дней")
print(f"Диапазон: {dates[0].date()} — {dates[-1].date()}")
print(f"Текущий курс: {close[-1]:.2f} тенге")
print(f"Мин: {close.min():.2f}, Макс: {close.max():.2f}\n")

# ============================================================
# 1. Прогноз на 7 дней вперёд (от последней даты)
# ============================================================
print("=" * 50)
print("1. ПРОГНОЗ НА 7 ДНЕЙ")
print("=" * 50)

context = close[-1024:]
point, quantile = model.forecast(horizon=7, inputs=[context])

print(f"\nТекущий курс: {close[-1]:.2f}")
print(f"\nПрогноз:")
for d in range(7):
    p = point[0, d]
    q10 = quantile[0, d, 1]
    q90 = quantile[0, d, 9]
    change = (p - close[-1]) / close[-1] * 100
    print(f"  День +{d+1}: {p:.2f} тг ({change:+.2f}%)  [диапазон: {q10:.2f} — {q90:.2f}]")

# ============================================================
# 2. Rolling evaluation: за последние 30 дней, прогноз на 1-3 дня
# ============================================================
print(f"\n{'=' * 50}")
print("2. ROLLING EVALUATION (30 дней)")
print("=" * 50)

NUM_WINDOWS = 30
CONTEXT = 1024

batch = []
for w in range(NUM_WINDOWS):
    end = len(close) - NUM_WINDOWS - 3 + w
    ctx = close[max(0, end - CONTEXT) : end]
    batch.append(ctx)

roll_point, _ = model.forecast(horizon=3, inputs=batch)

errors = {1: [], 2: [], 3: []}
for w in range(NUM_WINDOWS):
    end = len(close) - NUM_WINDOWS - 3 + w
    for h in [1, 2, 3]:
        actual = close[end + h - 1]
        pred = roll_point[w, h - 1]
        err = abs(actual - pred) / actual * 100
        errors[h].append(err)

for h in [1, 2, 3]:
    avg = np.mean(errors[h])
    mx = np.max(errors[h])
    print(f"  День +{h}: avg MAPE = {avg:.3f}%, max = {mx:.3f}%")

# ============================================================
# 3. Графики
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 14), facecolor="white")

# --- График 1: история + прогноз ---
ax = axes[0]
show_days = 90
hist_dates = np.arange(-show_days, 0)
pred_dates = np.arange(0, 7)

ax.plot(hist_dates, close[-show_days:], color="#333", lw=2, label="History")
ax.plot(pred_dates, point[0], color="#E91E63", lw=2.5, marker="o", ms=6,
        markerfacecolor="white", markeredgewidth=2, label="Forecast")
ax.fill_between(pred_dates, quantile[0, :, 1], quantile[0, :, 9],
                alpha=0.15, color="#E91E63", label="10-90% range")
ax.axvline(x=0, color="#BBB", ls=":", lw=1)
ax.set_title("USD/KZT — 7-day forecast", fontsize=14, fontweight="bold")
ax.set_ylabel("Tenge per $1", fontsize=11)
ax.set_xlabel("Days (0 = today)", fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.12)

# --- График 2: вся история ---
ax2 = axes[1]
ax2.plot(dates, close, color="#333", lw=1.5)
ax2.set_title("USD/KZT — full history (2020-2026)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Tenge per $1", fontsize=11)
ax2.grid(True, alpha=0.12)

# Аннотации ключевых событий
max_idx = np.argmax(close)
min_idx = np.argmin(close)
ax2.annotate(f"Max: {close[max_idx]:.0f}", xy=(dates[max_idx], close[max_idx]),
             fontsize=9, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", alpha=0.9))
ax2.annotate(f"Min: {close[min_idx]:.0f}", xy=(dates[min_idx], close[min_idx]),
             fontsize=9, ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#E8F5E9", alpha=0.9))

# --- График 3: ошибка по дням (rolling) ---
ax3 = axes[2]
days_r = np.arange(NUM_WINDOWS)
w = 0.25
ax3.bar(days_r - w, errors[1], w, color="#E91E63", alpha=0.8, label=f"Day+1 (avg {np.mean(errors[1]):.3f}%)")
ax3.bar(days_r, errors[2], w, color="#FF9800", alpha=0.8, label=f"Day+2 (avg {np.mean(errors[2]):.3f}%)")
ax3.bar(days_r + w, errors[3], w, color="#9C27B0", alpha=0.8, label=f"Day+3 (avg {np.mean(errors[3]):.3f}%)")
ax3.axhline(y=0.5, color="#4CAF50", ls="--", lw=1, alpha=0.7, label="0.5% threshold")
ax3.set_title("Daily prediction error (rolling 30 days)", fontsize=13, fontweight="bold")
ax3.set_ylabel("Error, %", fontsize=11)
ax3.set_xlabel("Window", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.12)

plt.tight_layout()
plt.savefig("kzt_forecast.png", dpi=150, bbox_inches="tight")

print(f"\nSaved: kzt_forecast.png")

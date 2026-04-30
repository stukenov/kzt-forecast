"""
Прогноз USD/KZT на 1 год (252 торговых дня).
Модель поддерживает max_horizon=256, поэтому за один вызов.
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

df = yf.download("USDKZT=X", start="2020-01-01", end="2026-02-01", progress=False)
close = df["Close"].values.flatten()
dates = df.index

HORIZON = 252  # ~1 year of trading days

context = close[-1024:]
point, quantile = model.forecast(horizon=HORIZON, inputs=[context])

pred = point[0]
q10 = quantile[0, :, 1]
q90 = quantile[0, :, 9]
q20 = quantile[0, :, 2]
q80 = quantile[0, :, 8]

print(f"Текущий курс: {close[-1]:.2f} тг")
print(f"\nПрогноз:")
milestones = [1, 7, 30, 60, 90, 180, 252]
for d in milestones:
    if d <= HORIZON:
        p = pred[d-1]
        change = (p - close[-1]) / close[-1] * 100
        lo = q10[d-1]
        hi = q90[d-1]
        print(f"  +{d:>3} дней ({d//30} мес): {p:>7.2f} тг ({change:+.1f}%)  [{lo:.0f} — {hi:.0f}]")

# --- Графики ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="white",
                         gridspec_kw={"height_ratios": [2, 1]})

# Верхний: история 2 года + прогноз 1 год
ax = axes[0]
show_history = 504  # ~2 года
hist = close[-show_history:]
hist_x = np.arange(-len(hist), 0)
pred_x = np.arange(0, HORIZON)

ax.plot(hist_x, hist, color="#333", lw=1.8, label="History (2 years)")
ax.plot(pred_x, pred, color="#E91E63", lw=2.5, label="Forecast (1 year)")
ax.fill_between(pred_x, q10, q90, alpha=0.1, color="#E91E63", label="10-90%")
ax.fill_between(pred_x, q20, q80, alpha=0.15, color="#E91E63", label="20-80%")
ax.axvline(x=0, color="#BBB", ls=":", lw=1)
ax.axhline(y=500, color="#2196F3", ls="--", lw=1, alpha=0.5, label="500 level")

# Аннотации по кварталам
for d, label in [(63, "Q1"), (126, "Q2"), (189, "Q3"), (252, "Q4")]:
    if d <= HORIZON:
        ax.annotate(f"{label}\n{pred[d-1]:.0f} tg",
                    xy=(d, pred[d-1]), fontsize=9, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0", alpha=0.9))

ax.set_title("USD/KZT — 1 year forecast", fontsize=15, fontweight="bold")
ax.set_ylabel("Tenge per $1", fontsize=12)
ax.set_xlabel("Days from today", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.12)

# Нижний: % изменение от текущего курса
ax2 = axes[1]
change_pct = (pred - close[-1]) / close[-1] * 100
change_q10 = (q10 - close[-1]) / close[-1] * 100
change_q90 = (q90 - close[-1]) / close[-1] * 100

ax2.plot(pred_x, change_pct, color="#E91E63", lw=2.5, label="Forecast")
ax2.fill_between(pred_x, change_q10, change_q90, alpha=0.15, color="#E91E63")
ax2.axhline(y=0, color="#333", lw=1)
ax2.set_title("Expected change from current rate (%)", fontsize=13, fontweight="bold")
ax2.set_ylabel("Change, %", fontsize=11)
ax2.set_xlabel("Days from today", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.12)

plt.tight_layout()
plt.savefig("kzt_yearly.png", dpi=150, bbox_inches="tight")
print("\nSaved: kzt_yearly.png")

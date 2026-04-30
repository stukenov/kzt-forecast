"""Простой график: история + прогноз + факт."""
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
HORIZON = 30
CONTEXT = 1024

inputs, actuals = [], {}
for t in tickers:
    df = yf.download(t, start="2000-01-01", end="2026-02-01", progress=False)
    s = df["Close"].values.flatten()
    inputs.append(s[-(CONTEXT + HORIZON):-HORIZON])
    actuals[t] = s[-HORIZON:]

point, _ = model.forecast(horizon=HORIZON, inputs=inputs)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
for i, ticker in enumerate(tickers):
    actual = actuals[ticker]
    pred = point[i]
    ctx = inputs[i]

    t_ctx = np.arange(-60, 0)
    t_pred = np.arange(0, HORIZON)

    ax = axes[i]
    ax.plot(t_ctx, ctx[-60:], color="#333333", lw=2, label="History")
    ax.plot(t_pred, actual, color="#2196F3", lw=2.5, label="Actual")
    ax.plot(t_pred, pred, color="#E91E63", lw=2.5, label="Forecast")
    ax.axvline(x=0, color="#BBBBBB", ls=":", lw=1)
    ax.set_title(ticker, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("Days", fontsize=10)
    if i == 0:
        ax.set_ylabel("Price, $", fontsize=10)

plt.suptitle("TimesFM 2.5 — 30-day Forecast", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("stocks_simple.png", dpi=150, bbox_inches="tight")
print("Saved: stocks_simple.png")

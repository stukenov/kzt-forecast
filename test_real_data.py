"""
TimesFM 2.5 — тест на реальных данных: акции (AAPL, TSLA, GOOG) + файнтюнинг.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

import timesfm

# ============================================================
# Загрузка модели
# ============================================================
print("Загрузка модели...")
torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# ============================================================
# 1. АКЦИИ — инференс
# ============================================================
print("\n" + "=" * 60)
print("1. АКЦИИ — ИНФЕРЕНС")
print("=" * 60)

tickers_eval = ["AAPL", "TSLA", "GOOG"]
tickers_train = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "PFE", "KO",
    "PEP", "CSCO", "INTC", "CMCSA", "ADBE", "NFLX", "CRM", "ABT", "TMO", "NKE",
]
HORIZON = 30
CONTEXT = 512

stock_data = {}
print(f"Скачиваем данные по {len(tickers_train)} тикерам с 2000 года...")
for ticker in tickers_train:
    try:
        df = yf.download(ticker, start="2000-01-01", end="2026-02-01", progress=False)
        close = df["Close"].values.flatten()
        if len(close) > 300:
            stock_data[ticker] = close
            print(f"  {ticker}: {len(close)} дней")
    except Exception:
        pass
print(f"Загружено {len(stock_data)} тикеров")

# Прогноз
inputs = []
actuals = {}
for ticker in tickers_eval:
    series = stock_data[ticker]
    # Берём контекст (без последних HORIZON дней) и actual для сравнения
    context = series[-(CONTEXT + HORIZON) : -HORIZON]
    actual = series[-HORIZON:]
    inputs.append(context)
    actuals[ticker] = actual

point_forecast, quantile_forecast = model.forecast(horizon=HORIZON, inputs=inputs)

# Метрики и вывод
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, ticker in enumerate(tickers_eval):
    actual = actuals[ticker]
    pred = point_forecast[i]
    q10 = quantile_forecast[i, :, 1]  # 10th percentile
    q90 = quantile_forecast[i, :, 9]  # 90th percentile

    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100

    print(f"\n{ticker}:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  Actual (последние 5):  {actual[-5:].round(2)}")
    print(f"  Predict (последние 5): {pred[-5:].round(2)}")

    ctx = inputs[i]
    t_ctx = range(len(ctx))
    t_pred = range(len(ctx), len(ctx) + HORIZON)

    axes[i].plot(t_ctx[-60:], ctx[-60:], label="Context", color="blue")
    axes[i].plot(t_pred, actual, label="Actual", color="green", linestyle="--")
    axes[i].plot(t_pred, pred, label="Forecast", color="red")
    axes[i].fill_between(t_pred, q10, q90, alpha=0.2, color="red", label="10-90% CI")
    axes[i].set_title(f"{ticker} — MAE=${mae:.2f}, MAPE={mape:.1f}%")
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stocks_forecast.png", dpi=150)
print("\nГрафик сохранён: stocks_forecast.png")

# ============================================================
# 2. ФАЙНТЮНИНГ на акциях AAPL
# ============================================================
print("\n" + "=" * 60)
print("2. ФАЙНТЮНИНГ НА AAPL")
print("=" * 60)

inner = model.model
trainable = inner._orig_mod if hasattr(inner, "_orig_mod") else inner
PATCH_LEN = trainable.p
OUTPUT_LEN = trainable.o


class MultiStockDataset(Dataset):
    def __init__(self, all_series, context_patches=4, horizon=128, step=1):
        self.context_len = context_patches * PATCH_LEN
        self.horizon = horizon
        self.samples = []
        total = self.context_len + self.horizon
        for series in all_series:
            # Нормализуем каждый тикер отдельно
            mu, std = series.mean(), series.std()
            if std < 1e-8:
                continue
            normed = (series - mu) / std
            for start in range(0, len(normed) - total, step):
                ctx = normed[start : start + self.context_len]
                fut = normed[start + self.context_len : start + total]
                self.samples.append((ctx.astype(np.float32), fut.astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, fut = self.samples[idx]
        return (
            torch.from_numpy(ctx),
            torch.zeros(len(ctx), dtype=torch.bool),
            torch.from_numpy(fut),
        )


# Разделяем каждый тикер: 80% train, 20% val
train_series = []
val_series = []
for ticker, series in stock_data.items():
    split = int(len(series) * 0.8)
    train_series.append(series[:split])
    val_series.append(series[split:])

train_ds = MultiStockDataset(train_series, context_patches=4, horizon=OUTPUT_LEN, step=3)
val_ds = MultiStockDataset(val_series, context_patches=4, horizon=OUTPUT_LEN, step=5)
print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Переводим на MPS (Apple Silicon GPU) если доступно
if torch.backends.mps.is_available():
    device = torch.device("mps")
    trainable.to(device)
    trainable.device = device
    print("Используем MPS (Apple Silicon GPU)")
else:
    device = trainable.device
    print(f"Используем {device}")

# Замораживаем всё, кроме output head
trainable.train()
for p in trainable.parameters():
    p.requires_grad = False
for p in trainable.output_projection_point.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, trainable.parameters()), lr=5e-5
)
loss_fn = nn.MSELoss()

NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
    trainable.train()
    train_loss = 0.0
    for ctx, mask, fut in train_loader:
        ctx, mask, fut = ctx.to(device), mask.to(device), fut.to(device)
        B = ctx.shape[0]
        (_, _, out, _), _ = trainable(
            ctx.reshape(B, -1, PATCH_LEN), mask.reshape(B, -1, PATCH_LEN)
        )
        pred = out[:, -1, :OUTPUT_LEN]
        if pred.dim() == 3:
            pred = pred[..., 5]
        loss = loss_fn(pred, fut)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    trainable.eval()
    val_loss = 0.0
    with torch.no_grad():
        for ctx, mask, fut in val_loader:
            ctx, mask, fut = ctx.to(device), mask.to(device), fut.to(device)
            B = ctx.shape[0]
            (_, _, out, _), _ = trainable(
                ctx.reshape(B, -1, PATCH_LEN), mask.reshape(B, -1, PATCH_LEN)
            )
            pred = out[:, -1, :OUTPUT_LEN]
            if pred.dim() == 3:
                pred = pred[..., 5]
            val_loss += loss_fn(pred, fut).item()

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} — "
        f"train_loss: {train_loss/len(train_loader):.6f}, "
        f"val_loss: {val_loss/len(val_loader):.6f}"
    )

# ============================================================
# 3. СРАВНЕНИЕ ДО И ПОСЛЕ ФАЙНТЮНИНГА
# ============================================================
print("\n" + "=" * 60)
print("3. СРАВНЕНИЕ: ДО vs ПОСЛЕ ФАЙНТЮНИНГА")
print("=" * 60)

trainable.eval()
# Возвращаем на CPU для forecast API
trainable.to("cpu")
trainable.device = torch.device("cpu")
point_after, quantile_after = model.forecast(horizon=HORIZON, inputs=inputs)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, ticker in enumerate(tickers_eval):
    actual = actuals[ticker]
    pred_before = point_forecast[i]
    pred_after = point_after[i]

    mae_before = np.mean(np.abs(actual - pred_before))
    mae_after = np.mean(np.abs(actual - pred_after))
    print(f"{ticker}: MAE before=${mae_before:.2f}, after=${mae_after:.2f}")

    ctx = inputs[i]
    t_ctx = range(len(ctx))
    t_pred = range(len(ctx), len(ctx) + HORIZON)

    axes[i].plot(t_ctx[-60:], ctx[-60:], label="Context", color="blue")
    axes[i].plot(
        t_pred, actual, label="Actual", color="green", linestyle="--", linewidth=2
    )
    axes[i].plot(
        t_pred,
        pred_before,
        label=f"Before FT (MAE=${mae_before:.2f})",
        color="orange",
    )
    axes[i].plot(
        t_pred, pred_after, label=f"After FT (MAE=${mae_after:.2f})", color="red"
    )
    axes[i].set_title(ticker)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stocks_before_after_ft.png", dpi=150)
print("График сохранён: stocks_before_after_ft.png")
print("\nГотово!")

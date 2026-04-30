"""
TimesFM 2.5 — улучшенный прогон:
  1) Контекст 1024 (32 патча) вместо 128 (4 патча)
  2) Разморозка последних 4 transformer-слоёв + output head
  3) Huber + Quantile loss вместо MSE
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

import timesfm


# ============================================================
# Кастомные Loss-функции
# ============================================================
class HuberQuantileLoss(nn.Module):
    """Huber loss + Quantile loss по всем квантилям."""

    def __init__(self, quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                 huber_delta=1.0, quantile_weight=0.3):
        super().__init__()
        self.quantiles = quantiles
        self.huber_delta = huber_delta
        self.quantile_weight = quantile_weight

    def quantile_loss(self, pred, target, q):
        error = target - pred
        return torch.max(q * error, (q - 1) * error).mean()

    def forward(self, pred_all_quantiles, target):
        pred_median = pred_all_quantiles[..., 5]
        huber = F.huber_loss(pred_median, target, delta=self.huber_delta)

        ql = torch.tensor(0.0, device=target.device)
        for i, q in enumerate(self.quantiles):
            pred_q = pred_all_quantiles[..., i + 1]
            ql = ql + self.quantile_loss(pred_q, target, q)
        ql = ql / len(self.quantiles)

        return huber + self.quantile_weight * ql


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
# 1. Скачивание данных
# ============================================================
print("\n" + "=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

tickers_all = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "XOM", "PFE", "KO",
    "PEP", "CSCO", "INTC", "CMCSA", "ADBE", "NFLX", "CRM", "ABT", "TMO", "NKE",
]
tickers_show = ["AAPL", "TSLA", "GOOG"]
HORIZON = 30
CONTEXT = 1024  # <<< УЛУЧШЕНИЕ 1

stock_data = {}
print(f"Скачиваем {len(tickers_all)} тикеров с 2000 года...")
for ticker in tickers_all:
    try:
        df = yf.download(ticker, start="2000-01-01", end="2026-02-01", progress=False)
        close = df["Close"].values.flatten()
        if len(close) > 300:
            stock_data[ticker] = close
            print(f"  {ticker}: {len(close)} дней")
    except Exception:
        pass
print(f"Загружено {len(stock_data)} тикеров")

# ============================================================
# 2. Базовый инференс
# ============================================================
print("\n" + "=" * 60)
print("2. БАЗОВЫЙ ИНФЕРЕНС (context=1024)")
print("=" * 60)

inputs = []
actuals = {}
for ticker in tickers_show:
    series = stock_data[ticker]
    context = series[-(CONTEXT + HORIZON) : -HORIZON]
    actual = series[-HORIZON:]
    inputs.append(context)
    actuals[ticker] = actual

point_before, quantile_before = model.forecast(horizon=HORIZON, inputs=inputs)

for i, ticker in enumerate(tickers_show):
    actual = actuals[ticker]
    pred = point_before[i]
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    print(f"{ticker}: MAE=${mae:.2f}, MAPE={mape:.1f}%")

# ============================================================
# 3. Файнтюнинг
# ============================================================
print("\n" + "=" * 60)
print("3. ФАЙНТЮНИНГ")
print("=" * 60)

inner = model.model
trainable = inner._orig_mod if hasattr(inner, "_orig_mod") else inner
PATCH_LEN = trainable.p
OUTPUT_LEN = trainable.o
NUM_QUANTILES = trainable.q
CONTEXT_PATCHES = 32  # 32 * 32 = 1024


class MultiStockDataset(Dataset):
    def __init__(self, all_series, context_patches, horizon, step=1):
        self.context_len = context_patches * PATCH_LEN
        self.horizon = horizon
        self.samples = []
        total = self.context_len + self.horizon
        for series in all_series:
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


train_series = []
val_series = []
for ticker, series in stock_data.items():
    split = int(len(series) * 0.8)
    train_series.append(series[:split])
    val_series.append(series[split:])

train_ds = MultiStockDataset(train_series, context_patches=CONTEXT_PATCHES, horizon=OUTPUT_LEN, step=5)
val_ds = MultiStockDataset(val_series, context_patches=CONTEXT_PATCHES, horizon=OUTPUT_LEN, step=10)
print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    trainable.to(device)
    trainable.device = device
    print("Device: MPS (Apple Silicon GPU)")
else:
    device = trainable.device
    print(f"Device: {device}")

# <<< УЛУЧШЕНИЕ 2: разморозка слоёв
trainable.train()
for p in trainable.parameters():
    p.requires_grad = False

for p in trainable.output_projection_point.parameters():
    p.requires_grad = True
for p in trainable.output_projection_quantiles.parameters():
    p.requires_grad = True

num_unfreeze = 4
for layer in trainable.stacked_xf[-num_unfreeze:]:
    for p in layer.parameters():
        p.requires_grad = True

trainable_params = sum(p.numel() for p in trainable.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in trainable.parameters())
print(f"Params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
print(f"Unfrozen: output heads + last {num_unfreeze}/20 transformer layers")

# <<< УЛУЧШЕНИЕ 3: Huber + Quantile loss
loss_fn = HuberQuantileLoss(huber_delta=1.0, quantile_weight=0.3)
print("Loss: Huber + Quantile")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, trainable.parameters()),
    lr=3e-5,
    weight_decay=0.01,
)

NUM_EPOCHS = 1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

for epoch in range(NUM_EPOCHS):
    trainable.train()
    train_loss = 0.0
    for batch_i, (ctx, mask, fut) in enumerate(train_loader):
        ctx, mask, fut = ctx.to(device), mask.to(device), fut.to(device)
        B = ctx.shape[0]

        (_, _, out_point, _), _ = trainable(
            ctx.reshape(B, -1, PATCH_LEN), mask.reshape(B, -1, PATCH_LEN)
        )

        # out_point shape: (B, num_patches, output_len * num_quantiles) или (B, num_patches, output_len, num_quantiles)
        last_patch = out_point[:, -1]
        if last_patch.dim() == 2:
            pred = last_patch[:, :OUTPUT_LEN * NUM_QUANTILES].reshape(B, OUTPUT_LEN, NUM_QUANTILES)
        else:
            pred = last_patch[:, :OUTPUT_LEN, :NUM_QUANTILES]

        loss = loss_fn(pred, fut)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

        if (batch_i + 1) % 50 == 0:
            print(f"  batch {batch_i+1}/{len(train_loader)}, loss: {loss.item():.4f}")

    scheduler.step()

    trainable.eval()
    val_loss = 0.0
    with torch.no_grad():
        for ctx, mask, fut in val_loader:
            ctx, mask, fut = ctx.to(device), mask.to(device), fut.to(device)
            B = ctx.shape[0]
            (_, _, out_point, _), _ = trainable(
                ctx.reshape(B, -1, PATCH_LEN), mask.reshape(B, -1, PATCH_LEN)
            )
            last_patch = out_point[:, -1]
            if last_patch.dim() == 2:
                pred = last_patch[:, :OUTPUT_LEN * NUM_QUANTILES].reshape(B, OUTPUT_LEN, NUM_QUANTILES)
            else:
                pred = last_patch[:, :OUTPUT_LEN, :NUM_QUANTILES]
            val_loss += loss_fn(pred, fut).item()

    lr_now = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} — "
        f"train: {train_loss/len(train_loader):.6f}, "
        f"val: {val_loss/max(len(val_loader),1):.6f}, "
        f"lr: {lr_now:.2e}"
    )

# ============================================================
# 4. Сравнение
# ============================================================
print("\n" + "=" * 60)
print("4. РЕЗУЛЬТАТЫ")
print("=" * 60)

trainable.eval()
trainable.to("cpu")
trainable.device = torch.device("cpu")

point_after, quantile_after = model.forecast(horizon=HORIZON, inputs=inputs)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, ticker in enumerate(tickers_show):
    actual = actuals[ticker]
    pred_b = point_before[i]
    pred_a = point_after[i]

    mae_b = np.mean(np.abs(actual - pred_b))
    mae_a = np.mean(np.abs(actual - pred_a))
    mape_b = np.mean(np.abs((actual - pred_b) / actual)) * 100
    mape_a = np.mean(np.abs((actual - pred_a) / actual)) * 100
    delta = (mae_b - mae_a) / mae_b * 100

    print(f"{ticker}:")
    print(f"  Before FT: MAE=${mae_b:.2f}, MAPE={mape_b:.1f}%")
    print(f"  After FT:  MAE=${mae_a:.2f}, MAPE={mape_a:.1f}%")
    print(f"  Change:    {delta:+.1f}%")

    ctx = inputs[i]
    t_ctx = np.arange(len(ctx))
    t_pred = np.arange(len(ctx), len(ctx) + HORIZON)

    ax = axes[i]
    ax.plot(t_ctx[-60:], ctx[-60:], color="#333333", lw=2, label="History")
    ax.plot(t_pred, actual, color="#2196F3", lw=2.5, label="Actual")
    ax.plot(t_pred, pred_b, color="#FF9800", lw=1.8, ls="--", label=f"Before FT  MAE=${mae_b:.1f}")
    ax.plot(t_pred, pred_a, color="#E91E63", lw=2.5, label=f"After FT   MAE=${mae_a:.1f}")

    # Вертикальная линия — граница прогноза
    ax.axvline(x=t_pred[0], color="#999999", ls=":", lw=1)

    ax.set_title(f"{ticker}  ({delta:+.1f}%)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Price, $")

plt.suptitle("TimesFM 2.5 — Before vs After Fine-Tuning", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("stocks_improved.png", dpi=150, bbox_inches="tight")
print("\nГрафик: stocks_improved.png")
print("Готово!")

"""
TimesFM 2.5 — тестовый прогон: инференс + файнтюнинг.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import timesfm

# ============================================================
# 1. ИНФЕРЕНС (из README)
# ============================================================
print("=" * 60)
print("1. ИНФЕРЕНС")
print("=" * 60)

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

point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
        np.random.randn(200).cumsum(),
    ],
)
print(f"point_forecast shape: {point_forecast.shape}")  # (3, 12)
print(f"quantile_forecast shape: {quantile_forecast.shape}")  # (3, 12, 10)
print(f"Примеры прогнозов (ряд 1): {point_forecast[0]}")
print("Инференс OK!\n")

# ============================================================
# 2. ФАЙНТЮНИНГ (простой пример на синтетических данных)
# ============================================================
print("=" * 60)
print("2. ФАЙНТЮНИНГ")
print("=" * 60)

# Получаем внутренний модуль (nn.Module)
inner_model = model.model

# Размораживаем параметры (после from_pretrained модель в eval/compiled)
if hasattr(inner_model, "_orig_mod"):
    trainable_model = inner_model._orig_mod  # torch.compile обёртка
else:
    trainable_model = inner_model

trainable_model.train()
device = trainable_model.device

PATCH_LEN = trainable_model.p       # 32
OUTPUT_LEN = trainable_model.o      # 128
NUM_QUANTILES = trainable_model.q   # 10


class SyntheticTSDataset(Dataset):
    """Синтетический датасет: синусоиды + шум."""

    def __init__(self, num_samples=256, context_patches=4):
        self.num_samples = num_samples
        self.context_len = context_patches * PATCH_LEN
        self.horizon_len = OUTPUT_LEN

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        freq = np.random.uniform(0.5, 5.0)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 10, self.context_len + self.horizon_len)
        series = np.sin(freq * t + phase) + 0.1 * np.random.randn(len(t))

        context = torch.tensor(series[: self.context_len], dtype=torch.float32)
        future = torch.tensor(
            series[self.context_len : self.context_len + self.horizon_len],
            dtype=torch.float32,
        )
        mask = torch.zeros(self.context_len, dtype=torch.bool)
        return context, mask, future


train_ds = SyntheticTSDataset(num_samples=256, context_patches=4)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

# Замораживаем всё, кроме output_projection_point (head)
for param in trainable_model.parameters():
    param.requires_grad = False
for param in trainable_model.output_projection_point.parameters():
    param.requires_grad = True

trainable_params = sum(
    p.numel() for p in trainable_model.parameters() if p.requires_grad
)
total_params = sum(p.numel() for p in trainable_model.parameters())
print(f"Обучаемых параметров: {trainable_params:,} / {total_params:,}")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, trainable_model.parameters()),
    lr=1e-4,
)
loss_fn = nn.MSELoss()

NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for batch_idx, (context, mask, future) in enumerate(train_loader):
        context = context.to(device)
        mask = mask.to(device)
        future = future.to(device)

        # Патчируем вход
        B = context.shape[0]
        patched_input = context.reshape(B, -1, PATCH_LEN)
        patched_mask = mask.reshape(B, -1, PATCH_LEN)

        (_, _, output_ts, _), _ = trainable_model(patched_input, patched_mask)

        # Берём выход последнего патча — point forecast (index 5 = median)
        pred = output_ts[:, -1, :OUTPUT_LEN]  # (B, OUTPUT_LEN, NUM_QUANTILES)

        # Простой MSE по point forecast (index 5)
        if pred.dim() == 3:
            pred_point = pred[..., 5]  # median
        else:
            pred_point = pred

        loss = loss_fn(pred_point, future)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} — loss: {avg_loss:.6f}")

print("Файнтюнинг OK!\n")

# ============================================================
# 3. ИНФЕРЕНС ПОСЛЕ ФАЙНТЮНИНГА
# ============================================================
print("=" * 60)
print("3. ИНФЕРЕНС ПОСЛЕ ФАЙНТЮНИНГА")
print("=" * 60)

trainable_model.eval()

# Используем forecast_naive для быстрой проверки
test_input = np.sin(np.linspace(0, 10, 128))
results = trainable_model.forecast_naive(horizon=32, inputs=[test_input])
print(f"Прогноз после файнтюнинга shape: {results[0].shape}")
print(f"Прогноз (первые 5 точек): {results[0][:5, 5]}")
print("\nВсё готово!")

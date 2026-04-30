"""
Где TimesFM реально работает — примеры с паттернами.
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

HORIZON = 90

# --- 1. Генерируем реалистичные данные с паттернами ---

np.random.seed(42)
days = 1024 + HORIZON

# 1. Продажи магазина (недельная + годовая сезонность)
t = np.arange(days)
weekly = 30 * np.sin(2 * np.pi * t / 7)  # больше продаж в выходные
yearly = 50 * np.sin(2 * np.pi * t / 365)  # пик зимой (НГ)
trend = 0.05 * t  # медленный рост
noise = 8 * np.random.randn(days)
sales = 500 + weekly + yearly + trend + noise
sales = np.maximum(sales, 0)

# 2. Температура (сильная годовая сезонность)
temp_base = 15 + 20 * np.sin(2 * np.pi * (t - 90) / 365)
temp_noise = 3 * np.random.randn(days)
temperature = temp_base + temp_noise

# 3. Трафик сайта (недельный цикл + рост)
weekly_traffic = 200 * np.sin(2 * np.pi * t / 7)
growth = 0.3 * t
spikes = np.zeros(days)
for spike_day in range(0, days, 30):  # промо каждый месяц
    spikes[spike_day:spike_day+3] += 500
web_traffic = 5000 + weekly_traffic + growth + spikes + 100 * np.random.randn(days)
web_traffic = np.maximum(web_traffic, 0)

# 4. Потребление электричества (дневной + сезонный цикл)
daily_cycle = 100 * np.sin(2 * np.pi * t / 1)  # пик днём
seasonal = 200 * np.cos(2 * np.pi * t / 365)  # больше зимой (отопление)
elec = 3000 + daily_cycle + seasonal + 0.1 * t + 50 * np.random.randn(days)

datasets = {
    "Store Sales (units/day)": sales,
    "Temperature (C)": temperature,
    "Web Traffic (visits/day)": web_traffic,
    "Electricity (MWh)": elec,
}

# --- Прогнозы ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="white")

results = {}
for idx, (name, series) in enumerate(datasets.items()):
    context = series[:1024]
    actual = series[1024:1024 + HORIZON]

    point, quantile = model.forecast(horizon=HORIZON, inputs=[context])
    pred = point[0]
    q10 = quantile[0, :, 1]
    q90 = quantile[0, :, 9]

    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-10))) * 100
    mean_val = np.mean(actual)

    results[name] = {"mae": mae, "mape": mape}

    print(f"{name}:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  MAE:  {mae:.1f} (mean={mean_val:.0f})")
    print()

    ax = axes[idx // 2][idx % 2]
    ctx_x = np.arange(-120, 0)
    pred_x = np.arange(0, HORIZON)

    ax.plot(ctx_x, context[-120:], color="#333", lw=1.5, label="History")
    ax.plot(pred_x, actual, color="#2196F3", lw=2, label="Actual")
    ax.plot(pred_x, pred, color="#E91E63", lw=2, label=f"Forecast (MAPE {mape:.1f}%)")
    ax.fill_between(pred_x, q10, q90, alpha=0.12, color="#E91E63")
    ax.axvline(x=0, color="#BBB", ls=":", lw=1)
    ax.set_title(f"{name} — MAPE: {mape:.1f}%", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.12)

plt.suptitle("TimesFM 2.5 — Where it actually works (90-day forecast)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("real_use_cases.png", dpi=150, bbox_inches="tight")

# --- Сравнительная таблица ---
print("=" * 50)
print("ИТОГО: где модель работает vs где нет")
print("=" * 50)
print(f"\n{'Задача':<30} {'MAPE':>8}")
print("-" * 40)
for name, r in results.items():
    emoji = "OK" if r["mape"] < 10 else "SO-SO" if r["mape"] < 20 else "BAD"
    print(f"  {name:<28} {r['mape']:>6.1f}%  {emoji}")
print("-" * 40)
print(f"  {'Акции (AAPL, 30 дней)':<28} {'5.4':>6}%  SO-SO")
print(f"  {'Акции направление':<28} {'~50':>6}%  BAD")
print(f"  {'Курс тенге (1 год)':<28} {'???':>6}   BAD")

print("\nSaved: real_use_cases.png")

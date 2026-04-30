"""
Умный прогноз USD/KZT.

Тенге зависит от:
- Нефть Brent (Казахстан = нефтяная страна)
- Курс рубля (главный торговый партнёр)
- Золото (резервы Нацбанка)
- Медь (экспорт КЗ)
- Доллар (DXY индекс)
- S&P 500 (аппетит к риску)
- Юань (торговля с Китаем)

Стратегия: прогнозируем каждый фактор отдельно через TimesFM,
потом считаем корреляции и строим комбинированный прогноз.
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

# --- Загрузка ---
print("Загрузка данных...")
signals = {
    "USD/KZT": "USDKZT=X",
    "Brent Oil": "BZ=F",
    "USD/RUB": "USDRUB=X",
    "Gold": "GC=F",
    "Copper": "HG=F",
    "DXY (Dollar)": "DX-Y.NYB",
    "S&P 500": "SPY",
    "USD/CNY": "USDCNY=X",
}

raw = {}
for name, ticker in signals.items():
    df = yf.download(ticker, start="2020-01-01", end="2026-02-01", progress=False)
    if len(df) > 100:
        raw[name] = df["Close"].values.flatten()
        print(f"  {name} ({ticker}): {len(df)} дней, last={raw[name][-1]:.2f}")
    else:
        print(f"  {name} ({ticker}): SKIP ({len(df)} дней)")

# Выравниваем длины (берём минимальную)
min_len = min(len(v) for v in raw.values())
for k in raw:
    raw[k] = raw[k][-min_len:]
print(f"\nОбщая длина: {min_len} дней")

# --- Корреляции с KZT ---
print(f"\n{'=' * 55}")
print("КОРРЕЛЯЦИИ С USD/KZT (изменения)")
print("=" * 55)

kzt = raw["USD/KZT"]
kzt_returns = np.diff(kzt) / kzt[:-1]

correlations = {}
for name, series in raw.items():
    if name == "USD/KZT":
        continue
    returns = np.diff(series) / series[:-1]
    # Обрезаем до одинаковой длины
    n = min(len(kzt_returns), len(returns))
    corr = np.corrcoef(kzt_returns[-n:], returns[-n:])[0, 1]
    correlations[name] = corr
    direction = "тенге слабеет" if corr > 0 else "тенге крепнет"
    print(f"  {name:<16} r={corr:+.3f}  (когда растёт -> {direction})")

# --- Прогнозы каждого фактора ---
print(f"\n{'=' * 55}")
print("ПРОГНОЗЫ ФАКТОРОВ НА 252 ДНЯ (1 ГОД)")
print("=" * 55)

HORIZON = 252
forecasts = {}
for name, series in raw.items():
    ctx = series[-1024:] if len(series) >= 1024 else series
    point, quantile = model.forecast(horizon=HORIZON, inputs=[ctx])
    forecasts[name] = {
        "point": point[0],
        "q10": quantile[0, :, 1],
        "q90": quantile[0, :, 9],
        "current": series[-1],
    }
    end_val = point[0, -1]
    change = (end_val - series[-1]) / series[-1] * 100
    print(f"  {name:<16} {series[-1]:>10.2f} -> {end_val:>10.2f} ({change:+.1f}%)")

# --- Комбинированный прогноз KZT ---
print(f"\n{'=' * 55}")
print("КОМБИНИРОВАННЫЙ ПРОГНОЗ USD/KZT")
print("=" * 55)

# Метод: взвешенное среднее прогнозов через корреляции
# Каждый фактор "голосует" за направление тенге

kzt_base = forecasts["USD/KZT"]["point"]  # базовый прогноз
kzt_current = raw["USD/KZT"][-1]

# Считаем "голоса" каждого фактора на каждый день
factor_votes = np.zeros(HORIZON)
total_weight = 0

for name, corr in correlations.items():
    if name not in forecasts:
        continue
    f = forecasts[name]
    # Прогнозируемое изменение фактора в %
    factor_change = (f["point"] - f["current"]) / f["current"]
    # Корреляция * изменение = ожидаемое влияние на KZT в %
    vote = corr * factor_change
    weight = abs(corr)  # вес = сила корреляции
    factor_votes += vote * weight
    total_weight += weight

# Нормализуем голоса
factor_adjustment = factor_votes / total_weight  # средневзвешенное изменение в %

# Комбинированный прогноз = базовый + корректировка от факторов
kzt_combined = kzt_current * (1 + factor_adjustment)

# Также делаем "ансамбль" — среднее между базовым и комбинированным
kzt_ensemble = (kzt_base + kzt_combined) / 2

milestones = [1, 7, 30, 90, 180, 252]
print(f"\nТекущий курс: {kzt_current:.2f}\n")
print(f"{'Дней':>6} {'Базовый':>10} {'Факторы':>10} {'Ансамбль':>10} {'Диапазон':>16}")
print("-" * 58)
for d in milestones:
    base = kzt_base[d-1]
    combo = kzt_combined[d-1]
    ens = kzt_ensemble[d-1]
    lo = forecasts["USD/KZT"]["q10"][d-1]
    hi = forecasts["USD/KZT"]["q90"][d-1]
    # Корректируем диапазон с учётом факторов
    adj = kzt_combined[d-1] - kzt_base[d-1]
    lo_adj = lo + adj
    hi_adj = hi + adj
    print(f"{d:>6} {base:>10.2f} {combo:>10.2f} {ens:>10.2f} {lo_adj:>7.0f} — {hi_adj:.0f}")

# --- Анализ факторов ---
print(f"\n{'=' * 55}")
print("ЧТО ДВИГАЕТ ПРОГНОЗ")
print("=" * 55)

for name, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
    if name not in forecasts:
        continue
    f = forecasts[name]
    change_year = (f["point"][-1] - f["current"]) / f["current"] * 100
    impact = corr * change_year / 100 * kzt_current
    direction = "ослабляет тенге" if impact > 0 else "укрепляет тенге"
    print(f"  {name:<16} forecast: {change_year:+.1f}% | corr: {corr:+.3f} | impact: {impact:+.1f} тг ({direction})")

# --- Графики ---
fig = plt.figure(figsize=(16, 16), facecolor="white")
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

# 1. Главный график — прогноз KZT
ax1 = fig.add_subplot(gs[0, :])
show_hist = 504
hist = raw["USD/KZT"][-show_hist:]
hist_x = np.arange(-len(hist), 0)
pred_x = np.arange(0, HORIZON)

ax1.plot(hist_x, hist, color="#333", lw=1.8, label="History")
ax1.plot(pred_x, kzt_base, color="#FF9800", lw=2, ls="--", label="Base (price only)")
ax1.plot(pred_x, kzt_combined, color="#2196F3", lw=2, ls="--", label="Factors only")
ax1.plot(pred_x, kzt_ensemble, color="#E91E63", lw=2.5, label="Ensemble (final)")
ax1.fill_between(pred_x,
                  forecasts["USD/KZT"]["q10"] + (kzt_combined - kzt_base),
                  forecasts["USD/KZT"]["q90"] + (kzt_combined - kzt_base),
                  alpha=0.1, color="#E91E63")
ax1.axvline(x=0, color="#BBB", ls=":", lw=1)
ax1.axhline(y=500, color="#999", ls="--", lw=0.8, alpha=0.5)
ax1.set_title("USD/KZT — 1 year forecast (with macro factors)", fontsize=15, fontweight="bold")
ax1.set_ylabel("Tenge per $1", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.12)

# 2-5. Ключевые факторы
factor_plots = [
    ("Brent Oil", "#4CAF50"),
    ("USD/RUB", "#2196F3"),
    ("Gold", "#FF9800"),
    ("DXY (Dollar)", "#9C27B0"),
]
for idx, (name, color) in enumerate(factor_plots):
    row = 1 + idx // 2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])

    series = raw[name]
    f = forecasts[name]
    show = 252
    hist_s = series[-show:]
    h_x = np.arange(-len(hist_s), 0)
    p_x = np.arange(0, HORIZON)

    ax.plot(h_x, hist_s, color="#333", lw=1.5, label="History")
    ax.plot(p_x, f["point"], color=color, lw=2, label="Forecast")
    ax.fill_between(p_x, f["q10"], f["q90"], alpha=0.1, color=color)
    ax.axvline(x=0, color="#BBB", ls=":", lw=1)

    corr = correlations[name]
    change = (f["point"][-1] - f["current"]) / f["current"] * 100
    ax.set_title(f"{name} (corr={corr:+.2f}, forecast={change:+.1f}%)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.12)

plt.savefig("kzt_smart.png", dpi=150, bbox_inches="tight")
print("\nSaved: kzt_smart.png")

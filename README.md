# KZT Forecast

Прогнозирование курса USD/KZT (казахстанский тенге) с помощью [TimesFM](https://github.com/google-research/timesfm) — модели временных рядов от Google Research.

## Что внутри

| Скрипт | Описание |
|--------|----------|
| `kzt_forecast.py` | Базовый прогноз USD/KZT на 7 дней с rolling evaluation |
| `kzt_smart.py` | Прогноз с мульти-горизонтом и ансамблем контекстных окон |
| `kzt_yearly.py` | Годовой прогноз курса |
| `daily_dashboard.py` | Ежедневный дашборд с текущим прогнозом |
| `backtest.py` | Бэктест стратегии на исторических данных |
| `smart_backtest.py` | Бэктест с адаптивной стратегией |
| `multi_signal_backtest.py` | Мульти-сигнальный бэктест (несколько индикаторов) |
| `rolling_eval.py` | Rolling evaluation точности модели |
| `real_use_cases.py` | Примеры применения прогнозов |
| `test_run.py` | Быстрый тест модели |
| `test_real_data.py` | Тесты на реальных данных |
| `test_improved.py` | Расширенные тесты |

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python kzt_forecast.py
```

## Данные

Курсы загружаются автоматически через [yfinance](https://github.com/ranaroussi/yfinance) (тикер `USDKZT=X`).

## Модель

Используется [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) — 200M параметров, контекст до 16k, квантильные прогнозы.

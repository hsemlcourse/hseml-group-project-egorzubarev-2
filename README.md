# CP1: Laptop Price Prediction

Проект первого чекпоинта учебного ML-курса. Задача — регрессия: по характеристикам ноутбука предсказать его розничную цену в рублях.

Данные собираются из каталогов **Citilink** и **DNS**, дедуплицируются по совпадающим спецификациям (без привязки к источнику), сохраняются в `laptops.db` (SQLite) и `data/raw/laptops.csv`. Аналитика, EDA и обучение моделей — в `notebooks/cp1.ipynb`.

---

## Структура репозитория

```text
.
├── data/
│   ├── raw/
│   │   ├── laptops.csv               ← финальный сырой датасет (Citilink + DNS)
│   │   └── browser_batches/          ← JSON-снимки DNS из браузерной сессии
│   └── processed/
│       └── laptops_clean.csv         ← очищенный датасет с engineered-признаками
├── models/
│   ├── baseline_ridge.joblib         ← Ridge с feature engineering
│   ├── linreg_raw.joblib             ← LinearRegression без feature engineering
│   └── baseline_metrics.json         ← метрики обеих моделей на validation и test
├── notebooks/
│   └── cp1.ipynb                     ← основной аналитический ноутбук
├── src/
│   ├── project_config.py             ← константы, пути, URL, SEED
│   ├── http_utils.py                 ← HTTP-сессия с retry/backoff, заголовки
│   ├── browser_fetcher.py            ← Playwright-клиент для каталогов
│   ├── scraper_common.py             ← общий парсинг текста и нормализация признаков
│   ├── citilink_scraper.py           ← сбор карточек и страниц /properties/ Citilink
│   ├── dns_scraper.py                ← нормализация JSON-payload DNS
│   ├── browser_bridge_server.py      ← локальный HTTP-сервер для приёма данных из браузера
│   ├── pipeline.py                   ← CLI-оркестрация сбора и сохранения датасета
│   ├── finalize_dataset.py           ← финальное слияние Citilink + DNS из артефактов
│   ├── storage.py                    ← дедупликация, fingerprint, экспорт в SQLite/CSV
│   ├── preprocessing.py              ← очистка и feature engineering
│   ├── modeling.py                   ← разбиение, baseline-модели, метрики
│   ├── train.py                      ← обучение и сохранение обеих моделей
│   └── predict.py                    ← предсказание цены по характеристикам ноутбука (CLI)
├── tests/
│   └── test.py                       ← 6 unit-тестов
├── Dockerfile
├── pyproject.toml                    ← конфиг Ruff и pytest
├── requirements.txt
├── laptops.db                        ← SQLite-база (таблица laptops)
└── README.md
```

---

## Как устроен сбор данных

### Citilink

1. Playwright рендерит страницы каталога `https://www.citilink.ru/catalog/noutbuki/` и извлекает ссылки на товары с ценами из DOM.
2. Для каждой карточки делаются обычные HTTP-запросы (`requests`) на страницу товара и на `/product/.../properties/` — там хранятся полные характеристики.
3. HTML парсится через BeautifulSoup + lxml, признаки нормализуются в `src/scraper_common.py`.

### DNS

DNS защищён антиботом Qrator — прямые Python-запросы получают 401/403. Рабочий путь:

1. Каталожные карточки и JSON-детали товаров (`/pwa/pwa/get-product/?id=GUID`) собраны через **аутентифицированную браузерную сессию** (browser-tools VS Code).
2. Данные отправлены POST-запросами в `src/browser_bridge_server.py` (локальный сервер на `127.0.0.1:8765`) и сохранены в `data/raw/browser_batches/`.
3. `src/finalize_dataset.py` читает эти JSON-файлы и объединяет их с Citilink-записями в один финальный датасет.

### Дедупликация

Каждая запись получает SHA-256 fingerprint по нормализованным полям (название, CPU, GPU, RAM, SSD, экран и т.д.). Дубли из разных источников схлопываются в одну строку; `price_rub` берётся как медиана найденных цен.

---

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python -m playwright install chromium
```

### 2. Сбор данных (Citilink)

```bash
# Полный скрейп Citilink (занимает ~30–60 минут):
python -m src.pipeline --skip-dns

# Только первые 3 страницы каталога (быстрая проверка):
python -m src.pipeline --max-citilink-pages 3 --skip-dns

# С видимым браузером (удобно при отладке):
python -m src.pipeline --max-citilink-pages 3 --skip-dns --headed
```

> DNS через CLI недоступен из-за антибота. DNS-данные уже сохранены в `data/raw/browser_batches/`.

### 3. Финализация датасета (слияние Citilink + DNS)

```bash
python -m src.finalize_dataset
```

Команда читает Citilink-записи из `data/raw/laptops.csv` и DNS-снимки из `data/raw/browser_batches/dns_products_repair_*.json`, объединяет и перезаписывает `data/raw/laptops.csv` и `laptops.db`.

### 4. Обучение моделей

```bash
python -m src.train
```

Обучает Ridge (со всеми признаками) и LinearRegression (без feature engineering), сохраняет:
- `models/baseline_ridge.joblib`
- `models/linreg_raw.joblib`
- `models/baseline_metrics.json`

### 5. Предсказание цены конкретного ноутбука

```bash
# Ridge (с feature engineering, по умолчанию):
python -m src.predict \
    --brand Lenovo --ram_gb 16 --storage_gb 512 \
    --screen_diagonal_inch 15.6 --cpu "Intel Core i5-12450H" \
    --gpu "NVIDIA GeForce RTX 3050" --os "No OS"
# → Предсказанная цена: 106 521 руб.

# LinearRegression (без feature engineering):
python -m src.predict \
    --brand Lenovo --ram_gb 16 --storage_gb 512 \
    --screen_diagonal_inch 15.6 --cpu "Intel Core i5-12450H" \
    --gpu "NVIDIA GeForce RTX 3050" --os "No OS" --model linreg
# → Предсказанная цена: 124 623 руб.
```

Пропущенные признаки автоматически заполняются медианой или модой из обучающих данных.

Доступные флаги `predict.py`:

| Флаг | Тип | Описание |
|---|---|---|
| `--brand` | str | Бренд, напр. `Lenovo`, `ASUS` |
| `--screen_diagonal_inch` | float | Диагональ экрана (дюймы) |
| `--screen_resolution` | str | Разрешение, напр. `1920x1080` |
| `--matrix_type` | str | Тип матрицы, напр. `IPS` |
| `--cpu` | str | Процессор, напр. `Intel Core i5-12450H` |
| `--gpu` | str | Видеокарта, напр. `NVIDIA GeForce RTX 3050` |
| `--ram_gb` | float | Объём ОЗУ (ГБ) |
| `--ram_type` | str | Тип памяти, напр. `DDR5` |
| `--storage_gb` | float | Объём накопителя (ГБ) |
| `--os` | str | ОС, напр. `No OS` или `Windows 11` |
| `--weight_kg` | float | Вес (кг) |
| `--model` | `ridge`\|`linreg` | Модель (по умолчанию `ridge`) |

### 6. Запуск ноутбука

```bash
jupyter lab
# открыть notebooks/cp1.ipynb и запустить все ячейки сверху вниз
```

Ноутбук работает поверх уже сохранённых `data/raw/laptops.csv` и `laptops.db` — живой скрейп не нужен.

---

## Описание ноутбука cp1.ipynb

| Секция | Содержание |
|---|---|
| **Загрузка** | Читает `data/raw/laptops.csv`, сверяет кол-во строк с `laptops.db` |
| **Очистка / Feature engineering** | `clean_laptops_dataframe()`: типы, фильтр выбросов цены (> 1 млн руб.), `resolution_width/height`, `cpu_vendor`, `gpu_vendor`, `is_discrete_gpu`, `is_gaming_series` |
| **EDA** | Гистограмма цен, топ-12 брендов, scatter RAM vs цена, scatter SSD vs цена, тепловая карта корреляций |
| **Разбиение** | train / validation / test ≈ 70 / 15 / 15, `SEED = 42` |
| **Baseline** | `DummyRegressor` (медиана) + Ridge с `OneHotEncoder` и `StandardScaler` |
| **LinearRegression (raw)** | Линейная регрессия только на сырых признаках, без engineered-колонок |
| **Сохранение** | `models/baseline_ridge.joblib`, `models/linreg_raw.joblib`, `models/baseline_metrics.json` |

---

## Результаты baseline

| Модель | Стадия | MAE (руб.) | RMSE (руб.) | R² |
|---|---|---:|---:|---:|
| DummyRegressor (медиана) | validation | 64 774 | 106 868 | −0.16 |
| Ridge (все признаки) | validation | 19 767 | 37 929 | 0.853 |
| LinearRegression (raw) | validation | 21 923 | 40 229 | 0.835 |
| Ridge (все признаки) | **test** | 18 660 | 29 577 | 0.888 |
| LinearRegression (raw) | **test** | 19 207 | 29 390 | **0.890** |

Медиана цены в датасете ~93 000 руб. MAE Ridge ~18–19 тыс. руб. ≈ 20% от медианы.

---

## Описание модулей `src/`

| Модуль | За что отвечает |
|---|---|
| `project_config.py` | Пути к файлам, URL каталогов, `SEED`, лимиты ценовых выбросов, задержки |
| `http_utils.py` | `requests.Session` с retry/backoff, случайный User-Agent, задержки между запросами |
| `browser_fetcher.py` | Playwright-клиент: рендер каталога Citilink/DNS, извлечение карточек из DOM |
| `scraper_common.py` | Регэкспы для цены, CPU, GPU, RAM, SSD, OS, матрицы; датакласс `LaptopRecord` |
| `citilink_scraper.py` | Запрос страницы товара + `/properties/`, парсинг BeautifulSoup, сборка `LaptopRecord` |
| `dns_scraper.py` | Разворачивание `characteristics` из JSON-payload DNS, сборка `LaptopRecord` |
| `browser_bridge_server.py` | Локальный HTTP-сервер (`:8765`) для приёма JSON-батчей из браузерной сессии |
| `pipeline.py` | CLI: `--skip-citilink`, `--skip-dns`, `--max-*-pages`, `--headed` |
| `finalize_dataset.py` | Слияние Citilink (из CSV) + DNS (из JSON-батчей), дедупликация, сохранение |
| `storage.py` | SHA-256 fingerprint, `merge_duplicates`, запись в CSV и SQLite |
| `preprocessing.py` | `load_raw_dataset`, `clean_laptops_dataframe`, `add_engineered_features` |
| `modeling.py` | `split_dataset`, `build_preprocessor`, `build_baseline_pipeline`, `build_linreg_raw_pipeline`, `evaluate_regression`, `fit_and_score`, `save_model` |
| `train.py` | CLI: обучает Ridge + LinearRegression, сохраняет `.joblib` и `baseline_metrics.json` |
| `predict.py` | CLI: принимает характеристики ноутбука, выводит предсказанную цену |

---

## Тесты

```bash
pytest           # запустить все тесты
ruff check src tests   # линтер
```

В `tests/test.py` 6 unit-тестов:

| Тест | Что проверяет |
|---|---|
| `test_parse_core_features_extracts_main_laptop_fields` | Парсер текста корректно извлекает бренд, диагональ, CPU, RAM, SSD, OS |
| `test_merge_duplicates_collapses_identical_records` | Два одинаковых ноутбука из разных источников схлопываются в одну строку с медианной ценой |
| `test_clean_laptops_dataframe_adds_engineered_columns` | После очистки появляются `resolution_width`, `cpu_vendor`, `is_gaming_series` |
| `test_clean_laptops_dataframe_drops_unreasonable_price_outliers` | Строки с ценой > 1 млн руб. отфильтровываются |
| `test_build_citilink_record_prefers_catalog_price` | Цена из карточки каталога приоритетнее ошибочной цены со страницы товара |
| `test_evaluate_regression_returns_expected_metrics` | Функция метрик возвращает `mae`, `rmse`, `r2` с правильными числами |

---

## Docker

```bash
docker build -t laptop-cp1 .
docker run --rm -p 8888:8888 laptop-cp1
```

Контейнер поднимает JupyterLab на порту 8888 с установленным Chromium для Playwright.

---

## Зависимости

Python 3.11. Основные пакеты:

```
beautifulsoup4   lxml          playwright    requests
pandas           numpy         scikit-learn  joblib
matplotlib       seaborn       tqdm
jupyterlab       pytest        ruff
```

Полный список с версиями — в `requirements.txt`.

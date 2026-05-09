# Laptop Price Prediction (CP1 + CP2)

Учебный ML-проект. Задача — регрессия: по характеристикам ноутбука предсказать его розничную цену в рублях.

Данные собираются из каталогов **Citilink** и **DNS**, дедуплицируются по совпадающим спецификациям (без привязки к источнику), сохраняются в `laptops.db` (SQLite) и `data/raw/laptops.csv`.

- `notebooks/cp1.ipynb` — EDA, очистка, baseline-модели (Ridge, LinearRegression).
- `notebooks/cp2.ipynb` — 10 экспериментов с подбором гиперпараметров, PCA, финальная модель (RandomForest, **test MAE = 13 880 руб., R² = 0.924**), feature importance.

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
│   ├── baseline_ridge.joblib         ← CP1: Ridge с feature engineering
│   ├── linreg_raw.joblib             ← CP1: LinearRegression без FE
│   ├── baseline_metrics.json         ← CP1: метрики baseline-моделей
│   ├── final_model.joblib            ← CP2: финальная модель (RandomForest tuned)
│   ├── experiments.csv               ← CP2: таблица 10 экспериментов
│   └── experiments.json              ← CP2: метрики и гиперпараметры
├── notebooks/
│   ├── cp1.ipynb                     ← CP1: EDA + baseline
│   └── cp2.ipynb                     ← CP2: 10 моделей, PCA, финал, importance
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
│   ├── experiments.py                ← CP2: 10 моделей, GridSearch/RandomSearch, PCA, финал
│   ├── train.py                      ← CP1: обучение и сохранение обеих baseline-моделей
│   └── predict.py                    ← предсказание цены по характеристикам ноутбука (CLI)
├── scripts/
│   └── build_cp2_notebook.py         ← скрипт генерации cp2.ipynb (one-shot, не runtime)
├── tests/
│   └── test.py                       ← 9 unit-тестов
├── Dockerfile
├── docker-compose.yml                ← jupyter lab на :8888
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

**CP1 — baseline:**

```bash
python -m src.train
```

Обучает Ridge (со всеми признаками) и LinearRegression (без feature engineering), сохраняет:
- `models/baseline_ridge.joblib`
- `models/linreg_raw.joblib`
- `models/baseline_metrics.json`

**CP2 — эксперименты и финальная модель:**

```bash
python -m src.experiments
```

Гоняет 10 конфигураций (LinReg, Ridge, Lasso, KNN, DecisionTree, RandomForest, GradientBoosting, LightGBM, Ridge+PCA) с GridSearch/RandomSearch на 5-fold CV, выбирает лучшую по `MAE_val`, переобучает на `train+val`, мерит на `test`, сохраняет:
- `models/experiments.csv` — таблица экспериментов
- `models/experiments.json` — полные метрики и гиперпараметры
- `models/final_model.joblib` — финальная модель

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

### 6. Запуск ноутбуков

```bash
jupyter lab
# notebooks/cp1.ipynb — EDA + baseline (CP1)
# notebooks/cp2.ipynb — эксперименты + финал (CP2)
# запустить все ячейки сверху вниз
```

Оба ноутбука работают поверх уже сохранённых `data/raw/laptops.csv` и `laptops.db` — живой скрейп не нужен. `cp2.ipynb` независим от `cp1.ipynb`: можно открывать любой.

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

## Описание ноутбука cp2.ipynb

| Секция | Содержание |
|---|---|
| **Данные и сплит** | Тот же `train/val/test` 70/15/15, `SEED = 42` — для сопоставимости с CP1 |
| **10 экспериментов** | Полный прогон `run_all()` из `src/experiments.py`: Linear (raw/FE), Ridge+grid, Lasso+grid, KNN+grid, DecisionTree+grid, RandomForest+random, GradientBoosting, LightGBM+random, Ridge+PCA |
| **PCA** | Scree-plot накопленной дисперсии, scatter первых 2 PC с раскраской по `log10(price)` |
| **Финальная модель** | `select_and_finalize()`: лучшая по `MAE_val`, переобучение на `train+val`, метрики на `test`, сохранение в `models/final_model.joblib` |
| **Интерпретируемость** | Top-15 признаков по `feature_importance` из `RandomForest` с восстановлением имён через `ColumnTransformer.get_feature_names_out()` |
| **CP1 vs CP2** | Сравнительная таблица: CP1 baseline (Ridge, LinearRegression raw) vs CP2 final (RandomForest) |

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

## CP2: эксперименты, ансамбли и финальная модель

### Запуск всех экспериментов

```bash
python -m src.experiments
```

Команда строит тот же `train/val/test` сплит (`SEED=42`), обучает 10 конфигураций моделей с подбором гиперпараметров (5-fold CV, `scoring=neg_MAE`), пишет таблицу в `models/experiments.csv` и `models/experiments.json`, выбирает лучшую по `MAE_val`, переобучает на `train+val`, мерит на `test` и сохраняет в `models/final_model.joblib`.

### Список экспериментов

| # | Имя | Поиск | Гиперпараметры |
|---|---|---|---|
| 1 | LinearRegression_raw | — | без FE |
| 2 | LinearRegression_FE | — | + FE-фичи |
| 3 | Ridge_FE_grid | Grid | alpha ∈ {0.1…100} |
| 4 | Lasso_FE_grid | Grid | alpha ∈ {0.1…1000} |
| 5 | KNN_FE_grid | Grid | n_neighbors, weights |
| 6 | DecisionTree_FE_grid | Grid | max_depth, min_samples_leaf |
| 7 | RandomForest_random | Random (10) | n_estimators, max_depth, min_samples_leaf, max_features |
| 8 | GradientBoosting | — | n_estimators=400, lr=0.05 |
| 9 | LightGBM_random | Random (15) | n_estimators, lr, max_depth, num_leaves, min_child_samples |
| 10 | Ridge_PCA95 | — | PCA до 95% дисперсии + Ridge |

### Таблица результатов (отсортирована по MAE_val)

| Модель | MAE val | RMSE val | R² val | Best params |
|---|---:|---:|---:|---|
| **RandomForest_random** | **17 648** | 37 484 | 0.857 | `n_estimators=400, max_depth=20, min_samples_leaf=1, max_features=0.5` |
| GradientBoosting | 18 959 | 38 930 | 0.845 | (default 400 / lr=0.05 / depth=4) |
| Ridge_FE_grid | 19 767 | 37 929 | 0.853 | `alpha=1.0` |
| Lasso_FE_grid | 20 657 | 39 764 | 0.839 | `alpha=10.0` |
| KNN_FE_grid | 20 964 | 44 413 | 0.799 | `n_neighbors=5, weights=distance` |
| LightGBM_random | 21 343 | 40 606 | 0.832 | `n_estimators=800, lr=0.1, num_leaves=127` |
| LinearRegression_FE | 21 703 | 40 131 | 0.836 | — |
| LinearRegression_raw | 21 922 | 40 229 | 0.835 | — |
| DecisionTree_FE_grid | 22 947 | 44 976 | 0.794 | `max_depth=None, min_samples_leaf=1` |
| Ridge_PCA95 | 23 088 | 41 219 | 0.827 | (PCA до 95%) |

### Финальная модель

| Стадия | MAE | RMSE | R² |
|---|---:|---:|---:|
| Validation | 17 648 | 37 484 | 0.857 |
| **Test** | **13 880** | **23 988** | **0.924** |

**RandomForest** уверенно бьёт CP1 baseline-Ridge (~−5 000 руб. MAE на test). Линейные модели не ловят нелинейные взаимодействия (бренд × GPU × RAM → premium-цена), один DecisionTree переобучается, KNN страдает от высокой размерности после OneHot, LightGBM на 1.5k наблюдений чуть нестабильнее. PCA сохраняет 95% дисперсии в десятках компонент, но для линейных моделей здесь не помогает — теряется структура категорий. PCA полезнее как 2D-визуализация в `cp2.ipynb`.

### Воспроизведение в ноутбуке

```bash
jupyter lab
# открыть notebooks/cp2.ipynb и запустить все ячейки сверху вниз
```

`cp2.ipynb` независим от `cp1.ipynb` и поверх готового `data/raw/laptops.csv` повторяет: сплит → 10 экспериментов → PCA-визуализацию → финальную модель → топ-15 признаков по `feature_importance` → сравнение с CP1.

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
| `experiments.py` | CP2: 10 моделей с GridSearch/RandomSearch, PCA, выбор финальной модели, сохранение в `models/experiments.{csv,json}` и `models/final_model.joblib` |
| `train.py` | CLI: обучает Ridge + LinearRegression, сохраняет `.joblib` и `baseline_metrics.json` |
| `predict.py` | CLI: принимает характеристики ноутбука, выводит предсказанную цену |

---

## Тесты

```bash
pytest           # запустить все тесты
ruff check src tests   # линтер
```

В `tests/test.py` 9 unit-тестов:

| Тест | Что проверяет |
|---|---|
| `test_parse_core_features_extracts_main_laptop_fields` | Парсер текста корректно извлекает бренд, диагональ, CPU, RAM, SSD, OS |
| `test_merge_duplicates_collapses_identical_records` | Два одинаковых ноутбука из разных источников схлопываются в одну строку с медианной ценой |
| `test_clean_laptops_dataframe_adds_engineered_columns` | После очистки появляются `resolution_width`, `cpu_vendor`, `is_gaming_series` |
| `test_clean_laptops_dataframe_drops_unreasonable_price_outliers` | Строки с ценой > 1 млн руб. отфильтровываются |
| `test_build_citilink_record_prefers_catalog_price` | Цена из карточки каталога приоритетнее ошибочной цены со страницы товара |
| `test_evaluate_regression_returns_expected_metrics` | Функция метрик возвращает `mae`, `rmse`, `r2` с правильными числами |
| `test_clean_text_handles_none_and_non_strings` | `clean_text` устойчив к `None` и пустым значениям |
| `test_dns_record_returns_none_when_price_is_missing` | DNS-парсер возвращает `price_rub=None`, а не `0`, если цены нет |
| `test_dns_record_falls_back_to_card_price_text` | DNS-парсер берёт цену из `priceText` карточки, если `product.price=0` |

---

## Docker

Через `docker-compose` (рекомендуется):

```bash
docker-compose up --build
# JupyterLab на http://localhost:8888 без токена
```

Или напрямую через `docker`:

```bash
docker build -t laptop-prediction .
docker run --rm -p 8888:8888 laptop-prediction
```

Контейнер поднимает JupyterLab на порту 8888 с установленным Chromium для Playwright. Для запуска экспериментов внутри контейнера:

```bash
docker-compose exec notebook python -m src.experiments
```

---

## Зависимости

Python 3.11. Основные пакеты:

```
beautifulsoup4   lxml          playwright    requests
pandas           numpy         scikit-learn  joblib
lightgbm         matplotlib    seaborn       tqdm
jupyterlab       pytest        ruff
```

Полный список с версиями — в `requirements.txt`.

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("stacking.log"), logging.StreamHandler()]
)

# Пример данных
np.random.seed(42)
X, y = np.random.rand(10000, 100), np.random.rand(10000)

# Преобразование в pandas.DataFrame с именами признаков
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Базовые модели
base_models = [
    ('lgbm', LGBMRegressor(n_estimators=100, max_depth=5, n_jobs=2)),  # Уменьшено n_jobs
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=2))  # Уменьшено n_jobs
]

# Мета-модель
meta_model = LinearRegression()

# Стекинг
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    n_jobs=2  # Уменьшено количество ядер
)

# Оценка стекинга с помощью кросс-валидации
logging.info("Оценка стекинга с помощью кросс-валидации...")
y_pred_cv = cross_val_predict(stacking_model, X_train, y_train, cv=5, n_jobs=2)
mae_cv = mean_absolute_error(y_train, y_pred_cv)
logging.info(f"Средний MAE на кросс-валидации: {mae_cv:.4f}")

# Обучение стекинга на всех данных
logging.info("Обучение стекинга моделей...")
stacking_model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
logging.info("Прогнозирование на тестовой выборке...")
y_pred = stacking_model.predict(X_test)

# Оценка качества на тестовой выборке
mae_test = mean_absolute_error(y_test, y_pred)
logging.info(f"MAE на тестовой выборке: {mae_test:.4f}")
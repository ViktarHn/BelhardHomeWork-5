import matplotlib.pyplot as plt
import joblib
from utils import load_data, convert_sparse_to_dense
import logging
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
data_dict = load_data()
X_test = data_dict['X_test']
y_test = data_dict['y_test']

# Преобразование разреженных матриц в плотные массивы (если необходимо)
X_test, _ = convert_sparse_to_dense(X_test, X_test)

# Загрузка всех моделей
models = {
    "Linear Regression": joblib.load('linear_regression_model.joblib'),
    "Random Forest": joblib.load('random_forest_model.joblib'),
    "Gradient Boosting": joblib.load('gradient_boosting_model.joblib'),
    "XGBoost": joblib.load('xgboost_model.joblib'),
    "CatBoost": joblib.load('catboost_model.joblib')
}

# Анализ остатков для всех моделей
for model_name, model in models.items():
    logging.info(f"Анализ остатков для модели: {model_name}...")
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Визуализация остатков (scatter plot)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, label='Остатки')
    plt.axhline(y=0, color='r', linestyle='--', label='Нулевая линия')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title(f'Остатки для {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Гистограмма остатков
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, label='Распределение остатков')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.title(f'Распределение остатков для {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Сохранение остатков в файл (опционально)
    np.save(f'{model_name.replace(" ", "_").lower()}_residuals.npy', residuals)
    logging.info(f"Остатки для модели {model_name} сохранены.")
import matplotlib.pyplot as plt
import joblib
from utils import load_data, convert_sparse_to_dense
import logging
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
data_dict = load_data()
X_train = data_dict['X_train']
y_train = data_dict['y_train']

# Преобразование разреженных матриц в плотные массивы (если необходимо)
X_train, _ = convert_sparse_to_dense(X_train, X_train)

# Загрузка всех моделей
models = {
    "Random Forest": joblib.load('random_forest_model.joblib'),
    "Gradient Boosting": joblib.load('gradient_boosting_model.joblib'),
    "XGBoost": joblib.load('xgboost_model.joblib'),
    "CatBoost": joblib.load('catboost_model.joblib')
}

# Визуализация важности признаков
for model_name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        logging.info(f"Анализ важности признаков для модели: {model_name}...")
        feature_importance = model.feature_importances_

        # Получение названий признаков (если доступно)
        try:
            feature_names = model.feature_names_in_
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]

        # Сортировка признаков по важности
        sorted_idx = np.argsort(feature_importance)[-20:]  # Топ-20 признаков
        sorted_feature_names = [feature_names[i] for i in sorted_idx]
        sorted_importance = feature_importance[sorted_idx]

        # Построение графика
        plt.figure(figsize=(12, 10))  # Увеличенный размер графика
        plt.barh(sorted_feature_names, sorted_importance, align='center', color='skyblue')
        plt.xlabel('Важность признаков', fontsize=12)
        plt.ylabel('Признаки', fontsize=12)
        plt.title(f'Важность признаков в {model_name}', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()  # Улучшение расположения элементов

        # Отображение графика
        plt.show()

        # Сохранение графика в файл (опционально)
        plt.savefig(f'{model_name.replace(" ", "_").lower()}_feature_importance.png', bbox_inches='tight')
        plt.close()  # Закрытие графика для освобождения памяти
        logging.info(f"График важности признаков для {model_name} сохранен.")
    else:
        logging.warning(f"Модель {model_name} не поддерживает атрибут feature_importances_.")
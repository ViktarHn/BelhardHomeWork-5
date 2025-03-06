import joblib
import matplotlib.pyplot as plt
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка лучшей модели
best_model = joblib.load('best_model.joblib')
logging.info("Лучшая модель успешно загружена.")

# Загрузка имен признаков
feature_names = joblib.load('feature_names.joblib')
logging.info("Имена признаков успешно загружены.")

# Получение важности признаков
if hasattr(best_model, 'feature_importances_'):
    feature_importance = best_model.feature_importances_
else:
    logging.warning("Модель не поддерживает атрибут feature_importances_.")
    feature_importance = None

if feature_importance is not None:
    # Сортировка признаков по важности
    sorted_idx = np.argsort(feature_importance)
    sorted_feature_names = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    # Визуализация важности признаков
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_importance)), sorted_importance, align='center')
    plt.yticks(range(len(sorted_importance)), sorted_feature_names)
    plt.xlabel('Важность признаков')
    plt.ylabel('Признаки')
    plt.title('Важность признаков в лучшей модели')
    plt.show()
    logging.info("График важности признаков построен.")
else:
    logging.info("Визуализация важности признаков недоступна для этой модели.")
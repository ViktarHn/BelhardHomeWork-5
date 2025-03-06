import joblib
import matplotlib.pyplot as plt
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка лучшей модели
best_model = joblib.load('best_model.joblib')
logging.info("Лучшая модель успешно загружена.")

# Загрузка предобработанных данных
data = joblib.load('preprocessed_data.joblib')
X = data['X']
y = data['y']

# Предсказания модели
y_pred = best_model.predict(X)

# Остатки
residuals = y - y_pred

# Визуализация остатков
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Остатки модели')
plt.show()
logging.info("График остатков построен.")
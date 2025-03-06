import joblib
import matplotlib.pyplot as plt
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_name):
    """Загрузка модели из файла."""
    filename = f"{model_name.replace(' ', '_').lower()}_model.joblib"
    if not os.path.exists(filename):
        logging.error(f"Файл модели {filename} не найден!")
        return None
    return joblib.load(filename)

# Загрузка данных и метрик
try:
    data = joblib.load('preprocessed_data.joblib')
    X, y = data['X'], data['y']
    results = joblib.load('model_results.joblib')
    best_model_name = max(results, key=lambda k: results[k]['R²'])
    logging.info("Данные и метрики успешно загружены.")
except Exception as e:
    logging.error(f"Ошибка загрузки данных: {str(e)}")
    exit()

# Список всех моделей
models = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "CatBoost"
]

# --------------------------------------------
# 1. Отдельные графики для каждой модели
# --------------------------------------------
logging.info("Построение отдельных графиков для каждой модели...")
for model_name in models:
    model = load_model(model_name)
    if model is None:
        continue
    
    y_pred = model.predict(X)
    
    # Создание фигуры
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5, label=model_name)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Предсказания модели: {model_name}')
    plt.legend()
    plt.grid(True)
    
    # Отображение графика
    plt.show(block=False)  # Не блокирует выполнение кода
    plt.pause(2)  # Пауза для просмотра (в секундах)
    
    # Сохранение графика
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_predictions.png")
    plt.close()
    logging.info(f"График для {model_name} сохранен.")

# --------------------------------------------
# 2. Общий график всех моделей
# --------------------------------------------
logging.info("Построение общего графика...")
plt.figure(figsize=(15, 10))
colors = ['blue', 'green', 'orange', 'purple', 'cyan']

for idx, model_name in enumerate(models):
    model = load_model(model_name)
    if model is None:
        continue
    
    y_pred = model.predict(X)
    plt.scatter(y, y_pred, alpha=0.3, color=colors[idx], label=model_name)

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение предсказаний всех моделей')
plt.legend()
plt.grid(True)

# Отображение и сохранение
plt.show(block=False)
plt.pause(5)  # Увеличенная пауза для общего графика
plt.savefig("all_models_predictions.png")
plt.close()
logging.info("Общий график сохранен.")

# --------------------------------------------
# 3. График лучшей модели
# --------------------------------------------
logging.info("Построение графика лучшей модели...")
best_model = load_model(best_model_name)
if best_model is not None:
    y_pred = best_model.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5, color='red', label=f'{best_model_name} (Best)')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Предсказания лучшей модели: {best_model_name}')
    plt.legend()
    plt.grid(True)
    
    plt.show(block=False)
    plt.pause(3)
    plt.savefig("best_model_predictions.png")
    plt.close()
    logging.info("График лучшей модели сохранен.")
else:
    logging.error("Не удалось загрузить лучшую модель!")
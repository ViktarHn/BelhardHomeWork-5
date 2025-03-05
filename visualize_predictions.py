import matplotlib.pyplot as plt
import joblib
from utils import load_data, convert_sparse_to_dense
import logging

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

# Визуализация предсказаний для каждой модели отдельно
for model_name, model in models.items():
    logging.info(f"Предсказание для модели: {model_name}...")
    y_pred = model.predict(X_test)

    # Создание отдельного графика для каждой модели
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label=model_name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Идеальная линия')
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'Сравнение предсказаний и реальных значений для {model_name}')
    plt.legend()
    plt.grid(True)
    
    # Отображение графика в Jupyter Notebook
    plt.show()
    
    # Сохранение графика в файл (опционально)
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_predictions.png')
    plt.close()  # Закрытие графика для освобождения памяти
    logging.info(f"График для модели {model_name} сохранен.")

logging.info("Визуализация предсказаний завершена.")
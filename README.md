# Проект: Предсказание прибыли на основе данных Superstore

## Описание проекта
Цель проекта — построить модель регрессии для предсказания прибыли (Profit) на основе данных из датасета "Sample - Superstore.csv". В процессе работы проводится анализ данных, предобработка, обучение нескольких моделей машинного обучения, их сравнение и выбор лучшей модели.

## Используемые данные
- **Датасет**: `Sample - Superstore.csv`
- **Целевая переменная**: Прибыль (Profit)
- **Признаки**: Объем продаж (Sales), количество товаров (Quantity), скидка (Discount), категория товара (Category), подкатегория товара (Sub-Category), регион (Region), способ доставки (Ship Mode).

## Файлы проекта. Порядок запуска
1. **`eda.py`**  
   - **Задача**: Проведение анализа данных (EDA), очистка данных, удаление лишних признаков.  
   - **Результат**: Создает файл `cleaned_data.csv` с очищенными данными.  
   - **Запуск**:  
     ```bash
     python eda.py
     ```

2. **`feature_selection.py`**  
   - **Задача**: Выбор ключевых признаков, предобработка данных (масштабирование числовых признаков, кодирование категориальных).  
   - **Результат**: Создает файлы `preprocessed_data.joblib`, `preprocessor.joblib` и `feature_names.joblib`.  
   - **Запуск**:  
     ```bash
     python feature_selection.py
     ```

3. **`hyperparameter_tuning.py`**  
   - **Задача**: Подбор гиперпараметров для моделей RandomForest и CatBoost с использованием Optuna.  
   - **Результат**: Создает файлы `best_params_rf.joblib` и `best_params_catboost.joblib`.  
   - **Запуск**:  
     ```bash
     python hyperparameter_tuning.py
     ```

4. **`train_models.py`**  
   - **Задача**: Обучение нескольких моделей (Linear Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost) и их сравнение по метрикам (MAE, MSE, RMSE, R²).  
   - **Результат**: Сохраняет модели в файлы (например, `linear_regression_model.joblib`) и метрики в `model_results.joblib`. Лучшая модель сохраняется в `best_model.joblib`.  
   - **Запуск**:  
     ```bash
     python train_models.py
     ```

5. **`analyze_feature_importance.py`**  
   - **Задача**: Визуализация важности признаков для лучшей модели.  
   - **Результат**: Строит график важности признаков и сохраняет его в файл.  
   - **Запуск**:  
     ```bash
     python analyze_feature_importance.py
     ```

6. **`visualize_predictions.py`**  
   - **Задача**: Визуализация предсказаний всех моделей.  
   - **Результат**: Строит отдельные графики для каждой модели, общий график для всех моделей и отдельный график для лучшей модели. Графики отображаются на экране и сохраняются в файлы.  
   - **Запуск**:  
     ```bash
     python visualize_predictions.py
     ```

7. **`analyze_residuals.py`**  
   - **Задача**: Анализ остатков (разницы между реальными и предсказанными значениями) для лучшей модели.  
   - **Результат**: Строит график остатков и сохраняет его в файл.  
   - **Запуск**:  
     ```bash
     python analyze_residuals.py
     ```
## Результаты

**Графики**:

Отдельные графики для каждой модели (например, `linear_regression_predictions.png`).

Общий график для всех моделей (`all_models_predictions.png`).

График лучшей модели (`best_model_predictions.png`).

График важности признаков (`feature_importance.png`).

График остатков (`residuals.png`).

**Модели**:

Все обученные модели сохраняются в файлы (например, `linear_regression_model.joblib`).

Лучшая модель сохраняется в `best_model.joblib`.

**Метрики**:

Метрики всех моделей сохраняются в `model_results.joblib`.

## Зависимости

Для запуска проекта необходимо установить следующие библиотеки:

    ```bash
    pip install pandas scikit-learn matplotlib seaborn joblib optuna xgboost catboost
    ```
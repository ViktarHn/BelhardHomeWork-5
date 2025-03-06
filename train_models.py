from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import logging
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
data = joblib.load('preprocessed_data.joblib')
X, y = data['X'], data['y']

# Загрузка гиперпараметров
best_params_rf = joblib.load('best_params_rf.joblib')
best_params_catboost = joblib.load('best_params_catboost.joblib')

# Инициализация моделей
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(**best_params_rf, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(**best_params_catboost, verbose=0, random_state=42)
}

# Обучение и оценка
results = {}
for model_name, model in models.items():
    logging.info(f"Обучение модели: {model_name}...")
    
    # Кросс-валидация
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Метрики
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    results[model_name] = {
        "CV MAE": round(cv_mae, 4),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R²": round(r2, 4)
    }
    
    # Сохранение модели
    joblib.dump(model, f'{model_name.replace(" ", "_").lower()}_model.joblib')
    logging.info(f"Модель {model_name} сохранена.")

# Сохранение метрик
joblib.dump(results, 'model_results.joblib')

# Определение лучшей модели
best_model_name = max(results, key=lambda k: results[k]['R²'])
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.joblib')
logging.info(f"Лучшая модель: {best_model_name} (R²={results[best_model_name]['R²']})")
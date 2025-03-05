from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import logging
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
data_dict = joblib.load('preprocessed_data.joblib')
X_train = data_dict['X_train']
y_train = data_dict['y_train']

# Модели и их гиперпараметры для оптимизации
models = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42),
        "params": {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 50, 100]
        }
    },
    "CatBoost": {
        "model": CatBoostRegressor(random_state=42, verbose=0),
        "params": {
            'iterations': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
    }
}

# Оптимизация гиперпараметров с использованием RandomizedSearchCV
for model_name, config in models.items():
    logging.info(f"Оптимизация гиперпараметров для модели: {model_name}...")
    random_search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        n_iter=20,  # Количество итераций
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    
    # Сохранение лучшей модели
    best_model = random_search.best_estimator_
    joblib.dump(best_model, f'optimized_{model_name.replace(" ", "_").lower()}_model.joblib')
    logging.info(f"Лучшие параметры для {model_name}: {random_search.best_params_}")
    logging.info(f"Лучшая модель сохранена как 'optimized_{model_name.replace(' ', '_').lower()}_model.joblib'")
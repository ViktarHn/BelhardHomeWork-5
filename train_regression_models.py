from data_loader import load_and_preprocess_data
from sklearn.linear_model import (
    LinearRegression, Lasso, Ridge, ElasticNet, 
    BayesianRidge, HuberRegressor, OrthogonalMatchingPursuit, 
    PassiveAggressiveRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, 
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import joblib
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка предобработанных данных
data_dict = joblib.load('preprocessed_data.joblib')
X_train = data_dict['X_train']
X_test = data_dict['X_test']
y_train = data_dict['y_train']
y_test = data_dict['y_test']

# Инициализация моделей
models = {
    # Линейные модели
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet(),
    "Bayesian Ridge": BayesianRidge(),
    "Huber Regressor": HuberRegressor(),
    "Orthogonal Matching Pursuit": OrthogonalMatchingPursuit(),
    "Passive Aggressive Regressor": PassiveAggressiveRegressor(max_iter=1000, tol=1e-3),
    
    # Деревья и ансамбли
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    
    # Градиентный бустинг
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(random_state=42, verbose=0),
    
    # Прочие
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Dummy Regressor": DummyRegressor()
}

# Словарь для результатов
results = {}

# Преобразование в плотные массивы (если необходимо)
def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

# Обучение и оценка
for model_name, model in models.items():
    try:
        logging.info(f"Обучение модели: {model_name}...")
        X_train_dense = to_dense(X_train)
        X_test_dense = to_dense(X_test)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train_dense, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            "CV MAE": round(cv_mae, 4),
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R²": round(r2, 4)
        }
        
        # Сохранение модели
        joblib.dump(model, f'{model_name.replace(" ", "_").lower()}_model.joblib')
        logging.info(f"[+] {model_name} успешно обучена и сохранена.")
        
    except Exception as e:
        logging.error(f"[!] Ошибка в {model_name}: {str(e)}")

# Вывод результатов
logging.info("\nРезультаты оценки моделей:")
for model_name, metrics in results.items():
    logging.info(f"\n{model_name}:")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value}")
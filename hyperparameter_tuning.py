import optuna
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
import joblib
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка предобработанных данных
data = joblib.load('preprocessed_data.joblib')
X = data['X']
y = data['y']

# Функция для оптимизации RandomForest
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    score = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
    return score

# Функция для оптимизации CatBoost
def objective_catboost(trial):
    iterations = trial.suggest_int('iterations', 500, 1500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
    depth = trial.suggest_int('depth', 4, 8)
    model = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0, random_state=42)
    score = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
    return score

# Оптимизация для RandomForest
study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=20)

# Оптимизация для CatBoost
study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(objective_catboost, n_trials=20)

# Сохранение лучших параметров
best_params_rf = study_rf.best_params
best_params_catboost = study_catboost.best_params

joblib.dump(best_params_rf, 'best_params_rf.joblib')
joblib.dump(best_params_catboost, 'best_params_catboost.joblib')

logging.info(f"Лучшие параметры для RandomForest: {best_params_rf}")
logging.info(f"Лучшие параметры для CatBoost: {best_params_catboost}")
logging.info("Лучшие параметры сохранены в файлы 'best_params_rf.joblib' и 'best_params_catboost.joblib'.")
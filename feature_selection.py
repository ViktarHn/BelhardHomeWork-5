import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка очищенных данных
data = pd.read_csv('cleaned_data.csv')
logging.info("Очищенные данные успешно загружены.")

# Выбор ключевых признаков
selected_features = ['Sales', 'Quantity', 'Discount', 'Category', 'Sub-Category', 'Region', 'Ship Mode']
X = data[selected_features]
y = data['Profit']

# Категориальные и числовые признаки
categorical_features = ['Category', 'Sub-Category', 'Region', 'Ship Mode']
numeric_features = ['Sales', 'Quantity', 'Discount']

# Создание пайплайна для предобработки
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Применение предобработки
X_preprocessed = preprocessor.fit_transform(X)
logging.info("Данные успешно предобработаны.")

# Сохранение предобработанных данных
joblib.dump({'X': X_preprocessed, 'y': y}, 'preprocessed_data.joblib')
logging.info("Предобработанные данные сохранены в файл 'preprocessed_data.joblib'.")

# Сохранение ColumnTransformer для получения имен признаков
joblib.dump(preprocessor, 'preprocessor.joblib')
logging.info("ColumnTransformer сохранен в файл 'preprocessor.joblib'.")

# Получение имен признаков
feature_names = preprocessor.get_feature_names_out()

# Сохранение имен признаков в отдельный файл
joblib.dump(feature_names, 'feature_names.joblib')
logging.info("Имена признаков сохранены в файл 'feature_names.joblib'.")
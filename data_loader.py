import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import numpy as np
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    logging.info("Загрузка данных...")
    data = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')

    logging.info("Удаление ненужных столбцов...")
    data = data.drop(['Row ID', 'Order ID', 'Customer ID', 'Customer Name'], axis=1)

    logging.info("Проверка на пропуски...")
    print("Пропущенные значения до обработки:")
    print(data.isnull().sum())

    def convert_dates(X):
        date_columns = ['Order Date', 'Ship Date']
        for col in date_columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
            X[col] = (X[col] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
        return X

    logging.info("Разделение на X и y...")
    X = data.drop('Profit', axis=1)
    y = data['Profit']

    logging.info("Преобразование дат...")
    X = convert_dates(X)

    numeric_imputer = SimpleImputer(strategy='median')
    date_imputer = SimpleImputer(strategy='most_frequent')

    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.drop(['Order Date', 'Ship Date'])
    date_features = ['Order Date', 'Ship Date']
    categorical_features = X.select_dtypes(include=['object']).columns

    logging.info("Создание пайплайна предобработки...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', numeric_imputer),
                ('scaler', StandardScaler())
            ]), numeric_features),
            
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
            
            ('date', date_imputer, date_features)
        ])

    logging.info("Разделение данных на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Применение препроцессора...")
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    logging.info("Сохранение препроцессора и данных...")
    joblib.dump(preprocessor, 'preprocessor.joblib')
    joblib.dump({'X_train': X_train, 'X_test': X_test, 
                'y_train': y_train, 'y_test': y_test}, 'preprocessed_data.joblib')

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("\nПосле обработки:")
    print(f"Размерности данных: Train {X_train.shape}, Test {X_test.shape}")
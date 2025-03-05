import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='feature_engineering.log'  # Логирование в файл
)

def create_new_features(data):
    """
    Создание новых признаков.
    """
    logging.info("Создание новых признаков: взаимодействие между Sales и Discount...")
    data['Sales_Discount'] = data['Sales'] * data['Discount']
    
    logging.info("Создание полиномиальных признаков для Sales и Quantity...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(data[['Sales', 'Quantity']])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Sales', 'Quantity']))
    
    logging.info("Создание временных признаков из Order Date...")
    data['Order_Date'] = pd.to_datetime(data['Order Date'])
    data['Order_Day'] = data['Order_Date'].dt.day
    data['Order_Month'] = data['Order_Date'].dt.month
    
    logging.info("Объединение всех признаков...")
    data = pd.concat([data, poly_df], axis=1)
    return data

def preprocess_data_with_new_features(data):
    """
    Предобработка данных с новыми признаками.
    """
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

if __name__ == "__main__":
    try:
        logging.info("Загрузка данных...")
        data = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')
        
        logging.info("Создание новых признаков...")
        data = create_new_features(data)
        
        logging.info("Предобработка данных с новыми признаками...")
        preprocessor = preprocess_data_with_new_features(data)
        
        logging.info("Сохранение предобработанных данных...")
        joblib.dump(preprocessor, 'preprocessor_with_new_features.joblib')
        data.to_csv('data_with_new_features.csv', index=False)
        logging.info("Предобработка завершена успешно.")
    except Exception as e:
        logging.error(f"Ошибка: {str(e)}")
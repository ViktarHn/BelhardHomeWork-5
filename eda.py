import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Проверка, существует ли уже файл cleaned_data.csv
if os.path.exists('cleaned_data.csv'):
    logging.info("Файл 'cleaned_data.csv' уже существует. Пропуск EDA.")
else:
    # Загрузка данных
    data = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')
    logging.info("Данные успешно загружены.")

    # Проверка на пропуски
    logging.info("Проверка на пропущенные значения:")
    logging.info(data.isnull().sum())

    # Корреляционная матрица для числовых признаков
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Корреляционная матрица')
    plt.show()

    # Анализ дублирующих признаков
    logging.info("Анализ дублирующих признаков:")
    logging.info(data[['City', 'Postal Code']].head(10))

    # Удаление лишних признаков
    data = data.drop(['Postal Code', 'Row ID', 'Order ID', 'Customer ID', 'Customer Name'], axis=1)
    logging.info("Лишние признаки удалены.")

    # Сохранение очищенных данных
    data.to_csv('cleaned_data.csv', index=False)
    logging.info("Очищенные данные сохранены в файл 'cleaned_data.csv'.")
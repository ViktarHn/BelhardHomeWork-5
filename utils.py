import joblib

def load_data(file_path='preprocessed_data.joblib'):
    """
    Загружает предобработанные данные из файла.
    Возвращает словарь с X_train, X_test, y_train, y_test.
    """
    data_dict = joblib.load(file_path)
    return data_dict

def convert_sparse_to_dense(X_train, X_test):
    """
    Преобразует разреженные матрицы в плотные массивы, если это необходимо.
    """
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    return X_train, X_test
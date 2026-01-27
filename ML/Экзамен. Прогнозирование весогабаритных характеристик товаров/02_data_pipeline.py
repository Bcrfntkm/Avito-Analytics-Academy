"""
Пайплайн загрузки и предобработки данных для прогнозирования весогабаритных характеристик товаров.

Этот модуль является основой для всего ML-пайплайна и содержит классы и функции для:
- Загрузки данных из parquet файлов с обработкой ошибок
- Разделения на train/validation выборки с поддержкой stratification
- Предобработки признаков (категориальных, числовых, текстовых, временных)
- Трансформации целевых переменных (log1p для стабилизации распределения)
- Вычисления метрики Macro Log-MAE (основная метрика соревнования)
- Создания submission файлов в требуемом формате

Основные компоненты:
    - DataLoader: Класс для загрузки и разделения данных
    - FeaturePreprocessor: Класс для предобработки всех типов признаков
    - Вспомогательные функции для работы с метриками и сохранения результатов

Примечания:
    - Все трансформации применяются последовательно и могут быть сохранены
    - Препроцессор поддерживает fit/transform паттерн для корректной работы с train/test
    - Log1p трансформация используется для нормализации распределения целевых переменных
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    Класс для загрузки данных и разделения на train/validation выборки.
    
    Attributes:
        data_dir (Path): Директория с данными
        train_df (pd.DataFrame): Обучающий датасет
        test_df (pd.DataFrame): Тестовый датасет
        train_split (pd.DataFrame): Train часть после разделения
        val_split (pd.DataFrame): Validation часть после разделения
    """
    
    def __init__(self, data_dir: str = '.'):
        """
        Инициализация загрузчика данных.
        
        Args:
            data_dir: Путь к директории с данными
        """
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        self.train_split = None
        self.val_split = None
        
    def load_data(self, train_file: str = 'train.parquet', 
                  test_file: str = 'test.parquet') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загрузка train и test данных из parquet файлов.
        
        Args:
            train_file: Имя файла с обучающими данными
            test_file: Имя файла с тестовыми данными
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train и test датафреймы
            
        Raises:
            FileNotFoundError: Если файлы не найдены
            Exception: При ошибках чтения файлов
        """
        train_path = self.data_dir / train_file
        test_path = self.data_dir / test_file
        
        if not train_path.exists():
            raise FileNotFoundError(f"Train файл не найден: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test файл не найден: {test_path}")
            
        try:
            print(f"Загрузка train данных из {train_path}...")
            self.train_df = pd.read_parquet(train_path)
            print(f"Train данные загружены: {self.train_df.shape}")
            
            print(f"Загрузка test данных из {test_path}...")
            self.test_df = pd.read_parquet(test_path)
            print(f"Test данные загружены: {self.test_df.shape}")
            
            return self.train_df, self.test_df
            
        except Exception as e:
            raise Exception(f"Ошибка при загрузке данных: {str(e)}")
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42,
                   stratify_column: str = 'category_name') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Разделение train данных на train/validation с stratification.
        
        Args:
            test_size: Доля validation выборки (по умолчанию 0.2)
            random_state: Seed для воспроизводимости
            stratify_column: Колонка для stratification
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train и validation датафреймы
            
        Raises:
            ValueError: Если train данные не загружены
        """
        if self.train_df is None:
            raise ValueError("Train данные не загружены. Сначала вызовите load_data()")
        
        # Проверка наличия колонки для stratification
        if stratify_column not in self.train_df.columns:
            print(f"Предупреждение: колонка {stratify_column} не найдена. "
                  f"Разделение без stratification.")
            stratify = None
        else:
            stratify = self.train_df[stratify_column]
        
        try:
            self.train_split, self.val_split = train_test_split(
                self.train_df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            print(f"Данные разделены:")
            print(f"  Train: {self.train_split.shape}")
            print(f"  Validation: {self.val_split.shape}")
            
            return self.train_split, self.val_split
            
        except Exception as e:
            raise Exception(f"Ошибка при разделении данных: {str(e)}")
    
    def get_batches(self, df: pd.DataFrame, batch_size: int = 1000) -> List[pd.DataFrame]:
        """
        Генерация батчей данных для обработки больших датасетов.
        
        Разбивает датафрейм на батчи фиксированного размера для последовательной обработки.
        Полезно при работе с большими объемами данных для экономии памяти.
        
        Args:
            df: Датафрейм для разбиения на батчи
            batch_size: Размер батча (количество строк в каждом батче)
            
        Returns:
            List[pd.DataFrame]: Список батчей (копии датафреймов)
            
        Example:
            >>> loader = DataLoader()
            >>> batches = loader.get_batches(train_df, batch_size=500)
            >>> for batch in batches:
            ...     process_batch(batch)
        """
        # Вычисляем количество батчей с учетом остатка
        n_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
        batches = []
        
        # Создаем батчи с копированием данных для безопасности
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batches.append(df.iloc[start_idx:end_idx].copy())
        
        print(f"Создано {len(batches)} батчей по {batch_size} записей")
        return batches


class FeaturePreprocessor:
    """
    Класс для предобработки признаков и целевых переменных.
    
    Выполняет:
    - Обработку пропусков
    - Log1p трансформацию таргетов
    - Кодирование категориальных признаков
    - Нормализацию числовых признаков
    - Извлечение временных признаков
    - Извлечение текстовых признаков
    
    Attributes:
        target_columns (List[str]): Список целевых переменных
        categorical_columns (List[str]): Список категориальных признаков
        numerical_columns (List[str]): Список числовых признаков
        label_encoders (Dict): Словарь с LabelEncoder'ами для категорий
        scaler (StandardScaler): Скейлер для числовых признаков
        is_fitted (bool): Флаг обученности препроцессора
    """
    
    def __init__(self, target_columns: List[str] = None):
        """
        Инициализация препроцессора.
        
        Args:
            target_columns: Список целевых переменных
        """
        self.target_columns = target_columns or ['item_weight', 'item_length', 
                                                   'item_width', 'item_height']
        self.categorical_columns = ['category_name', 'subcategory_name', 
                                     'microcat_name', 'item_condition']
        self.numerical_columns = ['item_price']
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка пропущенных значений в датафрейме.
        
        Стратегия обработки:
        - item_condition: заполняется значением 'unknown' (новая категория)
        - Остальные колонки: выводится предупреждение о наличии пропусков
        
        Args:
            df: Датафрейм для обработки
            
        Returns:
            pd.DataFrame: Датафрейм с обработанными пропусками
            
        Note:
            Метод создает копию датафрейма, не изменяя оригинал
        """
        df = df.copy()
        
        # Заполнение пропусков в item_condition специальным значением
        # 'unknown' будет закодирован как отдельная категория
        if 'item_condition' in df.columns:
            df['item_condition'] = df['item_condition'].fillna('unknown')
        
        # Проверка и вывод информации о других пропусках
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Обнаружены пропуски в колонках:")
            print(missing_counts[missing_counts > 0])
        
        return df
    
    def transform_targets(self, df: pd.DataFrame, inverse: bool = False) -> pd.DataFrame:
        """
        Log1p трансформация целевых переменных для стабилизации распределения.
        
        Log1p (log(1+x)) используется вместо обычного log для:
        - Обработки нулевых значений (log(1+0) = 0)
        - Уменьшения влияния выбросов
        - Приближения распределения к нормальному
        
        Args:
            df: Датафрейм с целевыми переменными
            inverse: Если True, выполняет обратную трансформацию (expm1)
            
        Returns:
            pd.DataFrame: Датафрейм с трансформированными таргетами
            
        Note:
            Обратная трансформация необходима для получения предсказаний
            в исходной шкале измерений
        """
        df = df.copy()
        
        for col in self.target_columns:
            if col in df.columns:
                if inverse:
                    # Обратная трансформация: expm1 = exp(x) - 1
                    df[col] = np.expm1(df[col])
                else:
                    # Прямая трансформация: log1p = log(1 + x)
                    df[col] = np.log1p(df[col])
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Кодирование категориальных признаков с помощью LabelEncoder.
        
        LabelEncoder преобразует категории в числа (0, 1, 2, ...).
        Это необходимо для работы с алгоритмами машинного обучения.
        
        Args:
            df: Датафрейм для кодирования
            fit: Если True, обучает энкодеры на данных (для train)
            
        Returns:
            pd.DataFrame: Датафрейм с закодированными категориями
            
        Raises:
            ValueError: Если энкодер не обучен при fit=False
            
        Note:
            Неизвестные категории в test данных заменяются на 'unknown'
        """
        df = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
                
            if fit:
                # Обучение энкодера на train данных
                self.label_encoders[col] = LabelEncoder()
                # Преобразование в строки для корректной работы с любыми типами
                df[col] = df[col].astype(str)
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                # Применение обученного энкодера к новым данным
                if col not in self.label_encoders:
                    raise ValueError(f"Энкодер для {col} не обучен. Сначала вызовите fit()")
                
                df[col] = df[col].astype(str)
                # Обработка неизвестных категорий (которых не было в train)
                known_classes = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known_classes else 'unknown'
                )
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def normalize_numerical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Нормализация числовых признаков с помощью StandardScaler.
        
        Args:
            df: Датафрейм для нормализации
            fit: Если True, обучает скейлер на данных
            
        Returns:
            pd.DataFrame: Датафрейм с нормализованными признаками
        """
        df = df.copy()
        
        if not self.numerical_columns:
            return df
        
        # Проверка наличия колонок
        available_cols = [col for col in self.numerical_columns if col in df.columns]
        
        if not available_cols:
            return df
        
        if fit:
            df[available_cols] = self.scaler.fit_transform(df[available_cols])
        else:
            df[available_cols] = self.scaler.transform(df[available_cols])
        
        return df
    
    def extract_date_features(self, df: pd.DataFrame,
                             date_column: str = 'order_date') -> pd.DataFrame:
        """
        Извлечение временных признаков из даты заказа.
        
        Создает дополнительные признаки, которые могут влиять на характеристики товаров:
        - Месяц (сезонность)
        - День недели (паттерны покупок)
        - День месяца
        - Квартал
        - Год
        
        Args:
            df: Датафрейм с датой
            date_column: Название колонки с датой
            
        Returns:
            pd.DataFrame: Датафрейм с дополнительными временными признаками
            
        Note:
            Автоматически преобразует колонку в datetime если необходимо
        """
        df = df.copy()
        
        if date_column not in df.columns:
            print(f"Предупреждение: колонка {date_column} не найдена")
            return df
        
        # Преобразование в datetime если колонка не в формате datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Извлечение различных временных компонент
        df['order_month'] = df[date_column].dt.month  # 1-12
        df['order_day_of_week'] = df[date_column].dt.dayofweek  # 0-6 (понедельник-воскресенье)
        df['order_day_of_month'] = df[date_column].dt.day  # 1-31
        df['order_quarter'] = df[date_column].dt.quarter  # 1-4
        df['order_year'] = df[date_column].dt.year
        
        return df
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение базовых текстовых признаков из title и description.
        
        Args:
            df: Датафрейм с текстовыми полями
            
        Returns:
            pd.DataFrame: Датафрейм с дополнительными текстовыми признаками
        """
        df = df.copy()
        
        # Признаки из title
        if 'title' in df.columns:
            df['title_length'] = df['title'].fillna('').astype(str).str.len()
            df['title_word_count'] = df['title'].fillna('').astype(str).str.split().str.len()
            df['title_upper_ratio'] = df['title'].fillna('').astype(str).apply(
                lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            )
        
        # Признаки из description
        if 'description' in df.columns:
            df['description_length'] = df['description'].fillna('').astype(str).str.len()
            df['description_word_count'] = df['description'].fillna('').astype(str).str.split().str.len()
            df['description_has_content'] = (df['description'].fillna('').astype(str).str.len() > 0).astype(int)
        
        return df
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePreprocessor':
        """
        Обучение препроцессора на обучающих данных.
        
        Args:
            df: Обучающий датафрейм
            
        Returns:
            FeaturePreprocessor: Обученный препроцессор
        """
        print("Обучение препроцессора...")
        
        # Обработка пропусков
        df = self.handle_missing_values(df)
        
        # Извлечение признаков
        df = self.extract_date_features(df)
        df = self.extract_text_features(df)
        
        # Обучение энкодеров и скейлера
        df = self.encode_categorical(df, fit=True)
        df = self.normalize_numerical(df, fit=True)
        
        self.is_fitted = True
        print("Препроцессор обучен")
        
        return self
    
    def transform(self, df: pd.DataFrame, transform_targets: bool = False) -> pd.DataFrame:
        """
        Применение предобработки к данным.
        
        Args:
            df: Датафрейм для трансформации
            transform_targets: Если True, применяет log1p к таргетам
            
        Returns:
            pd.DataFrame: Трансформированный датафрейм
        """
        if not self.is_fitted:
            raise ValueError("Препроцессор не обучен. Сначала вызовите fit()")
        
        df = df.copy()
        
        # Обработка пропусков
        df = self.handle_missing_values(df)
        
        # Извлечение признаков
        df = self.extract_date_features(df)
        df = self.extract_text_features(df)
        
        # Применение энкодеров и скейлера
        df = self.encode_categorical(df, fit=False)
        df = self.normalize_numerical(df, fit=False)
        
        # Трансформация таргетов если нужно
        if transform_targets:
            df = self.transform_targets(df, inverse=False)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, transform_targets: bool = False) -> pd.DataFrame:
        """
        Обучение и применение предобработки.
        
        Args:
            df: Датафрейм для обучения и трансформации
            transform_targets: Если True, применяет log1p к таргетам
            
        Returns:
            pd.DataFrame: Трансформированный датафрейм
        """
        self.fit(df)
        return self.transform(df, transform_targets=transform_targets)


def calculate_macro_log_mae(y_true: np.ndarray, y_pred: np.ndarray,
                            target_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Вычисление метрики Macro Log-MAE (основная метрика соревнования).
    
    Метрика вычисляется как среднее MAE по log1p-трансформированным таргетам:
    1. Применяется log1p к истинным и предсказанным значениям
    2. Вычисляется MAE для каждого таргета
    3. Берется среднее по всем таргетам (macro-averaging)
    
    Формула: Macro Log-MAE = mean([MAE(log1p(y_true_i), log1p(y_pred_i)) for i in targets])
    
    Args:
        y_true: Истинные значения (shape: [n_samples, n_targets])
        y_pred: Предсказанные значения (shape: [n_samples, n_targets])
        target_names: Названия целевых переменных (опционально)
        
    Returns:
        Dict[str, float]: Словарь с метриками:
            - '{target}_log_mae': MAE для каждого таргета
            - 'macro_log_mae': общий score (среднее по всем таргетам)
            
    Raises:
        ValueError: Если размерности y_true и y_pred не совпадают
        
    Example:
        >>> y_true = np.array([[100, 50], [200, 60]])
        >>> y_pred = np.array([[110, 48], [190, 62]])
        >>> scores = calculate_macro_log_mae(y_true, y_pred, ['weight', 'height'])
        >>> print(scores['macro_log_mae'])
    """
    if target_names is None:
        target_names = ['item_weight', 'item_length', 'item_width', 'item_height']
    
    # Проверка размерностей входных данных
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Размерности не совпадают: y_true {y_true.shape}, y_pred {y_pred.shape}")
    
    # Применение log1p трансформации к обоим массивам
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)
    
    # Вычисление MAE для каждого таргета отдельно
    scores = {}
    mae_values = []
    
    for i, target_name in enumerate(target_names):
        # MAE = mean(|y_true - y_pred|)
        mae = np.mean(np.abs(y_true_log[:, i] - y_pred_log[:, i]))
        scores[f'{target_name}_log_mae'] = mae
        mae_values.append(mae)
    
    # Общий Macro Log-MAE (среднее по всем таргетам)
    scores['macro_log_mae'] = np.mean(mae_values)
    
    return scores


def save_preprocessor(preprocessor: FeaturePreprocessor,
                     filepath: str = 'preprocessor.pkl') -> None:
    """
    Сохранение обученного препроцессора в файл для последующего использования.
    
    Сохраняет все обученные трансформеры (LabelEncoder, StandardScaler) и параметры
    препроцессора в pickle файл. Это позволяет применять те же трансформации
    к новым данным (например, к test датасету).
    
    Args:
        preprocessor: Обученный препроцессор (должен быть fitted)
        filepath: Путь для сохранения файла (по умолчанию 'preprocessor.pkl')
        
    Raises:
        ValueError: Если препроцессор не обучен
        Exception: При ошибках записи файла
        
    Example:
        >>> preprocessor = FeaturePreprocessor()
        >>> preprocessor.fit(train_df)
        >>> save_preprocessor(preprocessor, 'models/preprocessor.pkl')
    """
    if not preprocessor.is_fitted:
        raise ValueError("Препроцессор не обучен. Нечего сохранять.")
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"Препроцессор сохранен в {filepath}")
    except Exception as e:
        raise Exception(f"Ошибка при сохранении препроцессора: {str(e)}")


def load_preprocessor(filepath: str = 'preprocessor.pkl') -> FeaturePreprocessor:
    """
    Загрузка обученного препроцессора из файла.
    
    Восстанавливает сохраненный препроцессор со всеми обученными трансформерами.
    Используется для применения тех же трансформаций к test данным или при
    инференсе модели.
    
    Args:
        filepath: Путь к файлу с препроцессором (по умолчанию 'preprocessor.pkl')
        
    Returns:
        FeaturePreprocessor: Загруженный и готовый к использованию препроцессор
        
    Raises:
        FileNotFoundError: Если файл не найден
        Exception: При ошибках чтения файла
        
    Example:
        >>> preprocessor = load_preprocessor('models/preprocessor.pkl')
        >>> test_processed = preprocessor.transform(test_df)
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Файл препроцессора не найден: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Препроцессор загружен из {filepath}")
        return preprocessor
    except Exception as e:
        raise Exception(f"Ошибка при загрузке препроцессора: {str(e)}")


def create_submission(test_ids: np.ndarray, predictions: np.ndarray,
                     target_columns: Optional[List[str]] = None,
                     filepath: str = 'submission.csv') -> pd.DataFrame:
    """
    Создание submission файла для отправки на платформу соревнования.
    
    Формирует CSV файл в требуемом формате:
    - Первая колонка: id (идентификаторы тестовых объектов)
    - Остальные колонки: предсказания для каждой целевой переменной
    
    Args:
        test_ids: ID тестовых объектов (массив длины n_samples)
        predictions: Предсказания модели (shape: [n_samples, n_targets])
        target_columns: Названия целевых переменных (опционально)
        filepath: Путь для сохранения submission файла
        
    Returns:
        pd.DataFrame: Submission датафрейм (для проверки перед отправкой)
        
    Raises:
        ValueError: Если размерности не совпадают
        
    Example:
        >>> predictions = model.predict(X_test)
        >>> submission = create_submission(
        ...     test_ids=test_df['item_id'].values,
        ...     predictions=predictions,
        ...     target_columns=['weight', 'height', 'width', 'length'],
        ...     filepath='submission.csv'
        ... )
    """
    if target_columns is None:
        target_columns = ['item_weight', 'item_length', 'item_width', 'item_height']
    
    # Проверка размерностей для предотвращения ошибок
    if len(test_ids) != predictions.shape[0]:
        raise ValueError(f"Количество ID ({len(test_ids)}) не совпадает с "
                        f"количеством предсказаний ({predictions.shape[0]})")
    
    if predictions.shape[1] != len(target_columns):
        raise ValueError(f"Количество таргетов в предсказаниях ({predictions.shape[1]}) "
                        f"не совпадает с количеством названий ({len(target_columns)})")
    
    # Создание датафрейма с ID
    submission = pd.DataFrame({
        'id': test_ids
    })
    
    # Добавление предсказаний для каждого таргета
    for i, col in enumerate(target_columns):
        submission[col] = predictions[:, i]
    
    # Сохранение в CSV без индекса
    submission.to_csv(filepath, index=False)
    print(f"Submission файл сохранен: {filepath}")
    print(f"Размер: {submission.shape}")
    
    return submission


# Пример использования
if __name__ == "__main__":
    # Инициализация загрузчика
    loader = DataLoader(data_dir='.')
    
    # Загрузка данных
    try:
        train_df, test_df = loader.load_data()
        
        # Разделение на train/val
        train_split, val_split = loader.split_data(test_size=0.2, random_state=42)
        
        # Инициализация препроцессора
        preprocessor = FeaturePreprocessor()
        
        # Обучение на train данных
        train_processed = preprocessor.fit_transform(train_split, transform_targets=True)
        
        # Применение к validation данных
        val_processed = preprocessor.transform(val_split, transform_targets=True)
        
        # Сохранение препроцессора
        save_preprocessor(preprocessor, 'preprocessor.pkl')
        
        print("\nПример обработанных данных:")
        print(train_processed.head())
        
        # Пример вычисления метрики
        target_cols = ['item_weight', 'item_length', 'item_width', 'item_height']
        if all(col in val_split.columns for col in target_cols):
            # Создание dummy предсказаний для примера
            y_true = val_split[target_cols].values
            y_pred = y_true * 1.1  # Dummy предсказания
            
            scores = calculate_macro_log_mae(y_true, y_pred, target_cols)
            print("\nПример метрик:")
            for metric, value in scores.items():
                print(f"  {metric}: {value:.6f}")
        
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что файлы train.parquet и test.parquet находятся в текущей директории")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
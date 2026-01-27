"""
Модель с текстовыми признаками для прогнозирования весогабаритных характеристик товаров.

Этот модуль реализует улучшенную модель, которая использует текстовую информацию
из названий и описаний товаров для более точного предсказания весогабаритных характеристик.

Ключевые улучшения относительно baseline:
    - TF-IDF векторизация для title и description (300 признаков)
    - Извлечение размерных характеристик из текста с помощью regex:
        * Вес (кг, г)
        * Размеры (см, м, мм)
        * Высота, ширина, длина
    - Подсчет упоминаний размерных ключевых слов
    - Увеличенное количество деревьев в LightGBM (200 вместо 100)
    - Оптимизированные гиперпараметры (learning_rate=0.05, max_depth=8)

Архитектура:
    1. TextFeatureExtractor: Извлечение TF-IDF и размерных признаков
    2. ImprovedModel: LightGBM модель с текстовыми признаками
    3. Объединение табличных, TF-IDF и размерных признаков

Метрика: Macro Log-MAE
Ожидаемое улучшение: ~2% относительно baseline (0.443 -> 0.434)
"""

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import warnings

# Импорт из пайплайна
import sys
import importlib.util

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    spec = importlib.util.spec_from_file_location(
        "data_pipeline_02",
        current_dir / "02_data_pipeline.py"
    )
    data_pipeline_02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_pipeline_02)
    
    DataLoader = data_pipeline_02.DataLoader
    FeaturePreprocessor = data_pipeline_02.FeaturePreprocessor
    calculate_macro_log_mae = data_pipeline_02.calculate_macro_log_mae
    save_preprocessor = data_pipeline_02.save_preprocessor
    create_submission = data_pipeline_02.create_submission
except Exception as e:
    print(f"Ошибка импорта: {e}")
    raise

warnings.filterwarnings('ignore')


class TextFeatureExtractor:
    """
    Класс для извлечения текстовых признаков из title и description товаров.
    
    Реализует два типа извлечения признаков:
    1. TF-IDF векторизация: преобразует текст в числовые векторы
    2. Размерные характеристики: извлекает упоминания размеров и веса
    
    Выполняемые операции:
        - TF-IDF векторизация для title (100 признаков) и description (200 признаков)
        - Извлечение размерных характеристик с помощью regex паттернов:
            * Вес: "5 кг", "500 г"
            * Размеры: "10 см", "1.5 м", "100 мм"
            * Специфичные размеры: "высота: 20 см", "ширина: 10 см"
        - Подсчет упоминаний размерных ключевых слов
        - Подсчет общего количества чисел в тексте
    
    Attributes:
        title_vectorizer (TfidfVectorizer): Векторизатор для заголовков (n-граммы 1-2)
        desc_vectorizer (TfidfVectorizer): Векторизатор для описаний (n-граммы 1-2)
        size_patterns (Dict[str, str]): Regex паттерны для извлечения размеров
        size_keywords (List[str]): Ключевые слова, связанные с размерами
        is_fitted (bool): Флаг обученности экстрактора
        
    Example:
        >>> extractor = TextFeatureExtractor(title_max_features=100, desc_max_features=200)
        >>> extractor.fit(train_df)
        >>> tfidf_features, size_features = extractor.transform(test_df)
    """
    
    def __init__(self, title_max_features: int = 100, 
                 desc_max_features: int = 200,
                 ngram_range: Tuple[int, int] = (1, 2)):
        """
        Инициализация экстрактора текстовых признаков.
        
        Args:
            title_max_features: Максимальное количество признаков для title
            desc_max_features: Максимальное количество признаков для description
            ngram_range: Диапазон n-грамм для TF-IDF
        """
        self.title_vectorizer = TfidfVectorizer(
            max_features=title_max_features,
            ngram_range=ngram_range,
            lowercase=True,
            min_df=2,
            max_df=0.95,
            strip_accents='unicode'
        )
        
        self.desc_vectorizer = TfidfVectorizer(
            max_features=desc_max_features,
            ngram_range=ngram_range,
            lowercase=True,
            min_df=2,
            max_df=0.95,
            strip_accents='unicode'
        )
        
        self.is_fitted = False
        
        # Паттерны для извлечения размерных характеристик
        self.size_patterns = {
            'weight_kg': r'(\d+(?:[.,]\d+)?)\s*кг',
            'weight_g': r'(\d+(?:[.,]\d+)?)\s*г(?![а-я])',
            'height_cm': r'(?:высот[аы]|h)\s*[:-]?\s*(\d+(?:[.,]\d+)?)\s*см',
            'width_cm': r'(?:ширин[аы]|w)\s*[:-]?\s*(\d+(?:[.,]\d+)?)\s*см',
            'length_cm': r'(?:длин[аы]|l)\s*[:-]?\s*(\d+(?:[.,]\d+)?)\s*см',
            'size_cm': r'(\d+(?:[.,]\d+)?)\s*см',
            'size_m': r'(\d+(?:[.,]\d+)?)\s*м(?![а-я])',
            'size_mm': r'(\d+(?:[.,]\d+)?)\s*мм',
        }
        
        # Ключевые слова, связанные с размерами
        self.size_keywords = [
            'вес', 'весит', 'кг', 'грамм', 'г',
            'высота', 'ширина', 'длина', 'размер',
            'см', 'м', 'мм', 'габарит',
            'компактн', 'больш', 'малень', 'средн'
        ]
    
    def _extract_size_mentions(self, text: str) -> Dict[str, float]:
        """
        Извлечение размерных характеристик из текста с помощью regex.
        
        Ищет в тексте упоминания размеров и веса в различных форматах:
        - "5 кг", "500 г" (вес)
        - "высота: 20 см", "ширина 10 см" (специфичные размеры)
        - "15 см", "1.5 м" (общие размеры)
        
        Для каждого паттерна извлекает:
        - Первое найденное значение
        - Количество упоминаний
        
        Args:
            text: Текст для анализа (title + description)
            
        Returns:
            Dict[str, float]: Словарь с извлеченными признаками:
                - text_{pattern}_first: первое найденное значение
                - text_{pattern}_count: количество упоминаний
                - text_keyword_{keyword}: количество упоминаний ключевого слова
                - text_numbers_count: общее количество чисел в тексте
                
        Example:
            >>> extractor._extract_size_mentions("Вес 5 кг, высота 20 см")
            {'text_weight_kg_first': 5.0, 'text_weight_kg_count': 1, ...}
        """
        if pd.isna(text) or not isinstance(text, str):
            text = ''
        
        text_lower = text.lower()
        features = {}
        
        # Извлечение чисел с единицами измерения
        for pattern_name, pattern in self.size_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                # Берем первое найденное значение
                try:
                    value = float(matches[0].replace(',', '.'))
                    features[f'text_{pattern_name}_first'] = value
                    features[f'text_{pattern_name}_count'] = len(matches)
                except:
                    features[f'text_{pattern_name}_first'] = 0
                    features[f'text_{pattern_name}_count'] = 0
            else:
                features[f'text_{pattern_name}_first'] = 0
                features[f'text_{pattern_name}_count'] = 0
        
        # Подсчет упоминаний ключевых слов
        for keyword in self.size_keywords:
            features[f'text_keyword_{keyword}'] = text_lower.count(keyword)
        
        # Общее количество чисел в тексте
        numbers = re.findall(r'\d+(?:[.,]\d+)?', text)
        features['text_numbers_count'] = len(numbers)
        
        return features
    
    def _process_text_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Обработка текстовой колонки - заполнение пропусков.
        
        Args:
            df: Датафрейм
            column: Название колонки
            
        Returns:
            pd.DataFrame: Обработанный датафрейм
        """
        df = df.copy()
        if column in df.columns:
            df[column] = df[column].fillna('').astype(str)
        else:
            df[column] = ''
        return df
    
    def fit(self, df: pd.DataFrame) -> 'TextFeatureExtractor':
        """
        Обучение экстрактора на обучающих данных.
        
        Args:
            df: Обучающий датафрейм с колонками title и description
            
        Returns:
            TextFeatureExtractor: Обученный экстрактор
        """
        print("Обучение TextFeatureExtractor...")
        
        # Обработка текстовых колонок
        df = self._process_text_column(df, 'title')
        df = self._process_text_column(df, 'description')
        
        # Обучение векторизаторов
        print("  - Обучение TF-IDF для title...")
        self.title_vectorizer.fit(df['title'])
        
        print("  - Обучение TF-IDF для description...")
        self.desc_vectorizer.fit(df['description'])
        
        self.is_fitted = True
        print("TextFeatureExtractor обучен")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        """
        Применение извлечения признаков к данным.
        
        Args:
            df: Датафрейм для трансформации
            
        Returns:
            Tuple[csr_matrix, pd.DataFrame]: 
                - Sparse матрица с TF-IDF признаками
                - Датафрейм с дополнительными текстовыми признаками
        """
        if not self.is_fitted:
            raise ValueError("Экстрактор не обучен. Сначала вызовите fit()")
        
        print("Извлечение текстовых признаков...")
        
        # Обработка текстовых колонок
        df = self._process_text_column(df, 'title')
        df = self._process_text_column(df, 'description')
        
        # TF-IDF векторизация
        print("  - TF-IDF для title...")
        title_tfidf = self.title_vectorizer.transform(df['title'])
        
        print("  - TF-IDF для description...")
        desc_tfidf = self.desc_vectorizer.transform(df['description'])
        
        # Объединение TF-IDF матриц
        tfidf_features = hstack([title_tfidf, desc_tfidf])
        
        # Извлечение размерных характеристик
        print("  - Извлечение размерных характеристик...")
        size_features_list = []
        
        for idx, row in df.iterrows():
            # Объединяем title и description для поиска размеров
            combined_text = f"{row['title']} {row['description']}"
            size_features = self._extract_size_mentions(combined_text)
            size_features_list.append(size_features)
        
        size_features_df = pd.DataFrame(size_features_list, index=df.index)
        
        print(f"  ✓ Извлечено {tfidf_features.shape[1]} TF-IDF признаков")
        print(f"  ✓ Извлечено {size_features_df.shape[1]} размерных признаков")
        
        return tfidf_features, size_features_df
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        """
        Обучение и применение извлечения признаков.
        
        Args:
            df: Датафрейм для обучения и трансформации
            
        Returns:
            Tuple[csr_matrix, pd.DataFrame]: TF-IDF матрица и датафрейм с признаками
        """
        self.fit(df)
        return self.transform(df)


class ImprovedModel:
    """
    Улучшенная модель с текстовыми признаками на основе LightGBM.
    
    Attributes:
        target_columns (List[str]): Список целевых переменных
        models (Dict): Словарь с обученными моделями для каждого таргета
        feature_names (List[str]): Список названий признаков
        text_feature_names (List[str]): Список названий текстовых признаков
        feature_importances (Dict): Feature importance для каждой модели
    """
    
    def __init__(self, target_columns: List[str] = None, 
                 lgb_params: Dict = None):
        """
        Инициализация улучшенной модели.
        
        Args:
            target_columns: Список целевых переменных
            lgb_params: Параметры для LightGBM
        """
        self.target_columns = target_columns or [
            'real_weight', 'real_height', 'real_width', 'real_length'
        ]
        
        # Улучшенные параметры LightGBM
        self.lgb_params = lgb_params or {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 64,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        self.models = {}
        self.feature_names = None
        self.text_feature_names = None
        self.feature_importances = {}
        
    def prepare_features(self, df: pd.DataFrame,
                        tfidf_features: Optional[csr_matrix] = None,
                        text_features_df: Optional[pd.DataFrame] = None,
                        is_train: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Подготовка и объединение всех типов признаков для обучения/предсказания.
        
        Процесс подготовки:
        1. Извлечение табличных признаков (исключая служебные колонки)
        2. Добавление размерных текстовых признаков
        3. Добавление TF-IDF признаков (конвертация sparse -> dense)
        4. Горизонтальное объединение всех признаков
        5. Извлечение таргетов (только для train)
        
        Args:
            df: Датафрейм с табличными данными (после предобработки)
            tfidf_features: Sparse матрица с TF-IDF признаками (shape: [n_samples, ~300])
            text_features_df: Датафрейм с размерными текстовыми признаками (shape: [n_samples, ~50])
            is_train: Флаг обучающей выборки (если True, возвращает также таргеты)
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - X: Объединенные признаки (shape: [n_samples, ~366])
                - y: Таргеты (shape: [n_samples, 4]) или None для test
                
        Raises:
            ValueError: Если целевые переменные не найдены в train датафрейме
            
        Note:
            При первом вызове сохраняет названия признаков для последующего использования
        """
        # Исключаем колонки, которые не нужны для обучения
        exclude_cols = ['item_id', 'id'] + self.target_columns + [
            'title', 'description', 'image_name', 'image_path', 'order_date',
            'seller_id', 'buyer_id'
        ]
        
        # Выбираем только числовые признаки из табличных данных
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and 
                       df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        if self.feature_names is None:
            self.feature_names = feature_cols
            print(f"Используется {len(self.feature_names)} табличных признаков")
        
        # Табличные признаки
        X_tabular = df[self.feature_names].values
        
        # Добавляем текстовые признаки
        feature_parts = [X_tabular]
        
        if text_features_df is not None:
            if self.text_feature_names is None:
                self.text_feature_names = list(text_features_df.columns)
                print(f"Используется {len(self.text_feature_names)} дополнительных текстовых признаков")
            
            X_text = text_features_df[self.text_feature_names].values
            feature_parts.append(X_text)
        
        if tfidf_features is not None:
            # Конвертируем sparse матрицу в dense для LightGBM
            X_tfidf = tfidf_features.toarray()
            feature_parts.append(X_tfidf)
            print(f"Добавлено {X_tfidf.shape[1]} TF-IDF признаков")
        
        # Объединяем все признаки
        X = np.hstack(feature_parts)
        print(f"Итоговая размерность признаков: {X.shape}")
        
        if is_train:
            # Проверяем наличие таргетов
            available_targets = [col for col in self.target_columns if col in df.columns]
            if not available_targets:
                raise ValueError("Целевые переменные не найдены в датафрейме")
            y = df[available_targets].values
            return X, y
        else:
            return X, None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose: bool = True) -> Dict[str, float]:
        """
        Обучение отдельных моделей LightGBM для каждого таргета.
        
        Для каждой целевой переменной (вес, высота, ширина, длина) обучается
        отдельная модель с early stopping на validation данных.
        
        Процесс обучения:
        1. Создание LGBMRegressor с заданными параметрами
        2. Обучение с early stopping (20 раундов без улучшения)
        3. Сохранение модели и feature importance
        4. Оценка на validation (если предоставлен)
        
        Args:
            X_train: Обучающие признаки (shape: [n_samples, n_features])
            y_train: Обучающие таргеты (shape: [n_samples, 4])
            X_val: Валидационные признаки (опционально)
            y_val: Валидационные таргеты (опционально)
            verbose: Выводить прогресс обучения и метрики
            
        Returns:
            Dict[str, float]: Метрики на валидации (если X_val и y_val предоставлены):
                - {target}_log_mae: MAE для каждого таргета
                - macro_log_mae: средний MAE по всем таргетам
                
        Note:
            Использует early stopping для предотвращения переобучения
        """
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ С ТЕКСТОВЫМИ ПРИЗНАКАМИ")
        print("="*60)
        
        for i, target_name in enumerate(self.target_columns):
            if verbose:
                print(f"\n[{i+1}/{len(self.target_columns)}] Обучение модели для {target_name}...")
            
            # Создание модели
            model = lgb.LGBMRegressor(**self.lgb_params)
            
            # Подготовка данных для валидации
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val[:, i])]
            
            # Обучение
            model.fit(
                X_train, y_train[:, i],
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] if eval_set else None
            )
            
            self.models[target_name] = model
            
            # Сохранение feature importance
            self.feature_importances[target_name] = model.feature_importances_
            
            if verbose:
                best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else 'N/A'
                print(f"  ✓ Модель обучена. Best iteration: {best_iter}")
        
        # Оценка на валидации
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val, verbose=verbose)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание для всех таргетов.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            np.ndarray: Предсказания (shape: [n_samples, n_targets])
        """
        if not self.models:
            raise ValueError("Модели не обучены. Сначала вызовите train()")
        
        predictions = []
        for target_name in self.target_columns:
            pred = self.models[target_name].predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray,
                verbose: bool = True) -> Dict[str, float]:
        """
        Оценка модели на валидационных данных.
        
        Args:
            X_val: Валидационные признаки
            y_val: Валидационные таргеты
            verbose: Выводить метрики
            
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        # Получение предсказаний
        y_pred = self.predict(X_val)
        
        # Вычисление метрик
        metrics = calculate_macro_log_mae(y_val, y_pred, self.target_columns)
        
        if verbose:
            print("\n" + "="*60)
            print("МЕТРИКИ НА VALIDATION")
            print("="*60)
            for metric, value in metrics.items():
                if metric == 'macro_log_mae':
                    print(f"\n{'MACRO LOG-MAE':>30}: {value:.6f}")
                else:
                    target = metric.replace('_log_mae', '')
                    print(f"{target:>30}: {value:.6f}")
        
        return metrics
    
    def get_text_feature_importance(self, target_name: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Получение важности текстовых признаков для таргета.
        
        Args:
            target_name: Название таргета
            top_n: Количество топ признаков
            
        Returns:
            List[Tuple[str, float]]: Список (признак, важность)
        """
        if target_name not in self.feature_importances:
            raise ValueError(f"Feature importance для {target_name} не найдена")
        
        importances = self.feature_importances[target_name]
        
        # Индексы текстовых признаков (после табличных)
        n_tabular = len(self.feature_names)
        n_text = len(self.text_feature_names) if self.text_feature_names else 0
        
        if n_text == 0:
            return []
        
        # Важность текстовых признаков
        text_importances = importances[n_tabular:n_tabular + n_text]
        
        # Сортировка и выбор топ-N
        feature_importance_pairs = list(zip(self.text_feature_names, text_importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance_pairs[:top_n]
    
    def save_models(self, models_dir: str = 'models'):
        """
        Сохранение обученных моделей.
        
        Args:
            models_dir: Директория для сохранения моделей
        """
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        for target_name, model in self.models.items():
            filepath = models_path / f'text_model_{target_name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Модель {target_name} сохранена: {filepath}")
        
        # Сохранение feature names
        with open(models_path / 'text_feature_names.pkl', 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'text_feature_names': self.text_feature_names
            }, f)
        print(f"Feature names сохранены: {models_path / 'text_feature_names.pkl'}")


def save_training_log(metrics: Dict[str, float], 
                      baseline_metrics: Dict[str, float],
                      text_feature_importance: Dict,
                      log_dir: str = 'logs'):
    """
    Сохранение лога обучения с сравнением с baseline.
    
    Args:
        metrics: Метрики улучшенной модели
        baseline_metrics: Метрики baseline модели
        text_feature_importance: Важность текстовых признаков
        log_dir: Директория для логов
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    log_file = log_path / 'text_model_log.txt'
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("TEXT FEATURES MODEL TRAINING LOG\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("VALIDATION METRICS:\n")
        f.write("-"*60 + "\n")
        for metric, value in metrics.items():
            baseline_value = baseline_metrics.get(metric, 0)
            improvement = ((baseline_value - value) / baseline_value * 100) if baseline_value > 0 else 0
            f.write(f"{metric:>30}: {value:.6f} (baseline: {baseline_value:.6f}, ")
            f.write(f"улучшение: {improvement:+.2f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("TOP-5 ВАЖНЫХ ТЕКСТОВЫХ ПРИЗНАКОВ:\n")
        f.write("="*60 + "\n")
        
        for target_name, importance_list in text_feature_importance.items():
            f.write(f"\n{target_name.upper()}:\n")
            for feature, importance in importance_list[:5]:
                f.write(f"  {feature:>40}: {importance:>10.2f}\n")
    
    print(f"\nЛог обучения сохранен: {log_file}")


def main():
    """
    Основная функция для обучения модели с текстовыми признаками.
    """
    print("\n" + "="*60)
    print("TEXT FEATURES MODEL PIPELINE")
    print("="*60)
    
    # Baseline метрики для сравнения (из задания)
    baseline_metrics = {
        'real_weight_log_mae': 0.549,
        'real_height_log_mae': 0.531,
        'real_width_log_mae': 0.295,
        'real_length_log_mae': 0.397,
        'macro_log_mae': 0.443
    }
    
    data_dir = '.'
    
    try:
        # 1. Загрузка данных
        print("\n[1/7] Загрузка данных...")
        loader = DataLoader(data_dir=data_dir)
        train_df, test_df = loader.load_data()
        
        # 2. Разделение на train/validation
        print("\n[2/7] Разделение на train/validation...")
        train_split, val_split = loader.split_data(
            test_size=0.2, 
            random_state=42,
            stratify_column='category_name'
        )
        
        # 3. Предобработка табличных данных
        print("\n[3/7] Предобработка табличных данных...")
        
        preprocessor = FeaturePreprocessor()
        
        # Сохраняем таргеты отдельно
        target_cols = ['real_weight', 'real_height', 'real_width', 'real_length']
        train_targets = train_split[target_cols].copy()
        val_targets = val_split[target_cols].copy()
        
        # Обучение препроцессора на train данных
        train_processed = preprocessor.fit_transform(
            train_split,
            transform_targets=False
        )
        
        # Применение к validation данных
        val_processed = preprocessor.transform(
            val_split,
            transform_targets=False
        )
        
        # Добавляем таргеты обратно
        for col in target_cols:
            train_processed[col] = train_targets[col].values
            val_processed[col] = val_targets[col].values
        
        # 4. Извлечение текстовых признаков
        print("\n[4/7] Извлечение текстовых признаков...")
        text_extractor = TextFeatureExtractor(
            title_max_features=100,
            desc_max_features=200,
            ngram_range=(1, 2)
        )
        
        # Обучение и трансформация train данных
        train_tfidf, train_text_features = text_extractor.fit_transform(train_split)
        
        # Трансформация validation данных
        val_tfidf, val_text_features = text_extractor.transform(val_split)
        
        # Сохранение text extractor
        with open(f'{data_dir}/text_extractor.pkl', 'wb') as f:
            pickle.dump(text_extractor, f)
        print("TextFeatureExtractor сохранен")
        
        # 5. Подготовка данных для обучения
        print("\n[5/7] Подготовка признаков...")
        improved_model = ImprovedModel(target_columns=target_cols)
        
        X_train, y_train = improved_model.prepare_features(
            train_processed, 
            tfidf_features=train_tfidf,
            text_features_df=train_text_features,
            is_train=True
        )
        X_val, y_val = improved_model.prepare_features(
            val_processed,
            tfidf_features=val_tfidf,
            text_features_df=val_text_features,
            is_train=True
        )
        
        print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Val shape: X={X_val.shape}, y={y_val.shape}")
        
        # 6. Обучение модели
        print("\n[6/7] Обучение моделей...")
        metrics = improved_model.train(
            X_train, y_train,
            X_val, y_val,
            verbose=True
        )
        
        # Анализ важности текстовых признаков
        print("\n" + "="*60)
        print("ВАЖНОСТЬ ТЕКСТОВЫХ ПРИЗНАКОВ (TOP-5)")
        print("="*60)
        
        text_importance_dict = {}
        for target_name in target_cols:
            print(f"\n{target_name.upper()}:")
            importance_list = improved_model.get_text_feature_importance(target_name, top_n=5)
            text_importance_dict[target_name] = importance_list
            for feature, importance in importance_list:
                print(f"  {feature:>40}: {importance:>10.2f}")
        
        # Сравнение с baseline
        print("\n" + "="*60)
        print("СРАВНЕНИЕ С BASELINE")
        print("="*60)
        for metric, value in metrics.items():
            baseline_value = baseline_metrics.get(metric, 0)
            if baseline_value > 0:
                improvement = ((baseline_value - value) / baseline_value * 100)
                print(f"{metric:>30}: {value:.6f} -> улучшение: {improvement:+.2f}%")
        
        # Сохранение моделей
        improved_model.save_models(models_dir=f'{data_dir}/models')
        
        # Сохранение лога
        save_training_log(
            metrics,
            baseline_metrics,
            text_importance_dict,
            log_dir=f'{data_dir}/logs'
        )
        
        # 7. Предсказания на test данных
        print("\n[7/7] Создание submission файла...")
        
        # Предобработка test данных
        test_processed = preprocessor.transform(test_df, transform_targets=False)
        
        # Извлечение текстовых признаков для test
        test_tfidf, test_text_features = text_extractor.transform(test_df)
        
        X_test, _ = improved_model.prepare_features(
            test_processed,
            tfidf_features=test_tfidf,
            text_features_df=test_text_features,
            is_train=False
        )
        
        # Предсказания
        test_predictions = improved_model.predict(X_test)
        
        # Создание submission (используем правильные имена колонок для submission)
        submission_target_cols = ['weight', 'height', 'width', 'length']
        submission = create_submission(
            test_ids=test_df['item_id'].values,
            predictions=test_predictions,
            target_columns=submission_target_cols,
            filepath=f'{data_dir}/submission_with_text.csv'
        )
        
        print("\n" + "="*60)
        print("УЛУЧШЕННАЯ МОДЕЛЬ С ТЕКСТОВЫМИ ПРИЗНАКАМИ ГОТОВА!")
        print("="*60)
        print(f"\nМакро Log-MAE на validation: {metrics['macro_log_mae']:.6f}")
        baseline_macro = baseline_metrics['macro_log_mae']
        improvement = ((baseline_macro - metrics['macro_log_mae']) / baseline_macro * 100)
        print(f"Baseline Macro Log-MAE: {baseline_macro:.6f}")
        print(f"Улучшение: {improvement:+.2f}%")
        
        print(f"\nФайлы сохранены:")
        print(f"  - Модели: {data_dir}/models/text_model_*.pkl")
        print(f"  - Text Extractor: {data_dir}/text_extractor.pkl")
        print(f"  - Submission: {data_dir}/submission_with_text.csv")
        print(f"  - Лог: {data_dir}/logs/text_model_log.txt")
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        print("Убедитесь, что файлы train.parquet и test.parquet находятся в директории:")
        print(f"  {data_dir}/")
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
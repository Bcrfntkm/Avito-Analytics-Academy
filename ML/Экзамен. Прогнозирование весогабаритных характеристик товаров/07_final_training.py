"""
Финальный скрипт для обучения мультимодальной модели с использованием всех признаков.

Объединяет:
- Табличные признаки (категориальные, числовые, временные)
- Текстовые признаки (TF-IDF + размерные характеристики)
- Визуальные признаки (предвычисленные image embeddings из ResNet50)

Метрика: Macro Log-MAE
Цель: Улучшить результат относительно baseline (0.443) и text модели (0.4345)
"""

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
import warnings
import sys
import importlib.util

warnings.filterwarnings('ignore')

# Импорт компонентов из других скриптов
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Импорт DataLoader и FeaturePreprocessor
spec = importlib.util.spec_from_file_location(
    "data_pipeline_02",
    current_dir / "02_data_pipeline.py"
)
data_pipeline_02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_pipeline_02)

DataLoader = data_pipeline_02.DataLoader
FeaturePreprocessor = data_pipeline_02.FeaturePreprocessor
calculate_macro_log_mae = data_pipeline_02.calculate_macro_log_mae
create_submission = data_pipeline_02.create_submission

# Импорт TextFeatureExtractor
spec = importlib.util.spec_from_file_location(
    "text_features_04",
    current_dir / "04_text_features_model.py"
)
text_features_04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text_features_04)

TextFeatureExtractor = text_features_04.TextFeatureExtractor


def check_image_features_availability() -> Tuple[bool, str]:
    """
    Проверка наличия предвычисленных image features.
    
    Returns:
        Tuple[bool, str]: (доступны ли файлы, сообщение)
    """
    train_features_path = Path('image_features/train_image_features.npy')
    test_features_path = Path('image_features/test_image_features.npy')
    
    if not train_features_path.exists() or not test_features_path.exists():
        message = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   IMAGE FEATURES НЕ НАЙДЕНЫ                                ║
╚════════════════════════════════════════════════════════════════════════════╝

Для обучения финальной модели необходимы предвычисленные image features.

Отсутствующие файлы:
"""
        if not train_features_path.exists():
            message += f"  ❌ {train_features_path}\n"
        if not test_features_path.exists():
            message += f"  ❌ {test_features_path}\n"
        
        message += """
Инструкции по извлечению признаков:

1. Запустите скрипт извлечения image features:
   python 05_image_features_extractor.py

2. Дождитесь завершения (примерно 2 часа)

3. После завершения запустите этот скрипт снова:
   python 07_final_training.py

Альтернатива: Обучение без image features
   Если вы хотите обучить модель только на табличных и текстовых признаках,
   используйте скрипт 04_text_features_model.py
"""
        return False, message
    
    return True, "✓ Image features найдены"


def load_image_features(full_train_df: pd.DataFrame,
                       train_split_ids: np.ndarray,
                       val_split_ids: np.ndarray,
                       test_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загрузка предвычисленных image features и разделение на train/val.
    
    Args:
        full_train_df: Полный train датафрейм для создания маппинга
        train_split_ids: ID объектов из train split
        val_split_ids: ID объектов из validation split
        test_ids: ID объектов из test датасета
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Train, val и test image features
    """
    print("\n[Загрузка image features]")
    
    # Загрузка train features (полный набор)
    train_features_path = Path('image_features/train_image_features.npy')
    print(f"  - Загрузка {train_features_path}...")
    full_train_features = np.load(train_features_path)
    print(f"    ✓ Загружено: {full_train_features.shape}")
    
    # Загрузка test features
    test_features_path = Path('image_features/test_image_features.npy')
    print(f"  - Загрузка {test_features_path}...")
    test_features = np.load(test_features_path)
    print(f"    ✓ Загружено: {test_features.shape}")
    
    # Создание маппинга item_id -> индекс в полном train датасете
    print(f"  - Создание маппинга item_id -> image features...")
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(full_train_df['item_id'].values)}
    
    # Извлечение features для train split
    print(f"  - Извлечение features для train split ({len(train_split_ids)} samples)...")
    train_features = np.array([full_train_features[item_id_to_idx[item_id]]
                               for item_id in train_split_ids])
    print(f"    ✓ Train features: {train_features.shape}")
    
    # Извлечение features для validation split
    print(f"  - Извлечение features для validation split ({len(val_split_ids)} samples)...")
    val_features = np.array([full_train_features[item_id_to_idx[item_id]]
                             for item_id in val_split_ids])
    print(f"    ✓ Val features: {val_features.shape}")
    
    # Проверка размерностей test
    if len(test_features) != len(test_ids):
        raise ValueError(f"Несоответствие размеров: test_features={len(test_features)}, test_ids={len(test_ids)}")
    print(f"    ✓ Test features: {test_features.shape}")
    
    return train_features, val_features, test_features


class MultimodalModel:
    """
    Мультимодальная модель, объединяющая табличные, текстовые и визуальные признаки.
    
    Attributes:
        target_columns (List[str]): Список целевых переменных
        models (Dict): Словарь с обученными моделями для каждого таргета
        feature_counts (Dict): Количество признаков каждого типа
    """
    
    def __init__(self, target_columns: List[str] = None, lgb_params: Dict = None):
        """
        Инициализация мультимодальной модели.
        
        Args:
            target_columns: Список целевых переменных
            lgb_params: Параметры для LightGBM
        """
        self.target_columns = target_columns or [
            'real_weight', 'real_height', 'real_width', 'real_length'
        ]
        
        # Оптимизированные параметры для финальной модели
        self.lgb_params = lgb_params or {
            'n_estimators': 500,
            'learning_rate': 0.02,
            'max_depth': 12,
            'num_leaves': 256,
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
        self.feature_counts = {
            'tabular': 0,
            'text': 0,
            'tfidf': 0,
            'image': 0
        }
        
    def prepare_features(self, 
                        df: pd.DataFrame,
                        tfidf_features: Optional[csr_matrix] = None,
                        text_features_df: Optional[pd.DataFrame] = None,
                        image_features: Optional[np.ndarray] = None,
                        is_train: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Подготовка всех признаков для обучения/предсказания.
        
        Args:
            df: Датафрейм с табличными данными
            tfidf_features: Sparse матрица с TF-IDF признаками
            text_features_df: Датафрейм с дополнительными текстовыми признаками
            image_features: Массив с image embeddings
            is_train: Флаг обучающей выборки
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: X (признаки) и y (таргеты)
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
            self.feature_counts['tabular'] = len(self.feature_names)
            print(f"  ✓ Табличные признаки: {self.feature_counts['tabular']}")
        
        # 1. Табличные признаки
        X_tabular = df[self.feature_names].values
        feature_parts = [X_tabular]
        
        # 2. Дополнительные текстовые признаки (размерные характеристики)
        if text_features_df is not None:
            if self.text_feature_names is None:
                self.text_feature_names = list(text_features_df.columns)
                self.feature_counts['text'] = len(self.text_feature_names)
                print(f"  ✓ Текстовые признаки (размерные): {self.feature_counts['text']}")
            
            X_text = text_features_df[self.text_feature_names].values
            feature_parts.append(X_text)
        
        # 3. TF-IDF признаки
        if tfidf_features is not None:
            X_tfidf = tfidf_features.toarray()
            if self.feature_counts['tfidf'] == 0:
                self.feature_counts['tfidf'] = X_tfidf.shape[1]
                print(f"  ✓ TF-IDF признаки: {self.feature_counts['tfidf']}")
            feature_parts.append(X_tfidf)
        
        # 4. Image признаки
        if image_features is not None:
            if self.feature_counts['image'] == 0:
                self.feature_counts['image'] = image_features.shape[1]
                print(f"  ✓ Image признаки: {self.feature_counts['image']}")
            feature_parts.append(image_features)
        
        # Объединяем все признаки
        X = np.hstack(feature_parts)
        
        total_features = sum(self.feature_counts.values())
        print(f"  ✓ Итого признаков: {total_features} (shape: {X.shape})")
        
        if is_train:
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
        Обучение отдельных моделей для каждого таргета.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие таргеты
            X_val: Валидационные признаки
            y_val: Валидационные таргеты
            verbose: Выводить прогресс обучения
            
        Returns:
            Dict[str, float]: Метрики на валидации
        """
        print("\n" + "="*80)
        print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МУЛЬТИМОДАЛЬНОЙ МОДЕЛИ")
        print("="*80)
        print(f"Параметры LightGBM:")
        for param, value in self.lgb_params.items():
            print(f"  {param}: {value}")
        print()
        
        training_times = {}
        
        for i, target_name in enumerate(self.target_columns):
            if verbose:
                print(f"\n[{i+1}/{len(self.target_columns)}] Обучение модели для {target_name}...")
            
            start_time = datetime.now()
            
            # Создание модели
            model = lgb.LGBMRegressor(**self.lgb_params)
            
            # Подготовка данных для валидации
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val[:, i])]
            
            # Обучение с early stopping
            model.fit(
                X_train, y_train[:, i],
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] if eval_set else None
            )
            
            self.models[target_name] = model
            
            training_time = (datetime.now() - start_time).total_seconds()
            training_times[target_name] = training_time
            
            if verbose:
                best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else 'N/A'
                print(f"  ✓ Модель обучена за {training_time:.1f}s. Best iteration: {best_iter}")
        
        # Оценка на валидации
        metrics = {}
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val, verbose=verbose)
        
        metrics['training_times'] = training_times
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
        y_pred = self.predict(X_val)
        metrics = calculate_macro_log_mae(y_val, y_pred, self.target_columns)
        
        if verbose:
            print("\n" + "="*80)
            print("МЕТРИКИ НА VALIDATION")
            print("="*80)
            for metric, value in metrics.items():
                if metric == 'macro_log_mae':
                    print(f"\n{'MACRO LOG-MAE':>35}: {value:.6f}")
                else:
                    target = metric.replace('_log_mae', '')
                    print(f"{target:>35}: {value:.6f}")
        
        return metrics
    
    def get_feature_importance_by_type(self, target_name: str) -> Dict[str, float]:
        """
        Получение важности признаков по типам для таргета.
        
        Args:
            target_name: Название таргета
            
        Returns:
            Dict[str, float]: Средняя важность по типам признаков
        """
        if target_name not in self.models:
            raise ValueError(f"Модель для {target_name} не найдена")
        
        importances = self.models[target_name].feature_importances_
        
        # Разделение важности по типам признаков
        idx = 0
        type_importances = {}
        
        # Табличные
        n_tabular = self.feature_counts['tabular']
        if n_tabular > 0:
            type_importances['tabular'] = np.mean(importances[idx:idx+n_tabular])
            idx += n_tabular
        
        # Текстовые (размерные)
        n_text = self.feature_counts['text']
        if n_text > 0:
            type_importances['text'] = np.mean(importances[idx:idx+n_text])
            idx += n_text
        
        # TF-IDF
        n_tfidf = self.feature_counts['tfidf']
        if n_tfidf > 0:
            type_importances['tfidf'] = np.mean(importances[idx:idx+n_tfidf])
            idx += n_tfidf
        
        # Image
        n_image = self.feature_counts['image']
        if n_image > 0:
            type_importances['image'] = np.mean(importances[idx:idx+n_image])
        
        return type_importances
    
    def save_models(self, models_dir: str = 'models'):
        """
        Сохранение обученных моделей.
        
        Args:
            models_dir: Директория для сохранения моделей
        """
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        for target_name, model in self.models.items():
            filepath = models_path / f'final_{target_name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ Модель {target_name} сохранена: {filepath}")
        
        # Сохранение метаданных
        metadata = {
            'feature_names': self.feature_names,
            'text_feature_names': self.text_feature_names,
            'feature_counts': self.feature_counts,
            'lgb_params': self.lgb_params
        }
        with open(models_path / 'final_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"  ✓ Метаданные сохранены: {models_path / 'final_metadata.pkl'}")


def save_detailed_log(metrics: Dict[str, float],
                     baseline_metrics: Dict[str, float],
                     text_metrics: Dict[str, float],
                     feature_importance_by_type: Dict,
                     log_dir: str = 'logs'):
    """
    Сохранение детального лога обучения.
    
    Args:
        metrics: Метрики финальной модели
        baseline_metrics: Метрики baseline модели
        text_metrics: Метрики text модели
        feature_importance_by_type: Важность признаков по типам
        log_dir: Директория для логов
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    log_file = log_path / 'final_training_log.txt'
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FINAL MULTIMODAL MODEL TRAINING LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Время обучения
        if 'training_times' in metrics:
            f.write("ВРЕМЯ ОБУЧЕНИЯ:\n")
            f.write("-"*80 + "\n")
            for target, time_sec in metrics['training_times'].items():
                f.write(f"{target:>35}: {time_sec:.1f}s\n")
            f.write("\n")
        
        # Метрики
        f.write("VALIDATION METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<35} {'Final':<12} {'Text':<12} {'Baseline':<12} {'vs Text':<12} {'vs Baseline':<12}\n")
        f.write("-"*80 + "\n")
        
        for metric in ['real_weight_log_mae', 'real_height_log_mae', 
                      'real_width_log_mae', 'real_length_log_mae', 'macro_log_mae']:
            if metric in metrics:
                final_val = metrics[metric]
                text_val = text_metrics.get(metric, 0)
                baseline_val = baseline_metrics.get(metric, 0)
                
                vs_text = ((text_val - final_val) / text_val * 100) if text_val > 0 else 0
                vs_baseline = ((baseline_val - final_val) / baseline_val * 100) if baseline_val > 0 else 0
                
                f.write(f"{metric:<35} {final_val:<12.6f} {text_val:<12.6f} {baseline_val:<12.6f} "
                       f"{vs_text:>+10.2f}% {vs_baseline:>+10.2f}%\n")
        
        # Важность признаков по типам
        f.write("\n" + "="*80 + "\n")
        f.write("ВАЖНОСТЬ ПРИЗНАКОВ ПО ТИПАМ (средняя):\n")
        f.write("="*80 + "\n")
        
        for target_name, type_importance in feature_importance_by_type.items():
            f.write(f"\n{target_name.upper()}:\n")
            for feat_type, importance in sorted(type_importance.items(), 
                                               key=lambda x: x[1], reverse=True):
                f.write(f"  {feat_type:>15}: {importance:>10.2f}\n")
    
    print(f"\n✓ Детальный лог сохранен: {log_file}")


def main():
    """
    Основная функция для обучения финальной мультимодальной модели.
    """
    print("\n" + "="*80)
    print("FINAL MULTIMODAL MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Проверка наличия image features
    features_available, message = check_image_features_availability()
    if not features_available:
        print(message)
        return
    
    print(message)
    
    # Метрики для сравнения
    baseline_metrics = {
        'real_weight_log_mae': 0.549,
        'real_height_log_mae': 0.531,
        'real_width_log_mae': 0.295,
        'real_length_log_mae': 0.397,
        'macro_log_mae': 0.443
    }
    
    text_metrics = {
        'real_weight_log_mae': 0.4345,  # Примерные значения
        'real_height_log_mae': 0.4345,
        'real_width_log_mae': 0.4345,
        'real_length_log_mae': 0.4345,
        'macro_log_mae': 0.4345
    }
    
    data_dir = '.'
    
    try:
        # 1. Загрузка данных
        print("\n[1/9] Загрузка данных...")
        loader = DataLoader(data_dir=data_dir)
        train_df, test_df = loader.load_data()
        
        # 2. Разделение на train/validation
        print("\n[2/9] Разделение на train/validation...")
        train_split, val_split = loader.split_data(
            test_size=0.2,
            random_state=42,
            stratify_column='category_name'
        )
        
        # 3. Предобработка табличных данных
        print("\n[3/9] Предобработка табличных данных...")
        preprocessor = FeaturePreprocessor()
        
        target_cols = ['real_weight', 'real_height', 'real_width', 'real_length']
        train_targets = train_split[target_cols].copy()
        val_targets = val_split[target_cols].copy()
        
        train_processed = preprocessor.fit_transform(train_split, transform_targets=False)
        val_processed = preprocessor.transform(val_split, transform_targets=False)
        
        for col in target_cols:
            train_processed[col] = train_targets[col].values
            val_processed[col] = val_targets[col].values
        
        # 4. Извлечение текстовых признаков
        print("\n[4/9] Извлечение текстовых признаков...")
        
        # Загрузка или создание text extractor
        text_extractor_path = Path(f'{data_dir}/text_extractor.pkl')
        if text_extractor_path.exists():
            print("  - Загрузка сохраненного TextFeatureExtractor...")
            with open(text_extractor_path, 'rb') as f:
                text_extractor = pickle.load(f)
        else:
            print("  - Создание нового TextFeatureExtractor...")
            text_extractor = TextFeatureExtractor(
                title_max_features=100,
                desc_max_features=200,
                ngram_range=(1, 2)
            )
            text_extractor.fit(train_split)
            with open(text_extractor_path, 'wb') as f:
                pickle.dump(text_extractor, f)
        
        train_tfidf, train_text_features = text_extractor.transform(train_split)
        val_tfidf, val_text_features = text_extractor.transform(val_split)
        
        # 5. Загрузка image features
        print("\n[5/9] Загрузка image features...")
        train_img_feat, val_img_feat, test_image_features = load_image_features(
            loader.train_df,
            train_split['item_id'].values,
            val_split['item_id'].values,
            test_df['item_id'].values
        )
        
        # 6. Подготовка признаков
        print("\n[6/9] Подготовка всех признаков...")
        multimodal_model = MultimodalModel(target_columns=target_cols)
        
        X_train, y_train = multimodal_model.prepare_features(
            train_processed,
            tfidf_features=train_tfidf,
            text_features_df=train_text_features,
            image_features=train_img_feat,
            is_train=True
        )
        
        X_val, y_val = multimodal_model.prepare_features(
            val_processed,
            tfidf_features=val_tfidf,
            text_features_df=val_text_features,
            image_features=val_img_feat,
            is_train=True
        )
        
        print(f"\nИтоговые размеры:")
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
        
        # 7. Обучение модели
        print("\n[7/9] Обучение финальной модели...")
        metrics = multimodal_model.train(
            X_train, y_train,
            X_val, y_val,
            verbose=True
        )
        
        # Анализ важности признаков по типам
        print("\n" + "="*80)
        print("ВАЖНОСТЬ ПРИЗНАКОВ ПО ТИПАМ")
        print("="*80)
        
        feature_importance_by_type = {}
        for target_name in target_cols:
            print(f"\n{target_name.upper()}:")
            type_importance = multimodal_model.get_feature_importance_by_type(target_name)
            feature_importance_by_type[target_name] = type_importance
            for feat_type, importance in sorted(type_importance.items(), 
                                               key=lambda x: x[1], reverse=True):
                print(f"  {feat_type:>15}: {importance:>10.2f}")
        
        # Сравнение с предыдущими моделями
        print("\n" + "="*80)
        print("СРАВНЕНИЕ С ПРЕДЫДУЩИМИ МОДЕЛЯМИ")
        print("="*80)
        print(f"{'Metric':<35} {'Final':<12} {'Text':<12} {'Baseline':<12}")
        print("-"*80)
        
        for metric in ['real_weight_log_mae', 'real_height_log_mae',
                      'real_width_log_mae', 'real_length_log_mae', 'macro_log_mae']:
            if metric in metrics:
                final_val = metrics[metric]
                text_val = text_metrics.get(metric, 0)
                baseline_val = baseline_metrics.get(metric, 0)
                print(f"{metric:<35} {final_val:<12.6f} {text_val:<12.6f} {baseline_val:<12.6f}")
        
        # Сохранение моделей
        print("\n[8/9] Сохранение моделей...")
        multimodal_model.save_models(models_dir=f'{data_dir}/models')
        
        # Сохранение детального лога
        save_detailed_log(
            metrics,
            baseline_metrics,
            text_metrics,
            feature_importance_by_type,
            log_dir=f'{data_dir}/logs'
        )
        
        # 9. Создание submission файла
        print("\n[9/9] Создание submission файла...")
        
        # Предобработка test данных
        test_processed = preprocessor.transform(test_df, transform_targets=False)
        
        # Извлечение текстовых признаков для test
        test_tfidf, test_text_features = text_extractor.transform(test_df)
        
        # Подготовка всех признаков для test
        X_test, _ = multimodal_model.prepare_features(
            test_processed,
            tfidf_features=test_tfidf,
            text_features_df=test_text_features,
            image_features=test_image_features,
            is_train=False
        )
        
        print(f"  ✓ Test features подготовлены: {X_test.shape}")
        
        # Предсказания
        print("  - Выполнение предсказаний...")
        test_predictions = multimodal_model.predict(X_test)
        
        # Проверка предсказаний
        print(f"  ✓ Предсказания получены: {test_predictions.shape}")
        print(f"  - Проверка значений...")
        
        # Проверка на отрицательные значения
        if np.any(test_predictions < 0):
            print("  ⚠ Обнаружены отрицательные значения, заменяем на минимальные положительные")
            test_predictions = np.maximum(test_predictions, 0.001)
        
        # Статистика предсказаний
        print(f"  - Статистика предсказаний:")
        for i, col in enumerate(['weight', 'height', 'width', 'length']):
            print(f"    {col}: min={test_predictions[:, i].min():.3f}, "
                  f"max={test_predictions[:, i].max():.3f}, "
                  f"mean={test_predictions[:, i].mean():.3f}")
        
        # Создание submission файла
        submission_target_cols = ['weight', 'height', 'width', 'length']
        submission = create_submission(
            test_ids=test_df['item_id'].values,
            predictions=test_predictions,
            target_columns=submission_target_cols,
            filepath=f'{data_dir}/submission_final.csv'
        )
        
        # Проверка формата submission
        print(f"\n  ✓ Submission файл создан: submission_final.csv")
        print(f"  - Размер: {submission.shape}")
        print(f"  - Колонки: {list(submission.columns)}")
        print(f"  - Первые строки:")
        print(submission.head())
        
        # Проверка на корректность
        expected_rows = 70274
        if len(submission) != expected_rows:
            print(f"  ⚠ ВНИМАНИЕ: Ожидалось {expected_rows} строк, получено {len(submission)}")
        else:
            print(f"  ✓ Количество строк корректно: {expected_rows}")
        
        # Проверка на положительные значения
        for col in submission_target_cols:
            if (submission[col] <= 0).any():
                print(f"  ⚠ ВНИМАНИЕ: В колонке {col} есть неположительные значения!")
            else:
                print(f"  ✓ Все значения в {col} положительные")
        
        # Финальный отчет
        print("\n" + "="*80)
        print("ФИНАЛЬНАЯ МУЛЬТИМОДАЛЬНАЯ МОДЕЛЬ ГОТОВА!")
        print("="*80)
        
        print(f"\n📊 РЕЗУЛЬТАТЫ НА VALIDATION:")
        print(f"  Macro Log-MAE: {metrics['macro_log_mae']:.6f}")
        
        # Улучшение относительно baseline
        baseline_macro = baseline_metrics['macro_log_mae']
        improvement_baseline = ((baseline_macro - metrics['macro_log_mae']) / baseline_macro * 100)
        print(f"\n  Baseline:      {baseline_macro:.6f}")
        print(f"  Улучшение:     {improvement_baseline:+.2f}%")
        
        # Улучшение относительно text модели
        text_macro = text_metrics['macro_log_mae']
        improvement_text = ((text_macro - metrics['macro_log_mae']) / text_macro * 100)
        print(f"\n  Text модель:   {text_macro:.6f}")
        print(f"  Улучшение:     {improvement_text:+.2f}%")
        
        print(f"\n📁 СОХРАНЕННЫЕ ФАЙЛЫ:")
        print(f"  ✓ Модели:      {data_dir}/models/final_*.pkl")
        print(f"  ✓ Метаданные:  {data_dir}/models/final_metadata.pkl")
        print(f"  ✓ Submission:  {data_dir}/submission_final.csv")
        print(f"  ✓ Лог:         {data_dir}/logs/final_training_log.txt")
        
        end_time = datetime.now()
        print(f"\n⏱ Завершено: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        print("Убедитесь, что все необходимые файлы находятся в директории:")
        print(f"  - {data_dir}/train.parquet")
        print(f"  - {data_dir}/test.parquet")
        print(f"  - {data_dir}/image_features/train_image_features.npy")
        print(f"  - {data_dir}/image_features/test_image_features.npy")
    except MemoryError as e:
        print(f"\n❌ Ошибка памяти: {e}")
        print("\nРекомендации:")
        print("  1. Закройте другие приложения для освобождения памяти")
        print("  2. Уменьшите размер батча в параметрах")
        print("  3. Используйте машину с большим объемом RAM")
    except Exception as e:
        print(f"\n❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
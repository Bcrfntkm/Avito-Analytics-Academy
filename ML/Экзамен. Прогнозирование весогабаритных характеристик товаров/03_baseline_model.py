"""
Baseline модель для прогнозирования весогабаритных характеристик товаров.

Использует LightGBM для предсказания 4 таргетов:
- item_weight (вес)
- item_height (высота)
- item_width (ширина)
- item_length (длина)

Метрика: Macro Log-MAE
"""

import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings

# Импорт из пайплайна
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Пытаемся импортировать напрямую
    import importlib.util
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


class BaselineModel:
    """
    Baseline модель на основе LightGBM для каждого таргета отдельно.
    
    Attributes:
        target_columns (List[str]): Список целевых переменных
        models (Dict): Словарь с обученными моделями для каждого таргета
        feature_names (List[str]): Список названий признаков
        feature_importances (Dict): Feature importance для каждой модели
    """
    
    def __init__(self, target_columns: List[str] = None, 
                 lgb_params: Dict = None):
        """
        Инициализация baseline модели.
        
        Args:
            target_columns: Список целевых переменных
            lgb_params: Параметры для LightGBM
        """
        self.target_columns = target_columns or [
            'real_weight', 'real_height', 'real_width', 'real_length'
        ]
        
        # Базовые параметры LightGBM
        self.lgb_params = lgb_params or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 7,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        self.models = {}
        self.feature_names = None
        self.feature_importances = {}
        
    def prepare_features(self, df: pd.DataFrame, 
                        is_train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка признаков для обучения/предсказания.
        
        Args:
            df: Датафрейм с данными
            is_train: Флаг обучающей выборки
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (признаки) и y (таргеты, если is_train=True)
        """
        # Исключаем колонки, которые не нужны для обучения
        exclude_cols = ['item_id', 'id'] + self.target_columns + [
            'title', 'description', 'image_name', 'image_path', 'order_date',
            'seller_id', 'buyer_id'
        ]
        
        # Выбираем только числовые признаки
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and 
                       df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        if self.feature_names is None:
            self.feature_names = feature_cols
            print(f"Используется {len(self.feature_names)} признаков")
        
        X = df[self.feature_names].values
        
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
        Обучение отдельных моделей для каждого таргета.
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие таргеты
            X_val: Валидационные признаки (опционально)
            y_val: Валидационные таргеты (опционально)
            verbose: Выводить прогресс обучения
            
        Returns:
            Dict[str, float]: Метрики на валидации (если предоставлена)
        """
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ BASELINE МОДЕЛИ")
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
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)] if eval_set else None
            )
            
            self.models[target_name] = model
            
            # Сохранение feature importance
            self.feature_importances[target_name] = dict(
                zip(self.feature_names, model.feature_importances_)
            )
            
            if verbose:
                print(f"  ✓ Модель обучена. Best iteration: {model.best_iteration_ if hasattr(model, 'best_iteration_') else 'N/A'}")
        
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
    
    def get_feature_importance(self, target_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Получение топ-N важных признаков для таргета.
        
        Args:
            target_name: Название таргета
            top_n: Количество топ признаков
            
        Returns:
            pd.DataFrame: Датафрейм с важностью признаков
        """
        if target_name not in self.feature_importances:
            raise ValueError(f"Feature importance для {target_name} не найдена")
        
        importance_dict = self.feature_importances[target_name]
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df.head(top_n)
    
    def print_feature_importances(self, top_n: int = 5):
        """
        Вывод топ-N важных признаков для всех таргетов.
        
        Args:
            top_n: Количество топ признаков
        """
        print("\n" + "="*60)
        print(f"ТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ")
        print("="*60)
        
        for target_name in self.target_columns:
            print(f"\n{target_name.upper()}:")
            importance_df = self.get_feature_importance(target_name, top_n)
            for idx, row in importance_df.iterrows():
                print(f"  {row['feature']:>30}: {row['importance']:>10.2f}")
    
    def save_models(self, models_dir: str = 'models'):
        """
        Сохранение обученных моделей.
        
        Args:
            models_dir: Директория для сохранения моделей
        """
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        for target_name, model in self.models.items():
            filepath = models_path / f'lgb_{target_name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Модель {target_name} сохранена: {filepath}")
        
        # Сохранение feature names
        with open(models_path / 'feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"Feature names сохранены: {models_path / 'feature_names.pkl'}")
    
    def load_models(self, models_dir: str = 'models'):
        """
        Загрузка обученных моделей.
        
        Args:
            models_dir: Директория с моделями
        """
        models_path = Path(models_dir)
        
        if not models_path.exists():
            raise FileNotFoundError(f"Директория {models_dir} не найдена")
        
        # Загрузка feature names
        with open(models_path / 'feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Загрузка моделей
        for target_name in self.target_columns:
            filepath = models_path / f'lgb_{target_name}.pkl'
            with open(filepath, 'rb') as f:
                self.models[target_name] = pickle.load(f)
            print(f"Модель {target_name} загружена: {filepath}")


def save_training_log(metrics: Dict[str, float], 
                      feature_importances: Dict,
                      log_dir: str = 'logs'):
    """
    Сохранение лога обучения.
    
    Args:
        metrics: Метрики модели
        feature_importances: Feature importance для каждого таргета
        log_dir: Директория для логов
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'training_log_{timestamp}.txt'
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BASELINE MODEL TRAINING LOG\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("VALIDATION METRICS:\n")
        f.write("-"*60 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric:>30}: {value:.6f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("FEATURE IMPORTANCES (TOP-5):\n")
        f.write("="*60 + "\n")
        
        for target_name, importance_dict in feature_importances.items():
            f.write(f"\n{target_name.upper()}:\n")
            sorted_features = sorted(importance_dict.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                f.write(f"  {feature:>30}: {importance:>10.2f}\n")
    
    print(f"\nЛог обучения сохранен: {log_file}")


def main():
    """
    Основная функция для обучения baseline модели.
    """
    print("\n" + "="*60)
    print("BASELINE MODEL PIPELINE")
    print("="*60)
    
    # Путь к данным (текущая директория, так как скрипт запускается из неё)
    data_dir = '.'
    
    try:
        # 1. Загрузка данных
        print("\n[1/6] Загрузка данных...")
        loader = DataLoader(data_dir=data_dir)
        train_df, test_df = loader.load_data()
        
        # 2. Разделение на train/validation
        print("\n[2/6] Разделение на train/validation...")
        train_split, val_split = loader.split_data(
            test_size=0.2, 
            random_state=42,
            stratify_column='category_name'
        )
        
        # 3. Предобработка данных
        print("\n[3/6] Предобработка данных...")
        print(f"Колонки в train_split: {list(train_split.columns)}")
        
        preprocessor = FeaturePreprocessor()
        
        # Сохраняем таргеты отдельно
        target_cols = ['real_weight', 'real_height', 'real_width', 'real_length']
        print(f"Используемые целевые колонки: {target_cols}")
        train_targets = train_split[target_cols].copy()
        val_targets = val_split[target_cols].copy()
        
        # Обучение препроцессора на train данных
        train_processed = preprocessor.fit_transform(
            train_split,
            transform_targets=False  # Не трансформируем таргеты для LightGBM
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
        
        # Сохранение препроцессора (пропускаем из-за проблем с pickling)
        try:
            save_preprocessor(preprocessor, f'{data_dir}/preprocessor.pkl')
        except Exception as e:
            print(f"Предупреждение: не удалось сохранить препроцессор: {e}")
            print("Продолжаем без сохранения препроцессора...")
        
        # 4. Подготовка данных для обучения
        print("\n[4/6] Подготовка признаков...")
        baseline_model = BaselineModel()
        
        X_train, y_train = baseline_model.prepare_features(train_processed, is_train=True)
        X_val, y_val = baseline_model.prepare_features(val_processed, is_train=True)
        
        print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Val shape: X={X_val.shape}, y={y_val.shape}")
        
        # 5. Обучение модели
        print("\n[5/6] Обучение моделей...")
        metrics = baseline_model.train(
            X_train, y_train,
            X_val, y_val,
            verbose=True
        )
        
        # Вывод feature importance
        baseline_model.print_feature_importances(top_n=5)
        
        # Сохранение моделей
        baseline_model.save_models(models_dir=f'{data_dir}/models')
        
        # Сохранение лога
        save_training_log(
            metrics, 
            baseline_model.feature_importances,
            log_dir=f'{data_dir}/logs'
        )
        
        # 6. Предсказания на test данных
        print("\n[6/6] Создание submission файла...")
        
        # Предобработка test данных
        test_processed = preprocessor.transform(test_df, transform_targets=False)
        X_test, _ = baseline_model.prepare_features(test_processed, is_train=False)
        
        # Предсказания
        test_predictions = baseline_model.predict(X_test)
        
        # Создание submission (используем правильные имена колонок для submission)
        submission_target_cols = ['weight', 'height', 'width', 'length']
        submission = create_submission(
            test_ids=test_df['item_id'].values,
            predictions=test_predictions,
            target_columns=submission_target_cols,
            filepath=f'{data_dir}/submission.csv'
        )
        
        print("\n" + "="*60)
        print("BASELINE MODEL ГОТОВА!")
        print("="*60)
        print(f"\nМакро Log-MAE на validation: {metrics['macro_log_mae']:.6f}")
        print(f"\nФайлы сохранены:")
        print(f"  - Модели: {data_dir}/models/")
        print(f"  - Препроцессор: {data_dir}/preprocessor.pkl")
        print(f"  - Submission: {data_dir}/submission.csv")
        print(f"  - Лог: {data_dir}/logs/")
        
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
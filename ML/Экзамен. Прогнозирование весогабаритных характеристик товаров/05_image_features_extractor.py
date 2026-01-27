"""
Скрипт для извлечения признаков из изображений товаров с использованием предобученной CNN и GPU.

Использует ResNet50 для извлечения embeddings из изображений.
Оптимизирован для работы с GPU на MacBook (MPS - Metal Performance Shaders).
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


class ImageDataset(Dataset):
    """Dataset для загрузки изображений по путям."""
    
    def __init__(self, image_paths: List[str], transform=None):
        """
        Args:
            image_paths: список путей к изображениям
            transform: трансформации для изображений
        """
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Загрузка изображения
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, idx, True  # image, index, success_flag
            
        except Exception as e:
            # В случае ошибки возвращаем нулевой тензор
            if self.transform:
                # Создаем пустое изображение того же размера
                dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
                image = self.transform(dummy_image)
            else:
                image = torch.zeros(3, 224, 224)
                
            return image, idx, False  # image, index, success_flag


class ImageFeatureExtractor:
    """
    Класс для извлечения признаков из изображений с использованием предобученной CNN.
    
    Использует ResNet50 и извлекает embeddings из предпоследнего слоя (2048 признаков).
    Оптимизирован для работы с GPU на MacBook (MPS).
    """
    
    def __init__(self, model_name: str = 'resnet50', batch_size: int = 32, device: str = 'auto'):
        """
        Args:
            model_name: название модели ('resnet50' или 'efficientnet_b0')
            batch_size: размер батча для обработки
            device: устройство для вычислений ('auto', 'mps', 'cpu')
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Определение устройства
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("✓ Используется GPU (MPS - Metal Performance Shaders)")
            else:
                self.device = torch.device("cpu")
                print("⚠ GPU недоступен, используется CPU")
        else:
            self.device = torch.device(device)
            
        # Загрузка модели
        self.model = self._load_model()
        self.model.eval()  # Режим inference
        
        # Трансформации для изображений (стандартные для ImageNet)
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✓ Модель {model_name} загружена на {self.device}")
        
    def _load_model(self) -> nn.Module:
        """Загрузка предобученной модели."""
        if self.model_name == 'resnet50':
            # Загрузка ResNet50
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Удаляем последний слой классификации
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif self.model_name == 'efficientnet_b0':
            # Загрузка EfficientNet-B0
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Удаляем последний слой классификации
            model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Неподдерживаемая модель: {self.model_name}")
            
        return model.to(self.device)
    
    def fit(self, X=None, y=None):
        """Пустой метод для совместимости с sklearn pipeline."""
        return self
    
    def transform(self, image_paths: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Извлечение признаков из изображений.
        
        Args:
            image_paths: список путей к изображениям
            show_progress: показывать прогресс-бар
            
        Returns:
            numpy array с признаками размера (n_samples, n_features)
        """
        # Создание dataset и dataloader
        dataset = ImageDataset(image_paths, transform=self.image_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Для MPS лучше использовать 0
            pin_memory=False
        )
        
        features_list = []
        failed_indices = []
        
        # Отключаем градиенты для экономии памяти
        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Извлечение признаков") if show_progress else dataloader
            
            for images, indices, success_flags in iterator:
                # Перемещаем на устройство
                images = images.to(self.device)
                
                # Извлечение признаков
                features = self.model(images)
                
                # Преобразование в numpy
                features = features.cpu().numpy()
                
                # Flatten если нужно (для ResNet50: (batch, 2048, 1, 1) -> (batch, 2048))
                if features.ndim > 2:
                    features = features.reshape(features.shape[0], -1)
                
                features_list.append(features)
                
                # Сохраняем индексы неудачных загрузок
                for idx, success in zip(indices.numpy(), success_flags.numpy()):
                    if not success:
                        failed_indices.append(idx)
                
                # Очистка кэша GPU после каждого батча
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
        
        # Объединение всех признаков
        all_features = np.vstack(features_list)
        
        if failed_indices:
            print(f"⚠ Не удалось загрузить {len(failed_indices)} изображений (заполнены нулями)")
        
        return all_features
    
    def fit_transform(self, image_paths: List[str], y=None, show_progress: bool = True) -> np.ndarray:
        """
        Извлечение признаков (комбинация fit и transform).
        
        Args:
            image_paths: список путей к изображениям
            y: игнорируется (для совместимости)
            show_progress: показывать прогресс-бар
            
        Returns:
            numpy array с признаками
        """
        return self.transform(image_paths, show_progress=show_progress)
    
    def get_feature_dim(self) -> int:
        """Получить размерность признаков."""
        if self.model_name == 'resnet50':
            return 2048
        elif self.model_name == 'efficientnet_b0':
            return 1280
        return 0


def load_or_extract_features(
    df: pd.DataFrame,
    image_dir: str,
    cache_path: str,
    extractor: Optional[ImageFeatureExtractor] = None,
    force_extract: bool = False
) -> Tuple[np.ndarray, ImageFeatureExtractor]:
    """
    Загрузка или извлечение признаков из изображений.
    
    Если кэш существует - загружает из него, иначе извлекает признаки заново.
    
    Args:
        df: DataFrame с колонкой image_name
        image_dir: директория с изображениями
        cache_path: путь для сохранения/загрузки кэша
        extractor: экземпляр ImageFeatureExtractor (если None - создается новый)
        force_extract: принудительно извлечь признаки (игнорировать кэш)
        
    Returns:
        (features, extractor): признаки и экстрактор
    """
    # Проверка наличия кэша
    if os.path.exists(cache_path) and not force_extract:
        print(f"✓ Загрузка признаков из кэша: {cache_path}")
        features = np.load(cache_path)
        print(f"  Размер: {features.shape}")
        
        # Создаем экстрактор если не передан
        if extractor is None:
            extractor = ImageFeatureExtractor()
            
        return features, extractor
    
    # Извлечение признаков
    print(f"Извлечение признаков из {len(df)} изображений...")
    
    # Создаем экстрактор если не передан
    if extractor is None:
        extractor = ImageFeatureExtractor()
    
    # Формирование путей к изображениям
    image_paths = [
        os.path.join(image_dir, str(img_name))
        for img_name in df['image_name'].values
    ]
    
    # Извлечение признаков
    start_time = time.time()
    features = extractor.transform(image_paths)
    elapsed_time = time.time() - start_time
    
    print(f"✓ Признаки извлечены за {elapsed_time:.2f} сек")
    print(f"  Размер: {features.shape}")
    print(f"  Скорость: {len(df) / elapsed_time:.1f} изображений/сек")
    
    # Сохранение в кэш
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, features)
    print(f"✓ Признаки сохранены в кэш: {cache_path}")
    
    return features, extractor


def test_extraction(sample_size: int = 100):
    """
    Тестирование извлечения признаков на небольшой выборке.
    
    Args:
        sample_size: размер выборки для тестирования
    """
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ")
    print("=" * 80)
    
    # Загрузка данных
    train_df = pd.read_parquet('train.parquet')
    
    # Выборка для тестирования
    sample_df = train_df.head(sample_size)
    print(f"\nТестирование на {len(sample_df)} изображениях...")
    
    # Создание экстрактора
    extractor = ImageFeatureExtractor(model_name='resnet50', batch_size=32)
    
    # Формирование путей
    image_paths = [
        os.path.join('train', str(img_name))
        for img_name in sample_df['image_name'].values
    ]
    
    # Проверка существования изображений
    existing_images = sum(1 for path in image_paths if os.path.exists(path))
    print(f"Найдено изображений: {existing_images}/{len(image_paths)}")
    
    # Извлечение признаков
    print("\nИзвлечение признаков...")
    start_time = time.time()
    features = extractor.transform(image_paths)
    elapsed_time = time.time() - start_time
    
    # Статистика
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 80)
    print(f"Размер признаков: {features.shape}")
    print(f"Время обработки: {elapsed_time:.2f} сек")
    print(f"Скорость: {len(sample_df) / elapsed_time:.1f} изображений/сек")
    print(f"Среднее время на изображение: {elapsed_time / len(sample_df) * 1000:.1f} мс")
    
    # Статистика признаков
    print(f"\nСтатистика признаков:")
    print(f"  Среднее: {features.mean():.4f}")
    print(f"  Std: {features.std():.4f}")
    print(f"  Min: {features.min():.4f}")
    print(f"  Max: {features.max():.4f}")
    print(f"  Нулевых значений: {(features == 0).sum()} ({(features == 0).sum() / features.size * 100:.2f}%)")
    
    # Проверка памяти
    memory_mb = features.nbytes / (1024 * 1024)
    print(f"\nИспользование памяти: {memory_mb:.2f} MB")
    
    # Оценка для полного датасета
    full_size = len(train_df)
    estimated_time = elapsed_time * (full_size / len(sample_df))
    estimated_memory = memory_mb * (full_size / len(sample_df))
    
    print(f"\nОценка для полного датасета ({full_size} изображений):")
    print(f"  Время обработки: ~{estimated_time / 60:.1f} минут")
    print(f"  Использование памяти: ~{estimated_memory:.1f} MB")
    
    print("=" * 80)


def main():
    """Основная функция для извлечения признаков."""
    print("=" * 80)
    print("ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ ИЗОБРАЖЕНИЙ")
    print("=" * 80)
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    
    print(f"Train: {len(train_df)} записей")
    print(f"Test: {len(test_df)} записей")
    
    # Создание директории для кэша
    os.makedirs('image_features', exist_ok=True)
    
    # Создание экстрактора
    print("\nИнициализация экстрактора...")
    extractor = ImageFeatureExtractor(model_name='resnet50', batch_size=32)
    
    # Извлечение признаков для train
    print("\n" + "=" * 80)
    print("ОБРАБОТКА TRAIN")
    print("=" * 80)
    train_features, extractor = load_or_extract_features(
        df=train_df,
        image_dir='train',
        cache_path='image_features/train_image_features.npy',
        extractor=extractor,
        force_extract=False
    )
    
    # Извлечение признаков для test
    print("\n" + "=" * 80)
    print("ОБРАБОТКА TEST")
    print("=" * 80)
    test_features, extractor = load_or_extract_features(
        df=test_df,
        image_dir='test',
        cache_path='image_features/test_image_features.npy',
        extractor=extractor,
        force_extract=False
    )
    
    # Сохранение экстрактора
    print("\nСохранение экстрактора...")
    with open('image_feature_extractor.pkl', 'wb') as f:
        pickle.dump(extractor, f)
    print("✓ Экстрактор сохранен в image_feature_extractor.pkl")
    
    # Итоговая статистика
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"Train features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")
    print(f"Размерность признаков: {extractor.get_feature_dim()}")
    
    total_memory = (train_features.nbytes + test_features.nbytes) / (1024 * 1024)
    print(f"Общее использование памяти: {total_memory:.2f} MB")
    
    print("\n✓ Извлечение признаков завершено!")
    print("=" * 80)


if __name__ == '__main__':
    # Запуск тестирования на небольшой выборке
    # test_extraction(sample_size=100)
    
    # Обработка всех изображений:
    main()
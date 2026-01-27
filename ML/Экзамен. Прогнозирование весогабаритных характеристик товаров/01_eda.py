#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исследовательский анализ данных (EDA) для задачи прогнозирования 
весогабаритных характеристик товаров
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройки
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# Константы
DATA_DIR = Path(".")
TRAIN_FILE = DATA_DIR / "train.parquet"
OUTPUT_DIR = DATA_DIR / "eda_plots"
SUMMARY_FILE = DATA_DIR / "eda_summary.txt"

# Целевые переменные
TARGET_COLS = ['real_weight', 'real_height', 'real_width', 'real_length']

# Признаки
FEATURE_COLS = ['item_id', 'order_date', 'item_condition', 'item_price', 
                'category_name', 'subcategory_name', 'microcat_name', 
                'seller_id', 'buyer_id', 'title', 'description', 'image_name']


def create_output_dir():
    """Создание директории для сохранения графиков"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Директория для графиков создана: {OUTPUT_DIR}")


def load_data():
    """Загрузка данных из parquet файла"""
    print("\nЗагрузка данных...")
    
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Файл {TRAIN_FILE} не найден!")
    
    print(f"Загрузка данных из {TRAIN_FILE}...")
    df = pd.read_parquet(TRAIN_FILE)
    print(f"✓ Данные успешно загружены!")
    print(f"  Размер датасета: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    print(f"  Размер в памяти: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def basic_info(df):
    """Базовая информация о датасете"""
    print("\nБазовая информация о датасете:")
    
    info_text = []
    info_text.append("="*80)
    info_text.append("БАЗОВАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    info_text.append("="*80)
    info_text.append(f"\nРазмер датасета: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    info_text.append(f"Размер в памяти: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    
    # Типы данных
    print("\nТипы данных:")
    info_text.append("\nТипы данных:")
    dtype_info = df.dtypes.value_counts()
    for dtype, count in dtype_info.items():
        line = f"  {dtype}: {count} столбцов"
        print(line)
        info_text.append(line)
    
    # Пропущенные значения
    print("\nПропущенные значения:")
    info_text.append("\n\nПропущенные значения:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Столбец': missing.index,
        'Пропусков': missing.values,
        'Процент': missing_pct.values
    })
    missing_df = missing_df[missing_df['Пропусков'] > 0].sort_values('Пропусков', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
        info_text.append(missing_df.to_string(index=False))
    else:
        msg = "  Пропущенных значений не обнаружено!"
        print(msg)
        info_text.append(msg)
    
    # Дубликаты
    duplicates = df.duplicated().sum()
    dup_msg = f"\nДубликаты: {duplicates:,} строк ({duplicates/len(df)*100:.2f}%)"
    print(dup_msg)
    info_text.append(dup_msg)
    
    return info_text


def analyze_targets(df):
    """Анализ целевых переменных"""
    print("\nАнализ целевых переменных:")
    
    info_text = []
    info_text.append("\n" + "="*80)
    info_text.append("АНАЛИЗ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ")
    info_text.append("="*80)
    
    # Статистики
    print("\nОписательные статистики:")
    info_text.append("\nОписательные статистики:")
    stats_df = df[TARGET_COLS].describe()
    print(stats_df.to_string())
    info_text.append(stats_df.to_string())
    
    # Дополнительные статистики
    print("\nДополнительные статистики:")
    info_text.append("\n\nДополнительные статистики:")
    for col in TARGET_COLS:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        line = f"  {col}:"
        print(line)
        info_text.append(line)
        line = f"    Асимметрия (skewness): {skew:.3f}"
        print(line)
        info_text.append(line)
        line = f"    Эксцесс (kurtosis): {kurt:.3f}"
        print(line)
        info_text.append(line)
    
    # Корреляции между таргетами
    print("\nКорреляционная матрица целевых переменных:")
    info_text.append("\n\nКорреляционная матрица целевых переменных:")
    corr_matrix = df[TARGET_COLS].corr()
    print(corr_matrix.to_string())
    info_text.append(corr_matrix.to_string())
    
    # Выбросы (IQR метод)
    print("\nВыбросы (метод IQR):")
    info_text.append("\n\nВыбросы (метод IQR):")
    for col in TARGET_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers_pct = outliers / len(df) * 100
        line = f"  {col}: {outliers:,} выбросов ({outliers_pct:.2f}%)"
        print(line)
        info_text.append(line)
    
    return info_text


def plot_target_distributions(df):
    """Визуализация распределений целевых переменных"""
    print("\nСоздание графиков распределений целевых переменных...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Распределения целевых переменных', fontsize=16, y=1.02)
    
    for idx, col in enumerate(TARGET_COLS):
        # Гистограмма
        ax1 = axes[0, idx]
        ax1.hist(df[col], bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(col)
        ax1.set_ylabel('Частота')
        ax1.set_title(f'Распределение {col}')
        ax1.grid(True, alpha=0.3)
        
        # Гистограмма в лог-шкале
        ax2 = axes[1, idx]
        ax2.hist(np.log1p(df[col]), bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel(f'log1p({col})')
        ax2.set_ylabel('Частота')
        ax2.set_title(f'Распределение log1p({col})')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Сохранено: target_distributions.png")


def plot_target_boxplots(df):
    """Boxplot для выявления выбросов"""
    print("Создание boxplot для целевых переменных...")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Boxplot целевых переменных (выявление выбросов)', fontsize=16)
    
    for idx, col in enumerate(TARGET_COLS):
        ax = axes[idx]
        bp = ax.boxplot(df[col], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'target_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Сохранено: target_boxplots.png")


def plot_correlation_matrix(df):
    """Корреляционная матрица"""
    print("Создание корреляционной матрицы...")
    
    plt.figure(figsize=(10, 8))
    corr_matrix = df[TARGET_COLS].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                mask=mask)
    plt.title('Корреляционная матрица целевых переменных', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Сохранено: correlation_matrix.png")


def analyze_features(df):
    """Анализ признаков"""
    print("\nАнализ признаков:")
    
    info_text = []
    info_text.append("\n" + "="*80)
    info_text.append("АНАЛИЗ ПРИЗНАКОВ")
    info_text.append("="*80)
    
    # Анализ цены
    print("\nАнализ цены (item_price):")
    info_text.append("\n\nАнализ цены (item_price):")
    price_stats = df['item_price'].describe()
    print(price_stats.to_string())
    info_text.append(price_stats.to_string())
    
    # Анализ категорий
    print("\nРаспределение по категориям:")
    info_text.append("\n\nРаспределение по категориям:")
    
    for cat_col in ['category_name', 'subcategory_name', 'microcat_name']:
        n_unique = df[cat_col].nunique()
        line = f"\n  {cat_col}: {n_unique} уникальных значений"
        print(line)
        info_text.append(line)
        
        top_5 = df[cat_col].value_counts().head(5)
        print("    Топ-5 категорий:")
        info_text.append("    Топ-5 категорий:")
        for cat, count in top_5.items():
            line = f"      {cat}: {count:,} ({count/len(df)*100:.2f}%)"
            print(line)
            info_text.append(line)
    
    # Анализ состояния товара
    print("\nРаспределение по состоянию товара (item_condition):")
    info_text.append("\n\nРаспределение по состоянию товара (item_condition):")
    condition_counts = df['item_condition'].value_counts()
    for cond, count in condition_counts.items():
        line = f"  {cond}: {count:,} ({count/len(df)*100:.2f}%)"
        print(line)
        info_text.append(line)
    
    # Анализ текстовых полей
    print("\nАнализ текстовых полей:")
    info_text.append("\n\nАнализ текстовых полей:")
    
    df['title_length'] = df['title'].fillna('').astype(str).str.len()
    df['description_length'] = df['description'].fillna('').astype(str).str.len()
    
    for text_col in ['title_length', 'description_length']:
        stats = df[text_col].describe()
        line = f"\n  {text_col}:"
        print(line)
        info_text.append(line)
        print(f"    Среднее: {stats['mean']:.1f} символов")
        info_text.append(f"    Среднее: {stats['mean']:.1f} символов")
        print(f"    Медиана: {stats['50%']:.1f} символов")
        info_text.append(f"    Медиана: {stats['50%']:.1f} символов")
        print(f"    Макс: {stats['max']:.0f} символов")
        info_text.append(f"    Макс: {stats['max']:.0f} символов")
    
    # Временной анализ
    print("\nВременной анализ (order_date):")
    info_text.append("\n\nВременной анализ (order_date):")
    df['order_date'] = pd.to_datetime(df['order_date'])
    date_range = f"  Период: с {df['order_date'].min()} по {df['order_date'].max()}"
    print(date_range)
    info_text.append(date_range)
    
    df['year'] = df['order_date'].dt.year
    df['month'] = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.dayofweek
    
    print("\n  Распределение по годам:")
    info_text.append("\n  Распределение по годам:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        line = f"    {year}: {count:,} заказов"
        print(line)
        info_text.append(line)
    
    # Уникальные продавцы и покупатели
    print("\nУникальные пользователи:")
    info_text.append("\n\nУникальные пользователи:")
    n_sellers = df['seller_id'].nunique()
    n_buyers = df['buyer_id'].nunique()
    print(f"  Уникальных продавцов: {n_sellers:,}")
    info_text.append(f"  Уникальных продавцов: {n_sellers:,}")
    print(f"  Уникальных покупателей: {n_buyers:,}")
    info_text.append(f"  Уникальных покупателей: {n_buyers:,}")
    
    return info_text


def plot_price_vs_targets(df):
    """Scatter plots: цена vs весогабаритные характеристики"""
    print("Создание scatter plots: цена vs целевые переменные...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Зависимость весогабаритных характеристик от цены', fontsize=16)
    
    axes = axes.flatten()
    
    for idx, target in enumerate(TARGET_COLS):
        ax = axes[idx]
        
        # Используем подвыборку для ускорения отрисовки
        sample_size = min(10000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        ax.scatter(df_sample['item_price'], df_sample[target], 
                  alpha=0.3, s=10, edgecolors='none')
        ax.set_xlabel('Цена товара (₽)')
        ax.set_ylabel(target)
        ax.set_title(f'Цена vs {target}')
        ax.grid(True, alpha=0.3)
        
        # Добавляем линию тренда
        z = np.polyfit(df_sample['item_price'], df_sample[target], 1)
        p = np.poly1d(z)
        ax.plot(df_sample['item_price'].sort_values(), 
               p(df_sample['item_price'].sort_values()), 
               "r--", alpha=0.8, linewidth=2, label='Тренд')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'price_vs_targets.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Сохранено: price_vs_targets.png")


def plot_category_distribution(df):
    """Распределение по категориям"""
    print("Создание графиков распределения по категориям...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Распределение товаров по категориям', fontsize=16)
    
    cat_cols = ['category_name', 'subcategory_name', 'microcat_name']
    
    for idx, cat_col in enumerate(cat_cols):
        ax = axes[idx]
        top_categories = df[cat_col].value_counts().head(15)
        
        top_categories.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Количество товаров')
        ax.set_ylabel('')
        ax.set_title(f'Топ-15 {cat_col}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Добавляем значения на столбцы
        for i, v in enumerate(top_categories.values):
            ax.text(v + max(top_categories.values)*0.01, i, f'{v:,}', 
                   va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Сохранено: category_distribution.png")


def plot_temporal_patterns(df):
    """Временные паттерны"""
    print("Создание графиков временных паттернов...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Временные паттерны в данных', fontsize=16)
    
    # Заказы по месяцам
    ax1 = axes[0, 0]
    monthly_orders = df.groupby(df['order_date'].dt.to_period('M')).size()
    monthly_orders.plot(ax=ax1, marker='o', linewidth=2)
    ax1.set_xlabel('Месяц')
    ax1.set_ylabel('Количество заказов')
    ax1.set_title('Количество заказов по месяцам')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Заказы по дням недели
    ax2 = axes[0, 1]
    day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    dow_orders = df['day_of_week'].value_counts().sort_index()
    dow_orders.index = [day_names[i] for i in dow_orders.index]
    dow_orders.plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
    ax2.set_xlabel('День недели')
    ax2.set_ylabel('Количество заказов')
    ax2.set_title('Распределение заказов по дням недели')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=0)
    
    # Средний вес по месяцам
    ax3 = axes[1, 0]
    monthly_weight = df.groupby(df['order_date'].dt.to_period('M'))['real_weight'].mean()
    monthly_weight.plot(ax=ax3, marker='o', linewidth=2, color='green')
    ax3.set_xlabel('Месяц')
    ax3.set_ylabel('Средний вес (кг)')
    ax3.set_title('Средний вес товаров по месяцам')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Средняя цена по месяцам
    ax4 = axes[1, 1]
    monthly_price = df.groupby(df['order_date'].dt.to_period('M'))['item_price'].mean()
    monthly_price.plot(ax=ax4, marker='o', linewidth=2, color='purple')
    ax4.set_xlabel('Месяц')
    ax4.set_ylabel('Средняя цена (₽)')
    ax4.set_title('Средняя цена товаров по месяцам')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Сохранено: temporal_patterns.png")


def plot_text_length_analysis(df):
    """Анализ длины текстовых полей"""
    print("Создание графиков анализа текстовых полей...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Анализ длины текстовых полей', fontsize=16)
    
    # Распределение длины заголовков
    ax1 = axes[0, 0]
    ax1.hist(df['title_length'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Длина заголовка (символов)')
    ax1.set_ylabel('Частота')
    ax1.set_title('Распределение длины заголовков')
    ax1.grid(True, alpha=0.3)
    
    # Распределение длины описаний
    ax2 = axes[0, 1]
    ax2.hist(df['description_length'], bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Длина описания (символов)')
    ax2.set_ylabel('Частота')
    ax2.set_title('Распределение длины описаний')
    ax2.grid(True, alpha=0.3)
    
    # Связь длины заголовка с весом
    ax3 = axes[1, 0]
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    ax3.scatter(df_sample['title_length'], df_sample['real_weight'], 
               alpha=0.3, s=10, edgecolors='none')
    ax3.set_xlabel('Длина заголовка (символов)')
    ax3.set_ylabel('Вес (кг)')
    ax3.set_title('Длина заголовка vs Вес')
    ax3.grid(True, alpha=0.3)
    
    # Связь длины описания с весом
    ax4 = axes[1, 1]
    ax4.scatter(df_sample['description_length'], df_sample['real_weight'], 
               alpha=0.3, s=10, edgecolors='none', color='coral')
    ax4.set_xlabel('Длина описания (символов)')
    ax4.set_ylabel('Вес (кг)')
    ax4.set_title('Длина описания vs Вес')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'text_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Сохранено: text_length_analysis.png")


def save_summary(info_texts):
    """Сохранение сводки в текстовый файл"""
    print(f"\nСохранение сводки в {SUMMARY_FILE}...")
    
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ (EDA)\n")
        f.write("Задача: Прогнозирование весогабаритных характеристик товаров\n")
        f.write("="*80 + "\n\n")
        
        for info_block in info_texts:
            for line in info_block:
                f.write(line + "\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ВЫВОДЫ И РЕКОМЕНДАЦИИ\n")
        f.write("="*80 + "\n\n")
        f.write("1. Целевые переменные имеют положительную асимметрию - рекомендуется\n")
        f.write("   использовать логарифмическое преобразование (log1p)\n\n")
        f.write("2. Обнаружены выбросы в целевых переменных - необходима обработка\n\n")
        f.write("3. Целевые переменные коррелируют между собой - можно использовать\n")
        f.write("   multi-output модели или учитывать взаимосвязи\n\n")
        f.write("4. Цена товара показывает связь с весогабаритными характеристиками\n\n")
        f.write("5. Категории товаров могут быть важными признаками для предсказания\n\n")
        f.write("6. Текстовые поля (title, description) содержат полезную информацию\n\n")
        f.write("7. Временные паттерны присутствуют - можно добавить временные признаки\n\n")
        f.write("8. Метрика: Macro Log-MAE - оптимизация в логарифмической шкале\n\n")
    
    print(f"✓ Сводка сохранена в {SUMMARY_FILE}")


def main():
    """Основная функция"""
    print("\nИсследовательский анализ данных (EDA)")
    print("Задача: Прогнозирование весогабаритных характеристик товаров")
    
    # Создание директории для графиков
    create_output_dir()
    
    # Загрузка данных
    df = load_data()
    
    # Сбор информации для сводки
    all_info = []
    
    # Базовая информация
    info = basic_info(df)
    all_info.append(info)
    
    # Анализ целевых переменных
    info = analyze_targets(df)
    all_info.append(info)
    
    # Визуализации целевых переменных
    print("\nСоздание визуализаций...")
    plot_target_distributions(df)
    plot_target_boxplots(df)
    plot_correlation_matrix(df)
    
    # Анализ признаков
    info = analyze_features(df)
    all_info.append(info)
    
    # Визуализации признаков
    plot_price_vs_targets(df)
    plot_category_distribution(df)
    plot_temporal_patterns(df)
    plot_text_length_analysis(df)
    
    # Сохранение сводки
    save_summary(all_info)
    
    print("\nEDA завершен успешно!")
    print(f"Графики сохранены в: {OUTPUT_DIR}")
    print(f"Сводка сохранена в: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
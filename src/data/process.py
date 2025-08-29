# src/data/process.py

import os
import pandas as pd
from src.data.load_data import load_solubility_data

def process_solubility_data():
    """Обработка данных: очистка, фичи, статистика."""
    print("📥 Загрузка основного датасета...")
    df = load_solubility_data()

    print(f"📊 Исходный размер: {df.shape}")

    # --- ИЗМЕНЕНИЕ: ВЕСЬ БЛОК ПРО ПЛОТНОСТИ УДАЛЕН ---

    # 1. Удаление дубликатов
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"🗑️  Удалено дубликатов: {initial_count - len(df)}")

    # 2. Фильтрация по диапазону logS
    df = df[(df['log_s'] >= -12) & (df['log_s'] <= 2)].copy()
    print(f"🔍 Отфильтровано по logS (-12, 2): {len(df)} записей")

    # 3. Добавление простых фич
    df['temperature_c'] = df['temperature_k'] - 273.15
    df['is_high_solubility'] = (df['log_s'] > -4).astype(int)
    df['solvent_category'] = df['solvent'].apply(
        lambda x: 'water' if x == 'O' else
        'alcohol' if x in ['CCO', 'CC(C)O', 'CCCO'] else 'other'
    )

    # 4. Сохранение
    os.makedirs("data/processed", exist_ok=True)
    final_file_path = "data/processed/solubility_clean.csv" # Возвращаем старое имя файла
    df.to_csv(final_file_path, index=False)

    # 5. Статистика (убрали упоминания плотности)
    with open("data/processed/data_stats.txt", "w", encoding="utf-8") as f:
        f.write(f"Количество соединений: {df['compound_name'].nunique()}\n")
        f.write(f"Количество растворителей: {df['solvent'].nunique()}\n")
        f.write(f"Средний logS: {df['log_s'].mean():.2f}\n")
        f.write(f"Медианный logS: {df['log_s'].median():.2f}\n")
        f.write(f"Стандартное отклонение logS: {df['log_s'].std():.2f}\n")

    print(f"Обработано {len(df)} записей. Данные сохранены в {final_file_path}")
    return df
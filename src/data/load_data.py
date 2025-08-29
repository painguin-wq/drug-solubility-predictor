# src/data/load_data.py
import pandas as pd
import os


def load_solubility_data():
    """Загрузка BigSolDBv2.0.csv с явным указанием типов"""
    file_path = "data/raw/BigSolDBv2.0.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Файл не найден: {file_path}\n"
            "Убедитесь, что:\n"
            "1. Файл BigSolDBv2.0.csv загружен в папку data/raw/\n"
            "2. Название файла точное (с учётом регистра)\n"
            "3. Вы запускаете скрипт из корня проекта"
        )

    # Проверим, пустой ли файл
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Файл {file_path} пуст")

    # Указываем типы для критичных столбцов, остальные — object
    dtypes = {
        'smiles': 'object',
        'temperature_k': 'float64',
        'solvent': 'object',
        'solvent_smiles': 'object',
        'solubility_mol_l': 'float64',
        'solubility_mol_kg': 'float64',
        'log_s': 'float64',
        'compound_name': 'object',
        'cas': 'object',
        'pubchem_cid': 'object',
        'is_organic': 'object',
        'doi': 'object'
    }

    # Указываем low_memory=False, чтобы избежать предупреждения
    df = pd.read_csv(file_path, header=None, low_memory=False)

    # Назначаем колонки
    column_names = list(dtypes.keys())
    df = df.iloc[:, :len(column_names)]  # Ограничиваем число столбцов
    df.columns = column_names

    # Преобразуем числовые колонки
    df['temperature_k'] = pd.to_numeric(df['temperature_k'], errors='coerce')
    df['log_s'] = pd.to_numeric(df['log_s'], errors='coerce')
    df['solubility_mol_l'] = pd.to_numeric(df['solubility_mol_l'], errors='coerce')

    # Удаляем строки с NaN в критичных полях
    df = df.dropna(subset=['smiles', 'log_s'])

    print(f"✅ Загружено {len(df)} записей о растворимости")
    return df
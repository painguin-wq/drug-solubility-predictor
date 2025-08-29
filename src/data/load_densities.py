# src/data/load_densities.py
import pandas as pd
import os


def load_and_process_densities():
    """Загружает и обрабатывает данные о плотностях из BigSolDBv2.0_densities.csv."""
    file_path = "data/raw/BigSolDBv2.0_densities.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл плотностей не найден: {file_path}")

    # Прочитаем файл как CSV. Разделитель - запятая. Заголовок есть.
    # Учитываем, что в конце файла может не быть перевода строки.
    try:
        # Читаем в DataFrame
        df_raw = pd.read_csv(file_path)
        print(f"✅ Файл плотностей успешно прочитан. Строк: {len(df_raw)}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Файл плотностей {file_path} пуст или поврежден.")
    except Exception as e:
        raise ValueError(f"Ошибка при чтении файла плотностей {file_path}: {e}")

    # Проверим, что у нас есть ожидаемые столбцы
    expected_columns = ['Solvent', 'Temperature_K', 'Density_g/cm^3', 'Source']
    if list(df_raw.columns) != expected_columns:
        # Если заголовок не распознан, попробуем прочитать без заголовка и назначить вручную
        try:
            df_raw = pd.read_csv(file_path, header=None)
            if df_raw.shape[1] == 4:
                df_raw.columns = expected_columns
                # Удаляем потенциально первую строку с заголовком, если она стала частью данных
                if df_raw.iloc[0, 0] == 'Solvent':
                    df_raw = df_raw.drop(df_raw.index[0]).reset_index(drop=True)
            else:
                raise ValueError(f"Некорректное количество столбцов: ожидается 4, получено {df_raw.shape[1]}")
        except Exception as e:
            raise ValueError(f"Не удалось обработать структуру файла плотностей: {e}")

    # Переименуем столбцы для удобства
    df_densities = df_raw.rename(columns={
        'Solvent': 'solvent_name',
        'Temperature_K': 'temperature_k',
        'Density_g/cm^3': 'density_g_ml',  # Предполагаем, что г/см³ = г/мл
        'Source': 'doi'
    })

    # Преобразуем типы данных
    try:
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---

        # Температуру преобразуем в число, пропуски заменяем на NaN
        df_densities['temperature_k'] = pd.to_numeric(df_densities['temperature_k'], errors='coerce')

        # В столбце с плотностью сначала заменяем запятые на точки
        df_densities['density_g_ml'] = df_densities['density_g_ml'].astype(str).str.replace(',', '.', regex=False)
        # Теперь безопасно преобразуем в число, пропуски заменяем на NaN
        df_densities['density_g_ml'] = pd.to_numeric(df_densities['density_g_ml'], errors='coerce')

        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    except Exception as e:
        raise ValueError(f"Ошибка при преобразовании числовых данных: {e}")

    # Создадим столбец solvent_smiles для совместимости с основным датасетом
    # Это упрощенное сопоставление, в реальности может потребоваться более сложная логика
    smiles_map = {
        "ethanol": "CCO",
        "water": "O",
        "methanol": "CO",
        "acetonitrile": "CC#N",
        "dmso": "CS(C)=O",
        "thf": "C1CCOC1",
        "toluene": "Cc1ccccc1",
        "n-propanol": "CCCO",
        "isopropanol": "CC(C)O",
        "n-heptanol": "CCCCCCCO",
        "transcutol": "CCOCCOCCO",
        "cyclohexane": "C1CCCCC1",
        "chloroform": "ClC(Cl)Cl",
        "nmp": "CN1CCCC1=O",
        "ethyl acetate": "CCOC(C)=O",
        "n-hexadecane": "CCCCCCCCCCCCCCCC",
        "2-methoxyethanol": "COCCO",
        "2-ethoxyethanol": "CCOCCO"
        # Добавьте другие растворители по мере необходимости
    }
    # Приведем названия к нижнему регистру для сопоставления
    df_densities['solvent_smiles'] = df_densities['solvent_name'].str.lower().map(smiles_map)
    # Если не нашли SMILES, оставим оригинальное название
    df_densities['solvent_smiles'] = df_densities['solvent_smiles'].fillna(df_densities['solvent_name'])

    # Сохраняем обработанные данные
    os.makedirs("data/processed", exist_ok=True)
    df_densities.to_csv("data/processed/solvent_densities.csv", index=False)
    print(f"✅ Обработано и сохранено {len(df_densities)} записей о плотностях растворителей.")
    return df_densities


# Пример использования
if __name__ == "__main__":
    try:
        df_densities = load_and_process_densities()
        print("\nПервые несколько строк:")
        print(df_densities.head())
        print("\nТипы данных:")
        print(df_densities.dtypes)
        print(f"\nУникальные растворители: {df_densities['solvent_name'].unique()}")
    except Exception as e:
        print(f"Ошибка: {e}")
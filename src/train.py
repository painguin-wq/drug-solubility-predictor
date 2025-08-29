# src/train.py

from src.data.process import process_solubility_data
from src.features.smiles_featurizer import featurize_compounds
# --- ИЗМЕНЕНИЕ 1: Импортируем НОВУЮ функцию ---
from src.models.solubility_model import train_and_evaluate_models

def main():
    """
    Полный пайплайн для подготовки данных, обучения и сравнения моделей.
    """
    print("🚀 Этап 1: Обучение и сравнение моделей...")

    # 1. Загрузка и обработка основного датасета
    print("\n--- Шаг 1: Обработка основного датасета ---")
    df = process_solubility_data()

    # 2. Генерация молекулярных признаков
    print("\n--- Шаг 2: Генерация молекулярных признаков ---")
    df = featurize_compounds(df)
    df.to_csv("data/processed/solubility_with_features.csv", index=False)
    print("💾 Данные с фичами сохранены.")

    # 3. Обучение, оценка и сохранение лучшей модели
    print("\n--- Шаг 3: Обучение и сравнение моделей ---")
    # --- ИЗМЕНЕНИЕ 2: Вызываем НОВУЮ функцию ---
    train_and_evaluate_models(df)

    print("\n✅ Этап обучения и сравнения завершён. Лучшая модель сохранена.")


if __name__ == "__main__":
    main()
# src/main.py
from src.data.process import process_solubility_data  # Обновленная версия
from src.data.load_densities import load_and_process_densities  # Новый импорт
from src.features.smiles_featurizer import featurize_compounds
from src.visualization.plots import create_eda_report  # Убедитесь, что он может работать с новыми данными
from src.models.solubility_model import train_solubility_model, predict_optimal_conditions  # Обновленные версии


def main():
    print("🚀 Запуск проекта: ML-прогнозирование растворимости (с плотностью)")

    # 1. Загрузка и обработка плотностей (новый шаг)
    load_and_process_densities()  # Сохранит в data/processed/solvent_densities.csv

    # 2. Загрузка, обработка и интеграция плотностей с основными данными
    df = process_solubility_data()  # Использует обновленную функцию

    # 3. Фичи
    df = featurize_compounds(df)
    df.to_csv("data/processed/solubility_with_features_and_density.csv", index=False)
    print("💾 Данные с фичами и плотностью сохранены.")

    # 4. Визуализация (убедитесь, что plots.py может работать с новыми данными)
    # create_eda_report() # Может потребоваться обновление для отображения плотности

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---

    # 5. Обучение модели (обновленная версия)
    train_solubility_model(df)  # <-- ИЗМЕНЕНИЕ: Передаем df в функцию

    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    # 6. Пример оптимизации
    print("\n🔍 Пример оптимизации для Iohexol (с плотностью):")
    iohexol_smiles = "CC(=O)N(CC(O)CO)c1c(I)c(C(=O)NCC(O)CO)c(I)c(C(=O)NCC(O)CO)c1I"
    result = predict_optimal_conditions(iohexol_smiles)  # Использует обновленную функцию
    if isinstance(result, dict):
        print(f"Лучший растворитель (SMILES): {result['best_solvent_smiles']}")
        print(f"Оптимальная T: {result['best_temperature_c']:.1f} °C")
        print(f"Прогноз logS: {result['predicted_logS']:.3f}")
    else:
        print(result)  # Сообщение об ошибке

    print("\n✅ Проект завершён. Смотрите папки reports/ и models/")


if __name__ == "__main__":
    main()
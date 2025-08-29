# src/predict.py

from src.models.solubility_model import predict_optimal_conditions

def main():
    """
    Загружает обученную модель и делает предсказание для тестового SMILES.
    """
    print("🚀 Этап 2: Предсказание на основе обученной модели...")

    test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Кофеин

    print(f"\nАнализ соединения со SMILES: {test_smiles}")

    result = predict_optimal_conditions(test_smiles)

    if isinstance(result, dict):
        print("\n--- Результаты оптимизации ---")
        print(f"SMILES: {result['smiles']}")
        print(
            f"Оптимальные условия: Растворитель {result['best_solvent_smiles']} при температуре {result['best_temperature_c']:.1f} °C"
        )
        print(f"Прогнозируемый logS: {result['predicted_logS']:.3f}")
        print(f"Прогнозируемая растворимость: {result['predicted_solubility_mol_per_l']:.6f} моль/л")

        print("\n--- Топ-5 рекомендуемых условий ---")
        for i, cond in enumerate(result['top_5_conditions']):
            # --- ИЗМЕНЕНИЕ: Убираем вывод плотности ---
            print(
                f"  {i + 1}. Растворитель: {cond['solvent_smiles']:<10} "
                f"| T: {cond['temp_c']:>5.1f} °C "
                f"| logS: {cond['log_s']:.3f}"
            )
    else:
        print(f"\n❌ Ошибка: {result}")

    print("\n✅ Этап предсказания завершён.")


if __name__ == "__main__":
    main()
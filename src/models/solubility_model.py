# src/models/solubility_model.py
import joblib
import pandas as pd
import os
import time

# --- 1. Импортируем все регрессоры, которые хотим протестировать ---
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def train_and_evaluate_models(df):
    """
    Обучает большой набор регрессоров, сравнивает их и сохраняет лучшую модель.
    """
    print("📥 Подготовка данных для обучения...")

    # Разделяем признаки на числовые и категориальные для правильной обработки
    categorical_features = ['solvent']
    numerical_features = ['temperature_k', 'mol_weight', 'logp', 'tpsa', 'h_donors', 'h_acceptors']

    X = df[categorical_features + numerical_features].dropna()
    y = df.loc[X.index]['log_s']

    print(f"🧮 Используемые признаки: {categorical_features + numerical_features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 2. Определяем общий препроцессор с масштабированием! ---
    # OneHotEncoder для категорий и StandardScaler для чисел.
    # Это КРИТИЧЕСКИ ВАЖНО для линейных моделей, SVR и KNN.
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('scaler', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'  # На случай, если появятся другие столбцы
    )

    # --- 3. Определяем большой словарь моделей для тестирования ---
    models = {
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "SVR": SVR(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    }

    results = []
    best_mae = float('inf')
    best_model_pipeline = None
    best_model_name = ""

    # --- 4. Обучаем и оцениваем каждую модель в цикле ---
    for name, model in models.items():
        print(f"\n--- Обучение модели: {name} ---")
        start_time = time.time()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        end_time = time.time()
        training_time = end_time - start_time

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Модель": name,
            "MAE": mae,
            "R²": r2,
            "Время (сек)": training_time
        })

        print(f"✅ Результаты для {name}: MAE = {mae:.3f}, R² = {r2:.3f}, Время = {training_time:.2f} сек")

        if mae < best_mae:
            best_mae = mae
            best_model_pipeline = pipeline
            best_model_name = name

    # --- 5. Выводим красивую итоговую таблицу ---
    results_df = pd.DataFrame(results).sort_values(by="MAE").reset_index(drop=True)
    print("\n\n--- 📊 Сводная таблица результатов ---")
    print(results_df.to_string())

    # --- 6. Сохраняем только лучшую модель ---
    print(f"\n🏆 Лучшая модель по метрике MAE: {best_model_name} (MAE = {best_mae:.3f})")

    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    joblib.dump(best_model_pipeline, model_path)
    print(f"💾 Лучшая модель сохранена в: {model_path}")

    return best_model_pipeline


def predict_optimal_conditions(smiles, temp_range=(273, 350), solvents=None):
    """
    Эта функция остается БЕЗ ИЗМЕНЕНИЙ. Она просто загружает
    лучшую модель, какой бы она ни была.
    """
    if solvents is None:
        solvents = ["O", "CCO", "CC(C)O", "C1CCOC1", "CS(C)=O"]

    model_path = "models/best_model.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        return f"❌ Файл лучшей модели '{model_path}' не найден. Сначала запустите обучение."
    except Exception as e:
        return f"❌ Ошибка загрузки модели: {e}"

    from src.features.smiles_featurizer import calculate_molecular_features
    features = calculate_molecular_features(smiles)

    if pd.isna(features[0]) or any(pd.isna(f) for f in features):
        return "❌ Не удалось распарсить SMILES или рассчитать дескрипторы."

    mol_weight, logp, tpsa, h_donors, h_acceptors, _ = features
    print(f"✅ Дескрипторы рассчитаны.")

    best_solvent = None
    best_temp = 0
    max_solubility = -float('inf')
    results = []

    print("🔍 Поиск оптимальных условий...")
    for solvent_smiles in solvents:
        for temp in range(temp_range[0], temp_range[1] + 1, 10):
            X = pd.DataFrame([{
                'temperature_k': temp,
                'solvent': solvent_smiles,
                'mol_weight': mol_weight,
                'logp': logp,
                'tpsa': tpsa,
                'h_donors': h_donors,
                'h_acceptors': h_acceptors
            }])
            try:
                pred = model.predict(X)[0]
                results.append({
                    'solvent_smiles': solvent_smiles,
                    'temp_k': temp,
                    'temp_c': temp - 273.15,
                    'log_s': pred
                })

                if pred > max_solubility:
                    max_solubility = pred
                    best_solvent = solvent_smiles
                    best_temp = temp
            except Exception as e:
                print(f"Ошибка предсказания для {solvent_smiles} при {temp}K: {e}")
                continue

    if best_solvent is None:
        return "❌ Не удалось определить оптимальные условия."

    result = {
        'smiles': smiles,
        'best_solvent_smiles': best_solvent,
        'best_temperature_k': best_temp,
        'best_temperature_c': best_temp - 273.15,
        'predicted_logS': max_solubility,
        'predicted_solubility_mol_per_l': 10 ** max_solubility,
    }

    results.sort(key=lambda x: x['log_s'], reverse=True)
    result['top_5_conditions'] = results[:5]

    print("✅ Оптимизация завершена")
    return result
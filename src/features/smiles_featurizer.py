# src/features/smiles_featurizer.py
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors  # <-- Используем это
import pandas as pd
import numpy as np


def calculate_molecular_features(smiles):
    """
    Вычисление молекулярных дескрипторов.
    Возвращает список признаков или [np.nan] * 6 в случае ошибки.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * 6

        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Используем новые функции для H-связей
        h_donors = rdMolDescriptors.CalcNumHBD(mol)  # <-- Исправлено
        h_acceptors = rdMolDescriptors.CalcNumHBA(mol)  # <-- Исправлено

        # Правило Липински: пересчет вручную
        rule_of_five = int(
            (mol_weight <= 500) +
            (logp <= 5) +
            (h_donors <= 5) +
            (h_acceptors <= 10)
            >= 3)

        return [mol_weight, logp, tpsa, h_donors, h_acceptors, rule_of_five]

    except Exception as e:
        print(f"Ошибка при обработке SMILES: {smiles}, {e}")
        return [np.nan] * 6


def featurize_compounds(df):
    """Добавление молекулярных признаков к DataFrame."""
    print("🔬 Генерация молекулярных дескрипторов...")

    # Применяем функцию к каждому SMILES
    features = df['smiles'].apply(calculate_molecular_features)

    # Создаем DataFrame из признаков
    feature_df = pd.DataFrame(
        features.tolist(),
        columns=[
            'mol_weight',
            'logp',
            'tpsa',
            'h_donors',
            'h_acceptors',
            'rule_of_five'
        ]
    )

    # Конкатенируем с оригинальным DataFrame
    df_result = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

    print(f"✅ Добавлено {len(feature_df.columns)} молекулярных признаков")
    return df_result
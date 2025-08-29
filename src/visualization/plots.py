
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_eda_report():
    df = pd.read_csv("data/processed/solubility_clean.csv")
    os.makedirs("reports/figures", exist_ok=True)

    # 1. Распределение logS
    plt.figure(figsize=(10, 6))
    sns.histplot(df['log_s'], bins=50, kde=True, color='teal')
    plt.title('Распределение растворимости (logS)')
    plt.xlabel('logS (моль/л)')
    plt.ylabel('Частота')
    plt.axvline(df['log_s'].mean(), color='red', linestyle='--', label=f'Среднее: {df["log_s"].mean():.2f}')
    plt.legend()
    plt.savefig("reports/figures/logS_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Растворимость по растворителям
    plt.figure(figsize=(12, 6))
    top_solvents = df['solvent'].value_counts().head(10).index
    df_top = df[df['solvent'].isin(top_solvents)]
    sns.boxplot(data=df_top, x='solvent', y='log_s')
    plt.title('Растворимость по топ-10 растворителям')
    plt.xlabel('Растворитель')
    plt.ylabel('logS')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/figures/solubility_by_solvent.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Температурная зависимость
    plt.figure(figsize=(10, 6))
    avg_by_temp = df.groupby('temperature_k')['log_s'].mean().reset_index()
    plt.plot(avg_by_temp['temperature_k'], avg_by_temp['log_s'], 'o-', linewidth=2)
    plt.title('Средняя растворимость vs Температура')
    plt.xlabel('Температура (K)')
    plt.ylabel('Средний logS')
    plt.grid(alpha=0.3)
    plt.savefig("reports/figures/solubility_vs_temp.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Визуализации сохранены")

def create_eda_report_with_density():
    """
    Создает визуализации на основе обработанных данных,
    включая плотность растворителей.
    """
    # Загрузка обработанных данных с плотностями
    df = pd.read_csv("data/processed/solubility_with_density.csv")
    os.makedirs("reports/figures", exist_ok=True)

    # Настройка стиля
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.1)

    print("📊 Создание визуализаций...")

    # --- 1. Распределение плотностей растворителей ---
    plt.figure(figsize=(10, 6))
    # Группировка по растворителю и взятие медианной плотности для уникальности
    density_per_solvent = df.groupby('solvent_smiles')['density_g_ml'].median().sort_values()

    bars = plt.bar(range(len(density_per_solvent)), density_per_solvent.values, color='skyblue')
    plt.xlabel('Растворитель (SMILES)')
    plt.ylabel('Плотность (г/мл)')
    plt.title('Медианная плотность различных растворителей')
    plt.xticks(range(len(density_per_solvent)), density_per_solvent.index, rotation=45, ha='right')

    # Добавление значений на столбцы для лучшей читаемости
    for bar, value in zip(bars, density_per_solvent.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("reports/figures/density_per_solvent.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  - График плотности по растворителям сохранен.")

    # --- 2. Зависимость logS от плотности ---
    plt.figure(figsize=(10, 6))
    # Используем прозрачность (alpha) и маленькие точки для большого датасета
    sns.scatterplot(data=df, x='density_g_ml', y='log_s', alpha=0.5, s=10, color='purple')

    # Добавим линию тренда (LOWESS или линейная регрессия)
    # LOWESS может быть тяжелым, используем простую линейную регрессию
    sns.regplot(data=df, x='density_g_ml', y='log_s', scatter=False, color='red', line_kws={'linewidth': 1})

    plt.xlabel('Плотность растворителя (г/мл)')
    plt.ylabel('logS (моль/л)')
    plt.title('Зависимость растворимости (logS) от плотности растворителя')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/solubility_vs_density.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  - График logS vs плотность сохранен.")

    # --- 3. Boxplot logS по категориям плотности ---
    # Создадим категории плотности для лучшего сравнения
    df['density_category'] = pd.cut(df['density_g_ml'], bins=5,
                                    labels=['Очень низкая', 'Низкая', 'Средняя', 'Высокая', 'Очень высокая'])

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='density_category', y='log_s', palette='viridis')
    plt.xlabel('Категория плотности растворителя')
    plt.ylabel('logS (моль/л)')
    plt.title('Распределение растворимости в группах по плотности')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("reports/figures/solubility_by_density_category.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  - График logS по категориям плотности сохранен.")

    # --- 4. Scatter logS vs Density, раскрашенный по типу растворителя ---
    # Выберем топ-N наиболее частых растворителей для читаемости легенды
    top_solvents = df['solvent'].value_counts().head(6).index
    df_top_solvents = df[df['solvent'].isin(top_solvents)]

    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df_top_solvents,
        x='density_g_ml',
        y='log_s',
        hue='solvent',
        alpha=0.6,
        s=20,
        palette='tab10'
    )
    plt.xlabel('Плотность растворителя (г/мл)')
    plt.ylabel('logS (моль/л)')
    plt.title('Растворимость vs Плотность (цвет = тип растворителя)')
    plt.legend(title='Растворитель', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/solubility_vs_density_by_solvent.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  - График logS vs плотность по типам растворителей сохранен.")

    print("Все новые визуализации сохранены в reports/figures/")


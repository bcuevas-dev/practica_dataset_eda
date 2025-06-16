import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

sns.set(style="whitegrid")

def cargar_datos():
    path = 'data/processed/Dataset_clear.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra el archivo {path}")
    return pd.read_csv(path)

# ==================================================
# FRECUENCIAS DESCRIPTIVAS 
# ==================================================
def estadisticas_descriptivas(df):
    print("\n========== FRECUENCIAS DESCRIPTIVAS ==========")

    # Estado civil
    print("\nEstado Civil (%):")
    estado_civil = df['EstadoCivil_Ordinal'].value_counts(normalize=True).sort_index() * 100
    for index, value in estado_civil.items():
        descripcion = {
            1: "Soltero(a)",
            2: "Casado(a)",
            3: "Unión Libre",
            4: "Divorciado(a)",
            5: "Viudo(a)"
        }.get(index, "Otro")
        print(f"{descripcion}: {value:.2f}%")

    # Empleabilidad
    print("\nEmpleabilidad (%):")
    empleo = df['Trabaja_Ordinal'].value_counts(normalize=True).sort_index() * 100
    for index, value in empleo.items():
        descripcion = {
            1: "Empleado(a)",
            0: "Desempleado(a)"
        }.get(index, "Otro")
        print(f"{descripcion}: {value:.2f}%")

    # Desempeño Académico (Grado4Scala)
    print("\nDesempeño Académico (%):")
    desempeno = df['Grado4Scala_Ordinal'].value_counts(normalize=True).sort_index() * 100
    for index, value in desempeno.items():
        descripcion = {
            1: "F (Reprobado)",
            2: "D (Deficiente)",
            3: "C (Regular)",
            4: "B (Bueno)",
            5: "A (Excelente)"
        }.get(index, "Otro")
        print(f"{descripcion}: {value:.2f}%")

    # Cálculo de no aprobados (F, D, C)
    no_aprobados = desempeno.get(1, 0) + desempeno.get(2, 0) + desempeno.get(3, 0)
    print(f"\nPorcentaje de NO aprobados (F+D+C): {no_aprobados:.2f}%")

# ===========================================
# ESTADISTICA INFERENCIAL
# ===========================================
def estadisticas_inferenciales(df):
    print("\n========== ESTADISTICA INFERENCIAL ==========")

    # ANOVA
    df_anova = df[['Ingresos_Ordinal', 'Grado4Scala_Ordinal']].dropna()
    if not df_anova.empty:
        resultado = stats.f_oneway(df_anova['Ingresos_Ordinal'], df_anova['Grado4Scala_Ordinal'])
        print(f"\nANOVA: F={resultado.statistic:.4f}, p={resultado.pvalue:.4f}")

    # T-Test
    df_ttest = df[['Trabaja_Ordinal', 'Ingresos_Ordinal']].dropna()
    trabaja_si = df_ttest[df_ttest['Trabaja_Ordinal']==1]['Ingresos_Ordinal']
    trabaja_no = df_ttest[df_ttest['Trabaja_Ordinal']==0]['Ingresos_Ordinal']
    if not trabaja_si.empty and not trabaja_no.empty:
        resultado = stats.ttest_ind(trabaja_si, trabaja_no, equal_var=False)
        print(f"\nT-Test: t={resultado.statistic:.4f}, p={resultado.pvalue:.4f}")

    # Pearson
    df_pearson = df[['Edad_Ordinal', 'Ingresos_Ordinal']].dropna()
    if not df_pearson.empty:
        corr, p_value = stats.pearsonr(df_pearson['Edad_Ordinal'], df_pearson['Ingresos_Ordinal'])
        print(f"\nPearson: r={corr:.4f}, p={p_value:.4f}")

    # Spearman
    if not df_pearson.empty:
        corr, p_value = stats.spearmanr(df_pearson['Edad_Ordinal'], df_pearson['Ingresos_Ordinal'])
        print(f"\nSpearman: r={corr:.4f}, p={p_value:.4f}")

    # Chi-Cuadrado
    df_chi = df[['Genero_Ordinal', 'EstadoCivil_Ordinal']].dropna()
    if not df_chi.empty:
        tabla = pd.crosstab(df_chi['Genero_Ordinal'], df_chi['EstadoCivil_Ordinal'])
        if not tabla.empty:
            chi2, p, dof, expected = stats.chi2_contingency(tabla)
            print(f"\nChi-Cuadrado: chi2={chi2:.4f}, p={p:.4f}, gl={dof}")

    # Regresión Lineal
    df_reg = df[['Edad_Ordinal', 'Trabaja_Ordinal', 'Genero_Ordinal', 'Region_Ordinal', 'Ingresos_Ordinal']].dropna()
    if not df_reg.empty:
        X = df_reg[['Edad_Ordinal', 'Trabaja_Ordinal', 'Genero_Ordinal', 'Region_Ordinal']]
        y = df_reg['Ingresos_Ordinal']
        X = sm.add_constant(X)
        modelo = sm.OLS(y, X).fit()
        print("\nResumen regresión:")
        print(modelo.summary())

    # Kurtosis
    print("\n====== KURTOSIS ======")
    numeric_df = df.select_dtypes(include=['number'])
    for col in numeric_df.columns:
        k = kurtosis(df[col].dropna())
        print(f"{col}: Kurtosis = {k:.2f}")

# ===========================================
# GRAFICOS EXPLORATORIOS
# ===========================================
def graficos_eda(df):
    print("\n================ GRAFICOS EXPLORATORIOS ================")

    plt.figure(figsize=(8,5))
    sns.countplot(x='Grado4Scala_Ordinal', data=df)
    plt.title("Distribución de Desempeño Académico")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.boxplot(x='Grado4Scala_Ordinal', y='Ingresos_Ordinal', data=df)
    plt.title("Ingresos vs Desempeño Académico")
    plt.show()

    plt.figure(figsize=(10,8))
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Mapa de Correlaciones")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Edad_Ordinal', y='Ingresos_Ordinal', hue='Grado4Scala_Ordinal', data=df)
    plt.title("Edad vs Ingresos")
    plt.show()

# ===========================================
# MAIN
# ===========================================
def main():
    df = cargar_datos()
    estadisticas_descriptivas(df)
    estadisticas_inferenciales(df)
    graficos_eda(df)

if __name__ == '__main__':
    main()

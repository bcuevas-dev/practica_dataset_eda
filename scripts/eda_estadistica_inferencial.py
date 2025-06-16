# ==========================================================================================
# Proyecto: Análisis Exploratorio de Datos (EDA) - Desempeño Académico
# Versión Local para scripts
# ==========================================================================================

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import os

sns.set(style="whitegrid")

# ============================================
# CARGA LOCAL DE DATOS 
# ============================================
def cargar_datos():
    path = 'data/processed/Dataset_clear.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra el archivo {path}")
    return pd.read_csv(path)

# ============================================
# ESTADÍSTICA DESCRIPTIVA ENRIQUECIDA
# ============================================
def estadisticas_descriptivas(df):
    print("\n========== FRECUENCIAS DESCRIPTIVAS ==========")

    estado_civil = df['EstadoCivil_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nEstado Civil (%):")
    for index, value in estado_civil.items():
        descripcion = {
            1: "Soltero(a)", 2: "Casado(a)", 3: "Unión Libre",
            4: "Divorciado(a)", 5: "Viudo(a)"
        }.get(index, "Otro")
        print(f"{descripcion}: {value:.2f}%")

    genero = df['Genero_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nGénero (%):")
    for index, value in genero.items():
        descripcion = {1: "Femenino", 2: "Masculino"}.get(index, "Otro")
        print(f"{descripcion}: {value:.2f}%")

    empleo = df['Trabaja_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nEmpleabilidad (%):")
    for index, value in empleo.items():
        descripcion = {1: "Empleado(a)", 0: "Desempleado(a)"}.get(index, "Otro")
        print(f"{descripcion}: {value:.2f}%")

    desempeno = df['Grado4Scala_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nDesempeño Académico (%):")
    for index, value in desempeno.items():
        descripcion = {
            1: "F (Reprobado)", 2: "D (Deficiente)", 3: "C (Regular)",
            4: "B (Bueno)", 5: "A (Excelente)"
        }.get(index, "Otro")
        print(f"{descripcion}: {value:.2f}%")
    no_aprobados = desempeno.get(1, 0) + desempeno.get(2, 0) + desempeno.get(3, 0)
    print(f"\nPorcentaje de NO aprobados (F+D+C): {no_aprobados:.2f}%")

    region = df['Region_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nRegión (%):")
    for index, value in region.items():
        descripcion = {
            1: "Distrito Nacional", 2: "Norte", 3: "Sur", 4: "Este", 5: "Zona Fronteriza"
        }.get(index, f"Región {index}")
        print(f"{descripcion}: {value:.2f}%")

    zona_geo = df['ZonaGeo_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nZona Geográfica (%):")
    for index, value in zona_geo.items():
        descripcion = {1: "Urbana", 2: "Semiurbana", 3: "Rural"}.get(index, f"Zona {index}")
        print(f"{descripcion}: {value:.2f}%")

    marca = df['MarcaMercado_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nMarca de Mercado (%):")
    for index, value in marca.items():
        descripcion = {
            1: "Alta", 2: "Media-Alta", 3: "Media", 4: "Media-Baja", 5: "Baja"
        }.get(index, f"Marca {index}")
        print(f"{descripcion}: {value:.2f}%")

    complejidad = df['ComplejidadArea_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nComplejidad del Área (%):")
    for index, value in complejidad.items():
        descripcion = {
            1: "Muy Baja", 2: "Baja", 3: "Media", 4: "Alta", 5: "Muy Alta"
        }.get(index, f"Nivel {index}")
        print(f"{descripcion}: {value:.2f}%")

    modelo = df['ModeloEducativo_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nModelo Educativo (%):")
    for index, value in modelo.items():
        descripcion = {1: "Presencial", 2: "Semipresencial", 3: "Virtual"}.get(index, f"Modelo {index}")
        print(f"{descripcion}: {value:.2f}%")

    ingresos = df['Ingresos_Ordinal'].value_counts(normalize=True).sort_index() * 100
    print("\nIngresos (%):")
    for index, value in ingresos.items():
        descripcion = {
            1: "Muy Bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy Alto"
        }.get(index, f"Ingreso {index}")
        print(f"{descripcion}: {value:.2f}%")

# ============================================
# ESTADÍSTICA INFERENCIAL
# ============================================
def estadisticas_inferenciales(df):
    print("\n========== ESTADISTICA INFERENCIAL ==========")

    df_anova = df[['Ingresos_Ordinal', 'Grado4Scala_Ordinal']].dropna()
    resultado = stats.f_oneway(df_anova['Ingresos_Ordinal'], df_anova['Grado4Scala_Ordinal'])
    print(f"\nANOVA: F={resultado.statistic:.4f}, p={resultado.pvalue:.4f}")

    df_ttest = df[['Trabaja_Ordinal', 'Ingresos_Ordinal']].dropna()
    trabaja_si = df_ttest[df_ttest['Trabaja_Ordinal'] == 1]['Ingresos_Ordinal']
    trabaja_no = df_ttest[df_ttest['Trabaja_Ordinal'] == 0]['Ingresos_Ordinal']
    resultado = stats.ttest_ind(trabaja_si, trabaja_no, equal_var=False)
    print(f"\nT-Test: t={resultado.statistic:.4f}, p={resultado.pvalue:.4f}")

    df_pearson = df[['Edad_Ordinal', 'Ingresos_Ordinal']].dropna()
    corr, p_value = stats.pearsonr(df_pearson['Edad_Ordinal'], df_pearson['Ingresos_Ordinal'])
    print(f"\nPearson: r={corr:.4f}, p={p_value:.4f}")

    corr, p_value = stats.spearmanr(df_pearson['Edad_Ordinal'], df_pearson['Ingresos_Ordinal'])
    print(f"\nSpearman: r={corr:.4f}, p={p_value:.4f}")

    df_chi = df[['Genero_Ordinal', 'EstadoCivil_Ordinal']].dropna()
    tabla = pd.crosstab(df_chi['Genero_Ordinal'], df_chi['EstadoCivil_Ordinal'])
    chi2, p, dof, expected = stats.chi2_contingency(tabla)
    print(f"\nChi-Cuadrado: chi2={chi2:.4f}, p={p:.4f}, gl={dof}")

    df_reg = df[['Edad_Ordinal', 'Trabaja_Ordinal', 'Genero_Ordinal', 'Region_Ordinal', 'Ingresos_Ordinal']].dropna()
    X = df_reg[['Edad_Ordinal', 'Trabaja_Ordinal', 'Genero_Ordinal', 'Region_Ordinal']]
    y = df_reg['Ingresos_Ordinal']
    X = sm.add_constant(X)
    modelo = sm.OLS(y, X).fit()
    print("\nResumen regresión:")
    print(modelo.summary())

    print("\n====== KURTOSIS ======")
    numeric_df = df.select_dtypes(include=['number'])
    for col in numeric_df.columns:
        k = kurtosis(df[col].dropna())
        print(f"{col}: Kurtosis = {k:.2f}")

# ============================================
# GRÁFICOS EXPLORATORIOS
# ============================================
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

# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================
def main():
    df = cargar_datos()
    estadisticas_descriptivas(df)
    estadisticas_inferenciales(df)
    graficos_eda(df)

if __name__ == '__main__':
    main()

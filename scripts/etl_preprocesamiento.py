# ==========================================================================================
# Proyecto: Análisis Exploratorio de Datos (EDA) - Desempeño Académico
# Versión Local para scripts
# ==========================================================================================

import pandas as pd
import os

def cargar_datos():
    """
    Carga el dataset original.
    """
    path = 'data/raw/dataset_eda.xlsx'
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra el archivo {path}")
    return pd.read_excel(path)

def transformar_columnas(df):
    """
    Renombra las columnas originales a los nombres ordinales.
    """
    columnas_renombrar = {
        'Genero': 'Genero_Ordinal',
        'Estado civil': 'EstadoCivil_Ordinal',
        'Complejidad': 'ComplejidadArea_Ordinal',
        'Edad': 'Edad_Ordinal',
        'Trabaja': 'Trabaja_Ordinal',
        'Region': 'Region_Ordinal',
        'Zona Geo': 'ZonaGeo_Ordinal',
        'Imagen Marca': 'MarcaMercado_Ordinal',
        'Medelo educativo': 'ModeloEducativo_Ordinal',
        'Nivel Ingresos': 'Ingresos_Ordinal',
        'scalagpa': 'Grado4Scala_Ordinal'
    }
    
    df = df.rename(columns=columnas_renombrar)
    
    # Convertimos a entero 
    for col in columnas_renombrar.values():
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    return df

def auditar_datos(df, columnas_relevantes):
    """
    Audita las columnas relevantes.
    """
    print("\n🔎 Valores nulos por columna relevante:")
    print(df[columnas_relevantes].isnull().sum())

    filas_completas = len(df[columnas_relevantes].dropna())
    print(f"\n Total de filas completas para el modelo: {filas_completas}")
    return filas_completas

def limpiar_datos(df, columnas_relevantes):
    """
    Elimina filas incompletas solo para las columnas relevantes.
    """
    df_limpio = df.dropna(subset=columnas_relevantes)
    print(f"\n Eliminadas {len(df) - len(df_limpio)} filas incompletas (solo columnas relevantes).")
    return df_limpio

def guardar_datos(df):
    """
    Guarda el dataset limpio.
    """
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    path_salida = os.path.join(output_dir, 'Dataset_clear.csv')
    df.to_csv(path_salida, index=False)
    print(f"\n Archivo final generado correctamente: {path_salida}")

def main():
    df = cargar_datos()
    df = transformar_columnas(df)

    columnas_relevantes = [
        'Edad_Ordinal', 'Trabaja_Ordinal', 'Genero_Ordinal', 
        'Region_Ordinal', 'ZonaGeo_Ordinal', 'MarcaMercado_Ordinal',
        'ModeloEducativo_Ordinal', 'Ingresos_Ordinal', 'Grado4Scala_Ordinal'
    ]

    filas_completas = auditar_datos(df, columnas_relevantes)

    if filas_completas == 0:
        print("\n No hay suficientes datos completos para continuar.")
        return

    df_limpio = limpiar_datos(df, columnas_relevantes)
    guardar_datos(df_limpio)

if __name__ == '__main__':
    main()

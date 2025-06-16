# ==========================================================================================
# Proyecto: Análisis Exploratorio de Datos (EDA) - Desempeño Académico
# Versión Local para scripts
# ==========================================================================================

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def cargar_datos():
    """
    Carga el dataset limpio previamente procesado.
    """
    path = 'data/processed/Dataset_clear.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra el archivo {path}")
    return pd.read_csv(path)

def preparar_datos(df):
    
    columnas_relevantes = [
        'Edad_Ordinal', 'Trabaja_Ordinal', 'Genero_Ordinal', 
        'Region_Ordinal', 'ZonaGeo_Ordinal', 'MarcaMercado_Ordinal', 
        'ModeloEducativo_Ordinal', 'Ingresos_Ordinal', 'Grado4Scala_Ordinal'
    ]

    df_valid = df[columnas_relevantes].dropna()

    if df_valid.empty:
        print("No hay datos suficientes para entrenamiento ML.")
        return None, None, None, None

    X = df_valid.drop('Grado4Scala_Ordinal', axis=1)
    y = df_valid['Grado4Scala_Ordinal']

    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def svm_model(X_train, X_test, y_train, y_test):
    """
    Entrenamiento y evaluación de SVM.
    """
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    print("\n====== Support Vector Machine (SVM) ======")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

def ann_model(X_train, X_test, y_train, y_test):
    """
    Entrenamiento y evaluación de Red Neuronal Artificial básica.
    """
    mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print("\n====== Artificial Neural Network (ANN) ======")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

def main():
    df = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(df)

    if X_train is None:
        return

    svm_model(X_train, X_test, y_train, y_test)
    ann_model(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()

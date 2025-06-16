# Proyecto: Análisis de Dataset Educativo (EDA & ML)

---

## 📊 Descripción General

Este repositorio contiene un análisis estadístico exhaustivo y la implementación de modelos de machine learning sobre un dataset educativo. El proyecto fue desarrollado como práctica profesional para la asignatura **INF-7303-C1 (Ciencia de Datos I)** y está estructurado en módulos independientes para facilitar su mantenimiento y escalabilidad.

---

## 📁 Estructura del Proyecto

```plaintext
practica_dataset_eda/
│
├── data/
│   ├── raw/               # Dataset original (dataset_eda.xlsx)
│   └── processed/         # Dataset limpio (Dataset_clear.csv)
│
├── outputs/
│   └── visualizaciones/   # Gráficos y visualizaciones generados
│
├── scripts/
│   ├── etl_preprocesamiento.py        # ETL y transformación de variables
│   ├── eda_estadistica_inferencial.py # Análisis estadístico e inferencial
│   └── modelo_svm_ann.py              # Modelos supervisados SVM y ANN
│
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación del proyecto
```

---

## ⚙️ Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/bcuevas-dev/practica_dataset_eda.git
cd practica_dataset_eda
```

2. Crea y activa un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

**Principales dependencias:**
- pandas
- numpy
- scipy
- statsmodels
- scikit-learn
- seaborn
- matplotlib

---

## 🚀 Ejecución

El flujo de trabajo está dividido en tres etapas principales:

1. **Preprocesamiento de Datos**
    ```bash
    python scripts/etl_preprocesamiento.py
    ```
    Limpia el dataset original y genera el archivo procesado.

2. **Análisis Estadístico e Inferencial**
    ```bash
    python scripts/eda_estadistica_inferencial.py
    ```
    Realiza pruebas ANOVA, T-Test, Chi-cuadrado, correlaciones, regresión y genera gráficos exploratorios.

3. **Modelado Supervisado (SVM y ANN)**
    ```bash
    python scripts/modelo_svm_ann.py
    ```
    Entrena y evalúa modelos de clasificación supervisada.

## 🧪 Metodología de Análisis y Modelado

El análisis y modelado del dataset educativo se fundamentó en la aplicación de diversas pruebas estadísticas y técnicas de machine learning, detalladas a continuación:

- **ANOVA**: Comparación de medias entre múltiples grupos para identificar diferencias significativas.
- **T-Test**: Contraste de medias entre dos grupos para determinar diferencias estadísticamente relevantes.
- **Correlaciones (Pearson y Spearman)**: Medición de la relación y fuerza de asociación entre variables numéricas.
- **Chi-cuadrado**: Evaluación de la asociación entre variables categóricas.
- **Regresión lineal múltiple**: Modelado de la relación entre variables independientes y una variable dependiente.
- **Análisis de kurtosis**: Evaluación de la forma y concentración de la distribución de los datos.
- **Matrices de confusión**: Medición del desempeño de los modelos de clasificación mediante la comparación de predicciones y valores reales.
- **Modelos supervisados (SVM y ANN)**: Implementación de Support Vector Machines y Redes Neuronales Artificiales para tareas de clasificación y predicción de resultados educativos.

---

## 👨‍💻 Autor

**Bienvenido Cuevas**A

---
Detección de Fraude Bancario mediante Regresión Logística Binaria

Descripción

Este proyecto implementa un modelo de Machine Learning basado en Regresión Logística Binaria para la detección de transacciones fraudulentas realizadas con tarjetas de crédito.

El objetivo es clasificar cada transacción como legítima o fraudulenta utilizando técnicas de aprendizaje supervisado.

Dataset

Se utiliza el conjunto de datos Credit Card Fraud Detection obtenido de Kaggle.

Características del dataset:

- 284.807 transacciones.
- 31 variables.
- Variable objetivo: Class.
- Class = 0 → Transacción legítima.
- Class = 1 → Transacción fraudulenta.

Tecnologías Utilizadas

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-Learn

Dependencias

Instalar las librerías necesarias:

pip install pandas numpy scikit-learn matplotlib

Ejecución

1. Descargar el dataset creditcard.csv desde Kaggle.
2. Abrir el archivo Regresion_Logistica_Fraude_Bancario.ipynb en Google Colab.
3. Subir el archivo creditcard.csv al entorno de Colab.
4. Ejecutar las celdas en orden.
5. Revisar las matrices generadas por el modelo


Proceso Implementado

1. Carga del dataset.
2. Análisis de datos.
3. División entrenamiento/prueba (80%-20%).
4. Escalado de variables.
5. Entrenamiento de Regresión Logística.
6. Predicción.
7. Evaluación mediante Accuracy, Matriz de Confusión y Classification Report.

 Resultados

El modelo fue entrenado utilizando el 80% de los datos y validado con el 20% restante.

Las métricas evaluadas fueron:

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de Confusión

Autor

Elizabeth Ramirez Luna
Proyecto académico de Machine Learning e Ingeniería de Sistemas.

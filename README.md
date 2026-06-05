Detección de Fraude Bancario mediante Regresión Logística Binaria

Descripción

Este proyecto implementa un modelo de Machine Learning basado en Regresión Logística Binaria para la detección de transacciones fraudulentas realizadas con tarjetas de crédito.

El objetivo es clasificar cada transacción como legítima o fraudulenta utilizando técnicas de aprendizaje supervisado.

Conjunto de datos

Se utiliza el conjunto de datos de Detección de fraude de tarjetas de crédito obtenidos de Kaggle.

Características del conjunto de datos:

284.807 transacciones.
31 variables.
Variable objetivo: Clase.
Clase = 0 → Transacción legítima.
Clase = 1 → Transacción fraudulenta.
Tecnologías utilizadas

Pitón
Google Colab
Pandas
NumPy
Scikit-Learn
Dependencias

Instalar las librerías necesarias:

pip install pandas numpy scikit-learn matplotlib

Ejecución

Descargue el conjunto de datos creditcard.csv desde Kaggle.
Abrir el archivo Regresion_Logistica_Fraude_Bancario.ipynb en Google Colab.
Subir el archivo creditcard.csv al entorno de Colab.
Ejecutar las celdas en orden.
Revisar las matrices generadas por el modelo.
Proceso Implementado

Carga del dataset.
Análisis de datos.
División entrenamiento/prueba (80%-20%).
Escalado de variables.
Entrenamiento de Regresión Logística.
Predicción.
Evaluación mediante Precisión, Matriz de Confusión e Informe de Clasificación.
Resultados

El modelo fue entrenado utilizando el 80% de los datos y validado con el 20% restante.

Las métricas evaluadas fueron:

Exactitud
Precisión
Recordar
Puntuación F1
Matriz de Confusión
Autor

Elizabeth Ramirez Luna Proyecto académico de Machine Learning e Ingeniería de Sistemas.

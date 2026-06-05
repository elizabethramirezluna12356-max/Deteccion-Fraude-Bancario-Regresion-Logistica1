# ========================================
# MODELO - CARGA Y PREPARACIÓN DE DATOS
# ========================================

import pandas as pd
from sklearn.model_selection import train_test_split


def cargar_datos(ruta="creditcard.csv"):
    """Carga el dataset y elimina filas con valores faltantes."""
    df = pd.read_csv(ruta)
    df = df.dropna()

    print("Dimensiones después de limpiar:")
    print(df.shape)
    print("\nValores faltantes restantes:", df.isnull().sum().sum())

    return df


def preparar_variables(df):
    """Separa las variables predictoras (X) y la variable objetivo (y)."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("\nDimensión de X:", X.shape)
    print("Dimensión de y:", y.shape)

    return X, y


def dividir_datos(X, y, test_size=0.20, random_state=42):
    """Divide el dataset en conjuntos de entrenamiento y prueba."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("\nEntrenamiento:", X_train.shape)
    print("Prueba:", X_test.shape)

    return X_train, X_test, y_train, y_test

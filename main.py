# ========================================
# MAIN - SISTEMA DE DETECCIÓN DE FRAUDE
# Sistema de Detección de Fraude en Transacciones
# con Tarjetas de Crédito mediante Regresión Logística Binaria
# ========================================

from modelo import cargar_datos, preparar_variables, dividir_datos
from regresion_logistica import entrenar_modelo, predecir, evaluar, graficar_confusion


def main():
    # 1. Cargar y limpiar datos
    df = cargar_datos("creditcard.csv")

    # 2. Separar variables
    X, y = preparar_variables(df)

    # 3. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # 4. Entrenar modelo
    modelo = entrenar_modelo(X_train, y_train)

    # 5. Predecir
    y_pred = predecir(modelo, X_test)

    # 6. Evaluar
    evaluar(y_test, y_pred)

    # 7. Graficar matriz de confusión
    graficar_confusion(y_test, y_pred)


if __name__ == "__main__":
    main()

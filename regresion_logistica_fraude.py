# ========================================
# REGRESIÓN LOGÍSTICA - DETECCIÓN DE FRAUDE
# ========================================

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def entrenar_modelo(X_train, y_train, max_iter=5000):
    """Entrena y devuelve el modelo de regresión logística."""
    modelo = LogisticRegression(max_iter=max_iter)
    modelo.fit(X_train, y_train)
    print("Modelo entrenado correctamente.")
    return modelo


def predecir(modelo, X_test):
    """Realiza predicciones sobre el conjunto de prueba."""
    return modelo.predict(X_test)


def evaluar(y_test, y_pred):
    """Imprime exactitud, matriz de confusión y reporte de clasificación."""
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nExactitud: {accuracy:.4f}")
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))


def graficar_confusion(y_test, y_pred):
    """Genera y muestra la matriz de confusión visual."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm)
    plt.colorbar(im)

    ax.set_title("Matriz de Confusión - Detección de Fraude")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor Real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraude"])
    ax.set_yticklabels(["Normal", "Fraude"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()

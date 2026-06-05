# Sistema de Detección de Fraude en Transacciones
# con Tarjetas de Crédito mediante Regresión Logística Binaria

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ── Cargar dataset ──
df = pd.read_csv("creditcard.csv")
df = df.dropna()

# ── Variables ──
X = df.drop("Class", axis=1)
y = df["Class"]

# ── División entrenamiento/prueba ──
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ── Entrenamiento ──
modelo = LogisticRegression(max_iter=5000)
modelo.fit(X_train, y_train)

# ── Predicciones ──
y_pred = modelo.predict(X_test)

# ── Evaluación ──
print(f"Exactitud: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ── Gráfica ──
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

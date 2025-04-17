import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
import os

sns.set_palette("pastel") 

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)

# Leer el archivo Excel, saltando la primera fila
df = pd.read_csv("data/ganancias.csv", skiprows=1)

# Confirmar columnas
print("Columnas detectadas:", df.columns)

# Transformar de formato ancho a largo
df_largo = df.melt(id_vars="mes", var_name="año", value_name="ganancias")

# Asegurar orden correcto de meses
orden_meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio"]
df_largo["mes"] = pd.Categorical(df_largo["mes"], categories=orden_meses, ordered=True)

# Visualización: Gráfico de líneas
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_largo, x="mes", y="ganancias", hue="año", marker="o")
plt.title("Ganancias del 1er Semestre (2021–2023)")
plt.xlabel("Mes")
plt.ylabel("Ganancias")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/lineas.png", dpi=300)
plt.show()

# Agrupamos por año y mes para tener solo una fila por combinación
df_cluster = df_largo.groupby(["año", "mes"], as_index=False, observed=True)["ganancias"].sum()
print("Datos agrupados:\n", df_cluster)

# Clustering usando solo la columna de ganancias
X = df_cluster[["ganancias"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster["cluster"] = kmeans.fit_predict(X)

# Visualización: Gráfico de dispersión con Clustering KMeans
cluster_palette = ["#FFB3BA", "#BAE1FF", "#BFFCC6"]
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_cluster, x="mes", y="ganancias", hue="cluster", palette=cluster_palette)
plt.title("Clustering de Ganancias por Mes (KMeans)")
plt.xlabel("Mes")
plt.ylabel("Ganancias")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/kmeans_clustering.png", dpi=300)
plt.show()

# Gráfico de Barras – Ganancias Totales por Año
df_ano = df_largo.groupby("año", as_index=False)["ganancias"].sum()
plt.figure(figsize=(10, 6))
sns.barplot(data=df_ano, x="año", y="ganancias")
plt.title("Ganancias Totales por Año (2021–2023)")
plt.xlabel("Año")
plt.ylabel("Ganancias Totales")
plt.tight_layout()
plt.savefig("output/ganancias_totales_ano.png", dpi=300)
plt.show()

# Gráfico de Calor (Heatmap) – Ganancias por Mes y Año
df_heatmap = df_largo.pivot(index="mes", columns="año", values="ganancias")
plt.figure(figsize=(10, 6))
sns.heatmap(df_heatmap, annot=True, cmap="BuPu", fmt=".0f", linewidths=0.5)
plt.title("Heatmap de Ganancias por Mes y Año (2021–2023)")
plt.tight_layout()
plt.savefig("output/heatmap_ganancias.png", dpi=300)
plt.show()

# Boxplot – Distribución de Ganancias por Año
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_largo, x="año", y="ganancias")
plt.title("Distribución de Ganancias por Año (2021–2023)")
plt.xlabel("Año")
plt.ylabel("Ganancias")
plt.tight_layout()
plt.savefig("output/boxplot_ganancias.png", dpi=300)
plt.show()

# Regresión Lineal – Predicción de Ganancias por Año
# Convertir los años a valores numéricos para la regresión
df_largo['año_numeric'] = df_largo['año'].astype('category').cat.codes

# Entrenamiento del modelo de regresión lineal
model = LinearRegression()
model.fit(df_largo[['año_numeric']], df_largo['ganancias'])

# Predicción de ganancias
df_largo['predicciones'] = model.predict(df_largo[['año_numeric']])

# Visualización de la regresión
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_largo, x="año", y="ganancias", label="Datos reales")
sns.lineplot(data=df_largo, x="año", y="predicciones", color='#FF9999', label="Predicción")
plt.title("Regresión Lineal de Ganancias por Año (Predicción)")
plt.xlabel("Año")
plt.ylabel("Ganancias")
plt.legend()
plt.tight_layout()
plt.savefig("output/regresion_lineal.png", dpi=300)
plt.show()
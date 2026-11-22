# ---------------------------------------------------------------
# Parte 2. Obtencion y exploracion del dataset
# Fuente: Wine Quality (vino tinto), UCI Machine Learning Repository
# ---------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rutas import ORIGINAL, IMAGENES, VARIABLES

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
IMAGENES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Importar el dataset (el archivo esta separado por punto y coma)
df = pd.read_csv(ORIGINAL, sep=";")

# 2. Primeros 10 registros
print("=== PRIMEROS 10 REGISTROS ===")
print(df.head(10))

# 3. Dimensiones, columnas y tipos de datos
print("\n=== DIMENSIONES ===")
print("Filas y columnas:", df.shape)
print("\n=== COLUMNAS Y TIPOS DE DATOS ===")
df.info()

# 4. Estadisticas descriptivas
print("\n=== ESTADISTICAS DESCRIPTIVAS ===")
print(df.describe().round(3).T)

# 5. Valores nulos y duplicados
print("\n=== VALORES NULOS POR COLUMNA ===")
print(df.isnull().sum())
print("\nRegistros duplicados:", int(df.duplicated().sum()))

# 6. Graficas exploratorias
# 6.1 Distribucion de la calidad
plt.figure(figsize=(7, 4.5))
ax = sns.countplot(data=df, x="quality", color="#8c2d3f")
ax.bar_label(ax.containers[0])
plt.title("Distribucion de la calidad del vino")
plt.xlabel("Calidad (calificacion del catador)")
plt.ylabel("Cantidad de vinos")
plt.tight_layout()
plt.savefig(IMAGENES / "exploracion_1_calidad.png", dpi=150)
plt.close()

# 6.2 Histogramas de las variables elegidas para el agrupamiento
fig, ejes = plt.subplots(1, 3, figsize=(13, 4))
for eje, var in zip(ejes, VARIABLES):
    sns.histplot(df[var], kde=True, ax=eje, color="#8c2d3f")
    eje.set_title(f"Distribucion de {var}")
plt.tight_layout()
plt.savefig(IMAGENES / "exploracion_2_histogramas.png", dpi=150)
plt.close()

# 6.3 Mapa de calor de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 8})
plt.title("Correlacion entre las variables del dataset")
plt.tight_layout()
plt.savefig(IMAGENES / "exploracion_3_correlacion.png", dpi=150)
plt.close()

# 6.4 Diagramas de caja para ver valores atipicos
fig, ejes = plt.subplots(1, 3, figsize=(13, 4))
for eje, var in zip(ejes, VARIABLES):
    sns.boxplot(y=df[var], ax=eje, color="#d9a0a8")
    eje.set_title(f"Valores atipicos en {var}")
plt.tight_layout()
plt.savefig(IMAGENES / "exploracion_4_cajas.png", dpi=150)
plt.close()

# 7. Correlacion de las variables con la calidad (solo informativa)
print("\n=== CORRELACION CON LA CALIDAD ===")
print(df.corr()["quality"].drop("quality").sort_values(ascending=False).round(3))
print("\nGraficas guardadas en la carpeta imagenes/")

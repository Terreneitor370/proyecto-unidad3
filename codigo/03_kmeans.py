# ---------------------------------------------------------------
# Parte 4. Implementacion del algoritmo K-Means
# Genera datos/procesado/vino_clusters.csv con el cluster de cada vino
# ---------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from rutas import LIMPIO, CLUSTERS, IMAGENES, VARIABLES, VARIABLES_AMPLIADAS, SEMILLA

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")
IMAGENES.mkdir(exist_ok=True)

df = pd.read_csv(LIMPIO)

# 1. Seleccion de variables y escalamiento (StandardScaler)
#    Las variables tienen escalas distintas (alcohol ~10, sulfatos ~0.6);
#    sin escalar, K-Means daria mas peso al alcohol.
escalador = StandardScaler()
X = escalador.fit_transform(df[VARIABLES])
print("=== VARIABLES DEL MODELO ===", VARIABLES)
print("Registros:", X.shape[0])

# 2. Evaluar valores de K (2 a 8): inercia, silueta e indice Davies-Bouldin
print("\n=== EVALUACION DE K ===")
resultados = []
for k in range(2, 9):
    m = KMeans(n_clusters=k, random_state=SEMILLA, n_init=10).fit(X)
    resultados.append({
        "K": k,
        "inercia": round(m.inertia_, 1),
        "silueta": round(silhouette_score(X, m.labels_), 3),
        "davies_bouldin": round(davies_bouldin_score(X, m.labels_), 3),
    })
evaluacion = pd.DataFrame(resultados)
print(evaluacion.to_string(index=False))

fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
ejes[0].plot(evaluacion["K"], evaluacion["inercia"], marker="o", color="#8c2d3f")
ejes[0].set_title("Metodo del codo (inercia)")
ejes[0].set_xlabel("Numero de clusters (K)")
ejes[0].set_ylabel("Inercia")
ejes[1].plot(evaluacion["K"], evaluacion["silueta"], marker="o", color="#2d6a8c")
ejes[1].set_title("Coeficiente de silueta")
ejes[1].set_xlabel("Numero de clusters (K)")
ejes[1].set_ylabel("Silueta")
plt.tight_layout()
plt.savefig(IMAGENES / "kmeans_1_codo_silueta.png", dpi=150)
plt.close()

# 3. Seleccion de K: se elige el K con mayor coeficiente de silueta.
#    K = 3 tambien tiene el menor Davies-Bouldin entre K = 2 y K = 4 y
#    permite distinguir un perfil mas que K = 2.
K_FINAL = int(evaluacion.loc[evaluacion["silueta"].idxmax(), "K"])
print("\nK elegido:", K_FINAL)

# 4. Modelo final y asignacion de cluster con fit_predict()
modelo = KMeans(n_clusters=K_FINAL, random_state=SEMILLA, n_init=10)
df["cluster"] = modelo.fit_predict(X)
print("Iteraciones realizadas:", modelo.n_iter_)
print("Inercia:", round(modelo.inertia_, 2))
print("Silueta:", round(silhouette_score(X, df["cluster"]), 3))
print("Davies-Bouldin:", round(davies_bouldin_score(X, df["cluster"]), 3))

# 5. Centroides en unidades originales y cantidad de registros por cluster
centroides = pd.DataFrame(escalador.inverse_transform(modelo.cluster_centers_),
                          columns=VARIABLES).round(3)
centroides.index.name = "cluster"
print("\n=== CENTROIDES (unidades originales) ===")
print(centroides)
print("\n=== REGISTROS POR CLUSTER ===")
print(df["cluster"].value_counts().sort_index())

# 6. Grafica de los agrupamientos
fig, ejes = plt.subplots(1, 2, figsize=(13, 5.5))
paleta = sns.color_palette("Set2", K_FINAL)
sns.scatterplot(data=df, x="alcohol", y="volatile acidity", hue="cluster",
                palette=paleta, alpha=0.7, ax=ejes[0])
ejes[0].scatter(centroides["alcohol"], centroides["volatile acidity"], marker="X",
                s=250, color="red", edgecolor="black", label="Centroides")
ejes[0].set_title("Alcohol vs acidez volatil")
ejes[0].legend(title="Cluster")
sns.scatterplot(data=df, x="alcohol", y="sulphates", hue="cluster",
                palette=paleta, alpha=0.7, ax=ejes[1], legend=False)
ejes[1].scatter(centroides["alcohol"], centroides["sulphates"], marker="X",
                s=250, color="red", edgecolor="black")
ejes[1].set_title("Alcohol vs sulfatos")
plt.suptitle(f"Agrupamiento de vinos tintos con K-Means (K = {K_FINAL})")
plt.tight_layout()
plt.savefig(IMAGENES / "kmeans_2_clusters.png", dpi=150)
plt.close()

# 7. Comparar dos configuraciones del modelo
#    A: 3 variables (alcohol, acidez volatil, sulfatos)
#    B: 5 variables (se agregan acido citrico y azucar residual)
print("\n=== COMPARACION DE DOS CONFIGURACIONES ===")
XB = StandardScaler().fit_transform(df[VARIABLES_AMPLIADAS])
modelo_b = KMeans(n_clusters=K_FINAL, random_state=SEMILLA, n_init=10)
etiquetas_b = modelo_b.fit_predict(XB)
comparacion = pd.DataFrame({
    "Configuracion": ["A: 3 variables", "B: 5 variables"],
    "Variables": [len(VARIABLES), len(VARIABLES_AMPLIADAS)],
    "Inercia": [round(modelo.inertia_, 1), round(modelo_b.inertia_, 1)],
    "Silueta": [round(silhouette_score(X, df["cluster"]), 3), round(silhouette_score(XB, etiquetas_b), 3)],
    "Davies-Bouldin": [round(davies_bouldin_score(X, df["cluster"]), 3),
                       round(davies_bouldin_score(XB, etiquetas_b), 3)],
    "Registros por cluster": [sorted(df["cluster"].value_counts().tolist(), reverse=True),
                              sorted(pd.Series(etiquetas_b).value_counts().tolist(), reverse=True)],
})
print(comparacion.to_string(index=False))
print("\nNota: la inercia no es comparable entre A y B porque tienen distinto numero de variables;")
print("la silueta y Davies-Bouldin si permiten comparar la calidad de la separacion.")

# 8. Guardar el dataset con los clusters asignados
df.to_csv(CLUSTERS, index=False)
print("\nDataset con clusters guardado:", CLUSTERS.name, df.shape)

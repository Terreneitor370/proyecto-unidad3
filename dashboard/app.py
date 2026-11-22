# ----------------------------------------------------------------
# Dashboard interactivo: Segmentacion de vinos tintos con K-Means
# Proyecto integrador - Unidad 3 (IDGS 9-2)
#
# Pagina 1: Analisis general (dataset original)
# Pagina 2: Segmentacion mediante K-Means (dataset procesado)
#
# Ejecutar desde la raiz del repositorio:
#   streamlit run dashboard/app.py
# ----------------------------------------------------------------
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Rutas relativas al proyecto (nunca rutas de la computadora)
RAIZ = Path(__file__).resolve().parent.parent
ORIGINAL = RAIZ / "datos" / "original" / "winequality-red.csv"
CLUSTERS = RAIZ / "datos" / "procesado" / "vino_clusters.csv"

VARIABLES = ["alcohol", "volatile acidity", "sulphates"]
SEMILLA = 42

# Metricas del modelo final (generadas por codigo/03_kmeans.py con random_state=42)
METRICAS_MODELO = {
    "K": 3,
    "inercia": 2093.07,
    "silueta": 0.335,
    "davies_bouldin": 1.133,
    "iteraciones": 11,
}
COMPARACION_CONFIGURACIONES = pd.DataFrame(
    {
        "Configuracion": ["A: 3 variables", "B: 5 variables"],
        "Variables usadas": ["alcohol, acidez volatil, sulfatos",
                             "alcohol, acidez volatil, sulfatos, acido citrico, azucar residual"],
        "Inercia": [2093.1, 4291.6],
        "Silueta": [0.335, 0.214],
        "Davies-Bouldin": [1.133, 1.665],
        "Registros por cluster": ["737 / 278 / 344", "640 / 392 / 327"],
    }
)

# Nombre descriptivo de cada cluster (Parte 5 del proyecto)
NOMBRES_CLUSTERS = {
    0: "Vinos comunes",
    1: "Vinos equilibrados",
    2: "Vinos premium",
}
DESCRIPCION_CLUSTERS = {
    0: "Son la mayoria del mercado (737 vinos, 54%). Tienen el alcohol mas bajo "
       "(9.8% prom.), la acidez volatil mas alta (0.61) y los sulfatos mas bajos "
       "(0.57). Su calidad promedio es 5.27, la mas baja de los tres grupos.",
    1: "Grupo moderado en alcohol (10.2%) pero con la acidez volatil mas baja "
       "(0.41) y los sulfatos mas altos (0.84). La mezcla de baja acidez y buena "
       "conservacion les da una calidad promedio de 5.85 con solo 278 vinos (20%).",
    2: "Los vinos de mayor graduacion alcoholica (11.9% prom.) con acidez "
       "volatil baja (0.44) y sulfatos intermedios (0.66). Son 344 vinos (26%) y "
       "alcanzan la mejor calidad promedio: 6.19, con la mayor concentracion de "
       "vinos de calidad 7 y 8.",
}
HALLAZGOS = [
    "El alcohol es la variable que mas separa los grupos: el cluster de vinos "
    "premium promedia 11.9 grados frente a 9.8 del grupo comun, y a la vez es el "
    "de mejor calidad (6.19 frente a 5.27).",
    "La acidez volatil alta deteriora el perfil: el grupo con mas acidez "
    "volatil (0.61) concentra 9 de los 9 vinos de calidad 3 y el 82% de los de "
    "calidad 4, mientras el grupo de baja acidez casi no tiene vinos fallidos.",
    "Los sulfatos altos acompanan a los vinos mejor valorados cuando van juntos "
    "de una acidez baja: el grupo equilibrado (0.84 de sulfatos) mejora 0.58 "
    "puntos de calidad promedio sobre el grupo comun sin ser el mas alcoholico.",
]
RECOMENDACIONES = [
    "Priorizar el control de la acidez volatil en bodega: mantenerla por debajo "
    "de 0.45 g/L, porque es la principal marca de los lotes de calidad baja.",
    "Usar la segmentacion como referencia de surtido: los vinos premium y "
    "equilibrados (46% del catalogo) merecen mas espacio y promocion que su "
    "peso actual en estanteria.",
    "Completar la caracterizacion con variables sensoriales (cuerpo, aroma) "
    "que el dataset no incluye, para afinar la oferta por perfil de cliente.",
]
LIMITACIONES = [
    "K-Means asume grupos esfericos y de tamano similar; la silueta de 0.335 "
    "indica separacion moderada, con zonas de frontera entre clusters.",
    "El modelo solo uso 3 variables quimicas; variables como el tipo de uva, "
    "la añada o el precio no estan en el dataset y pueden explicar parte de la "
    "calidad.",
    "La columna quality es un promedio de opiniones de catadores (estudios del "
    "mismo panel), por lo que la relacion cluster-calidad es indicativa y no "
    "una etiqueta de calidad real.",
]

COLORES_CLUSTERS = ["#b0413e", "#3f7cb0", "#3f8f5f"]


@st.cache_data
def cargar_original() -> pd.DataFrame:
    return pd.read_csv(ORIGINAL, sep=";")


@st.cache_data
def cargar_clusters() -> pd.DataFrame:
    df = pd.read_csv(CLUSTERS)
    df["nombre_cluster"] = df["cluster"].map(
        lambda c: f"Cluster {c} · {NOMBRES_CLUSTERS[c]}"
    )
    return df


def metrica_kpi(df: pd.DataFrame, columna: str, formato: str) -> float:
    return df[columna].agg(["mean"]).iloc[0]


# ----------------------------------------------------------------
# Pagina 1. Analisis general (solo dataset original)
# ----------------------------------------------------------------
def pagina_analisis_general():
    st.title("Análisis general del dataset de vinos tintos")
    st.markdown(
        "Exploración del dataset original **Wine Quality (UCI)**: 1,599 vinos "
        "tintos y 11 variables fisicoquimicas + calidad (escala 3 a 8)."
    )
    st.divider()

    df = cargar_original()

    # --- Filtros interactivos ---
    st.sidebar.header("Filtros")
    calidad = st.sidebar.slider(
        "Calidad del vino",
        int(df["quality"].min()), int(df["quality"].max()),
        (int(df["quality"].min()), int(df["quality"].max())),
    )
    alcohol = st.sidebar.slider(
        "Alcohol (% vol.)",
        float(df["alcohol"].min()), float(df["alcohol"].max()),
        (float(df["alcohol"].min()), float(df["alcohol"].max())),
        step=0.1,
    )
    acidez = st.sidebar.slider(
        "Acidez volatil (g/L)",
        float(df["volatile acidity"].min()), float(df["volatile acidity"].max()),
        (float(df["volatile acidity"].min()), float(df["volatile acidity"].max())),
        step=0.01,
    )
    filtro = (
        df["quality"].between(*calidad)
        & df["alcohol"].between(*alcohol)
        & df["volatile acidity"].between(*acidez)
    )
    df_f = df[filtro]

    if df_f.empty:
        st.warning("No hay vinos con esos filtros. Amplia los rangos.")
        return

    # --- 4 KPI ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vinos analizados", f"{len(df_f):,}")
    c2.metric("Calidad promedio", f"{df_f['quality'].mean():.2f}")
    c3.metric("Alcohol promedio", f"{df_f['alcohol'].mean():.2f} %")
    c4.metric("Acidez volatil promedio", f"{df_f['volatile acidity'].mean():.3f} g/L")

    st.divider()

    # --- Graficas generales ---
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df_f, x="quality",
            title="Distribución de la calidad",
            color_discrete_sequence=["#8c2d3f"], category_orders={"quality": sorted(df_f["quality"].unique())},
        )
        fig.update_layout(xaxis_title="Calidad (3 = peor, 8 = mejor)", yaxis_title="Vinos", showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.histogram(
            df_f, x="alcohol", nbins=30,
            title="Distribución del alcohol",
            color_discrete_sequence=["#2d6a8c"],
        )
        fig.update_layout(xaxis_title="Alcohol (% vol.)", yaxis_title="Vinos", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(
            df_f, x="alcohol", y="volatile acidity", color="quality",
            color_continuous_scale="RdYlGn",
            title="Alcohol vs acidez volátil (color = calidad)",
            labels={"alcohol": "Alcohol (% vol.)", "volatile acidity": "Acidez volátil (g/L)"},
        )
        st.plotly_chart(fig, width="stretch")
    with col4:
        corr = df_f.corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            title="Correlación entre variables",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("Ver los datos filtrados"):
        st.dataframe(df_f, width="stretch")


# ----------------------------------------------------------------
# Pagina 2. Segmentacion mediante K-Means
# ----------------------------------------------------------------
def pagina_segmentacion():
    st.title("Segmentación mediante K-Means")
    st.markdown(
        "Los vinos se agruparon con **K-Means (K = 3)** usando alcohol, acidez "
        "volátil y sulfatos, escalados con StandardScaler "
        f"(random_state = {SEMILLA})."
    )
    st.divider()

    df = cargar_clusters()

    # --- Filtros interactivos ---
    st.sidebar.header("Filtros")
    opciones = sorted(df["cluster"].unique())
    clusters_sel = st.sidebar.multiselect(
        "Clusters a mostrar", opciones, default=opciones,
        format_func=lambda c: f"Cluster {c} · {NOMBRES_CLUSTERS[c]}",
    )
    alcohol = st.sidebar.slider(
        "Alcohol (% vol.)",
        float(df["alcohol"].min()), float(df["alcohol"].max()),
        (float(df["alcohol"].min()), float(df["alcohol"].max())),
        step=0.1,
    )
    sulfatos = st.sidebar.slider(
        "Sulfatos (g/L)",
        float(df["sulphates"].min()), float(df["sulphates"].max()),
        (float(df["sulphates"].min()), float(df["sulphates"].max())),
        step=0.01,
    )
    filtro = (
        df["cluster"].isin(clusters_sel)
        & df["alcohol"].between(*alcohol)
        & df["sulphates"].between(*sulfatos)
    )
    df_f = df[filtro]

    if df_f.empty:
        st.warning("No hay vinos con esos filtros. Amplia los rangos.")
        return

    # --- 4 KPI ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vinos segmentados", f"{len(df_f):,}")
    c2.metric("Clusters", len(clusters_sel))
    c3.metric("Silueta del modelo", f"{METRICAS_MODELO['silueta']:.3f}")
    c4.metric("Calidad promedio", f"{df_f['quality'].mean():.2f}")

    st.divider()

    # --- Distribucion y dispersion ---
    col1, col2 = st.columns(2)
    with col1:
        cuenta = df_f["nombre_cluster"].value_counts().reset_index()
        cuenta.columns = ["Grupo", "Vinos"]
        fig = px.bar(
            cuenta, x="Grupo", y="Vinos", color="Grupo",
            color_discrete_sequence=COLORES_CLUSTERS,
            title="Distribución de registros por clúster", text_auto=True,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.scatter(
            df_f, x="alcohol", y="volatile acidity", color="nombre_cluster",
            color_discrete_sequence=COLORES_CLUSTERS, opacity=0.65,
            title="Agrupamientos: alcohol vs acidez volátil",
            labels={"alcohol": "Alcohol (% vol.)", "volatile acidity": "Acidez volátil (g/L)"},
            hover_data=["quality", "sulphates"],
        )
        # Centroides del modelo final (unidades originales, random_state=42)
        centroides = pd.DataFrame(
            {
                "alcohol": [9.843, 10.215, 11.855],
                "volatile acidity": [0.610, 0.415, 0.444],
                "cluster": [0, 1, 2],
            }
        )
        centroides["nombre_cluster"] = centroides["cluster"].map(
            lambda c: f"Cluster {c} · {NOMBRES_CLUSTERS[c]}"
        )
        fig.add_trace(
            go.Scatter(
                x=centroides["alcohol"], y=centroides["volatile acidity"],
                mode="markers+text", marker=dict(symbol="x", size=14, color="black"),
                text=["C0", "C1", "C2"], textposition="top center",
                name="Centroides",
            )
        )
        st.plotly_chart(fig, width="stretch")

    # --- Dos graficas comparativas ---
    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(
            df_f, x="nombre_cluster", y="alcohol", color="nombre_cluster",
            color_discrete_sequence=COLORES_CLUSTERS,
            title="Comparativa: alcohol por clúster",
            labels={"nombre_cluster": "Grupo", "alcohol": "Alcohol (% vol.)"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with col4:
        medias = df_f.groupby("nombre_cluster")[VARIABLES + ["quality"]].mean().reset_index()
        medias_largas = medias.melt(
            id_vars="nombre_cluster", var_name="Variable", value_name="Promedio"
        )
        medias_largas["Promedio escalado"] = medias_largas.groupby("Variable")["Promedio"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min() if s.max() > s.min() else 1)
        )
        fig = px.bar(
            medias_largas, x="Variable", y="Promedio escalado",
            color="nombre_cluster", barmode="group",
            color_discrete_sequence=COLORES_CLUSTERS,
            title="Comparativa de medias por clúster (escala común 0–1)",
            labels={"nombre_cluster": "Grupo", "Promedio escalado": "Promedio escalado"},
        )
        fig.update_layout(xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    # --- Relacion cluster - calidad ---
    st.subheader("Relación entre clústeres y calidad")
    tabla_calidad = pd.crosstab(df_f["nombre_cluster"], df_f["quality"])
    fig = px.imshow(
        tabla_calidad, text_auto=True, aspect="auto",
        title="Vinos por clúster y nivel de calidad",
        color_continuous_scale="Reds", labels={"color": "Vinos"},
    )
    fig.update_layout(xaxis_title="Calidad", yaxis_title="Grupo")
    st.plotly_chart(fig, width="stretch")

    # --- Tabla de estadisticas descriptivas por cluster ---
    st.subheader("Estadísticas descriptivas por clúster")
    stats = df_f.groupby("nombre_cluster")[VARIABLES + ["quality"]].agg(
        ["count", "mean", "std", "min", "max"]
    ).round(3)
    stats.columns = [f"{var} · {est}" for var, est in stats.columns]
    st.dataframe(stats, width="stretch")

    tabla_comparativa = df_f.groupby("nombre_cluster").agg(
        Vinos=("cluster", "size"),
        Alcohol_prom=("alcohol", "mean"),
        Acidez_volatil_prom=("volatile acidity", "mean"),
        Sulfatos_prom=("sulphates", "mean"),
        Calidad_prom=("quality", "mean"),
    ).round(2)
    st.markdown("**Tabla comparativa de los grupos**")
    st.dataframe(tabla_comparativa, width="stretch")

    # --- Comparacion de configuraciones del modelo ---
    st.subheader("Comparación de dos configuraciones del modelo")
    st.dataframe(COMPARACION_CONFIGURACIONES, width="stretch")
    st.caption(
        "La inercia no es comparable entre configuraciones (depende del número "
        "de variables). Por silueta y Davies-Bouldin, la configuración A "
        "(3 variables) separa mejor los grupos, por eso es la usada."
    )

    # --- Descripcion de los grupos ---
    st.subheader("Descripción de los grupos identificados")
    for c in clusters_sel:
        with st.expander(f"**Cluster {c} · {NOMBRES_CLUSTERS[c]}**"):
            st.markdown(DESCRIPCION_CLUSTERS[c])

    # --- Hallazgos, recomendaciones y limitaciones ---
    st.subheader("Hallazgos relevantes")
    for i, h in enumerate(HALLAZGOS, 1):
        st.markdown(f"{i}. {h}")

    st.subheader("Recomendaciones")
    for i, r in enumerate(RECOMENDACIONES, 1):
        st.markdown(f"{i}. {r}")

    st.subheader("Limitaciones del modelo")
    for i, l in enumerate(LIMITACIONES, 1):
        st.markdown(f"{i}. {l}")

    with st.expander("Ver el dataset con clusters filtrado"):
        st.dataframe(df_f, width="stretch")


# ----------------------------------------------------------------
# Configuracion de la aplicacion de dos paginas
# ----------------------------------------------------------------
app = st.navigation(
    [
        st.Page(pagina_analisis_general, title="Análisis general"),
        st.Page(pagina_segmentacion, title="Segmentación K-Means"),
    ]
)
st.set_page_config(
    page_title="Dashboard · Vinos tintos K-Means",
    layout="wide",
)
app.run()

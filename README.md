# Segmentación de vinos tintos mediante K-Means

Proyecto integrador de la Unidad 3 de Extracción de Conocimiento en Bases de Datos (IDGS 9-2, UTSLRC).

**Integrantes:** Blanca Isabel Sánchez Celis y Kassandra Cuadras Avilez.

## Descripción

Los vinos suelen clasificarse por precio, marca o etiqueta, pero eso no dice qué tan parecidos son en su composición química. En este proyecto se agrupan 1,359 vinos tintos según sus propiedades fisicoquímicas con el algoritmo **K-Means**, para descubrir qué perfiles de vino existen sin etiquetas previas.

La columna `quality` (calificación de un catador) no se usa para entrenar el modelo. Se emplea al final para comprobar si los grupos encontrados se relacionan con la calidad. Los resultados se presentan en un dashboard interactivo de dos páginas hecho con Streamlit.

## Dataset

- **Nombre:** Wine Quality (vino tinto).
- **Fuente:** UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/186/wine+quality>.
- **Tamaño:** 1,599 registros y 12 variables numéricas (11 fisicoquímicas y la calidad).
- **Archivo:** `datos/original/winequality-red.csv` (separado por punto y coma).
- **Después de la limpieza:** 1,359 registros (se eliminaron 240 duplicados).

## Metodología

1. **Exploración:** primeros registros, dimensiones, tipos, estadísticas, nulos, duplicados y 4 gráficas.
2. **Limpieza:** eliminación de duplicados y tratamiento de valores atípicos por el criterio del rango intercuartílico (IQR); los valores extremos se limitan al borde del rango en lugar de borrar vinos.
3. **Preparación:** escalamiento con `StandardScaler`.
4. **K-Means:** variables `alcohol`, `volatile acidity` y `sulphates`; se evaluaron valores de K de 2 a 8 con la inercia, el coeficiente de silueta y el índice Davies-Bouldin; modelo final con `fit_predict()` y `random_state=42`.
5. **Comparación de configuraciones:** 3 variables contra 5 variables (se agregan `citric acid` y `residual sugar`).
6. **Análisis y dashboard:** estadísticas por grupo, nombres descriptivos, hallazgos y recomendaciones.

## Resultados principales

Con K = 3 se obtuvo una silueta de 0.335 y un índice Davies-Bouldin de 1.133.

| Clúster | Nombre | Vinos | Alcohol | Acidez volátil | Sulfatos | Calidad promedio |
|---|---|---|---|---|---|---|
| 0 | Vinos comunes | 737 | 9.84 | 0.61 | 0.57 | 5.27 |
| 1 | Vinos equilibrados | 278 | 10.22 | 0.41 | 0.85 | 5.85 |
| 2 | Vinos premium | 344 | 11.86 | 0.44 | 0.66 | 6.19 |

La configuración de 3 variables separa mejor los grupos (silueta 0.335) que la de 5 variables (silueta 0.214), por lo que se conservó. La separación entre grupos es moderada: hay zonas de frontera donde los vinos de clústeres vecinos se traslapan.

## Herramientas

Python 3, pandas, NumPy, scikit-learn, matplotlib, seaborn, Plotly y Streamlit.

## Estructura del repositorio

```
datos/
  original/            dataset original (winequality-red.csv)
  procesado/           vino_limpio.csv y vino_clusters.csv (con el clúster de cada vino)
codigo/
  rutas.py             rutas relativas y variables compartidas
  01_exploracion.py    exploración del dataset y gráficas exploratorias
  02_limpieza.py       limpieza y tratamiento de valores atípicos
  03_kmeans.py         evaluación de K, modelo final, centroides y comparación
dashboard/
  app.py               dashboard de Streamlit (2 páginas)
.streamlit/
  config.toml          tema del dashboard
documentacion/         documento PDF del proyecto
imagenes/              gráficas generadas por los scripts
requirements.txt       bibliotecas necesarias
```

## Cómo ejecutar el proyecto

Todos los comandos se ejecutan desde la raíz del repositorio.

**1. Clonar el repositorio e instalar las bibliotecas**

```bash
git clone https://github.com/Terreneitor370/proyecto-unidad3.git
cd proyecto-unidad3
pip install -r requirements.txt
```

**2. Ejecutar los scripts en orden** (cada uno usa el resultado del anterior)

```bash
python codigo/01_exploracion.py
python codigo/02_limpieza.py
python codigo/03_kmeans.py
```

- `01_exploracion.py` imprime la exploración y guarda las gráficas en `imagenes/`.
- `02_limpieza.py` genera `datos/procesado/vino_limpio.csv`.
- `03_kmeans.py` genera `datos/procesado/vino_clusters.csv` y las gráficas del modelo.

Los archivos procesados ya vienen incluidos en el repositorio, así que este paso solo es necesario para reproducir los resultados.

**3. Abrir el dashboard**

```bash
streamlit run dashboard/app.py
```

Se abre en el navegador en `http://localhost:8501`. El menú lateral permite cambiar entre las dos páginas:

- **Análisis general:** indicadores, estadísticas y gráficas del dataset original, con filtros de calidad, alcohol y acidez volátil.
- **Segmentación K-Means:** distribución por clúster, dispersión con centroides, comparaciones, tabla de estadísticas, descripción de los grupos, hallazgos, recomendaciones y limitaciones, con filtros de clúster, alcohol y sulfatos.

## Limitaciones

- K-Means asume grupos de forma esférica y tamaño parecido; la silueta de 0.335 indica separación moderada.
- Solo se usaron 3 variables químicas; factores como el tipo de uva, la añada o el precio no están en el dataset.
- La calidad es una calificación de catadores, por lo que su relación con los grupos es indicativa y no una etiqueta real.

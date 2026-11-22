# Segmentacion de vinos tintos mediante K-Means

Proyecto integrador de la Unidad 3 - Extraccion de Conocimiento en Bases de Datos (IDGS 9-2).

Integrantes: Blanca Isabel Sanchez Celis y Kassie.

## Descripcion

Se agrupan vinos tintos segun su composicion quimica con el algoritmo K-Means, para descubrir perfiles de vino sin etiquetas previas. La columna `quality` no se usa para entrenar; se emplea al final para comprobar si los grupos se relacionan con la calidad. Los resultados se presentan en un dashboard interactivo hecho con Streamlit.

## Dataset

Wine Quality (vino tinto), UCI Machine Learning Repository: 1,599 registros y 12 variables. Archivo: `datos/original/winequality-red.csv` (separado por punto y coma).

## Herramientas

Python, pandas, scikit-learn, matplotlib, seaborn y Streamlit.

## Estructura del repositorio

```
datos/original/       dataset original
datos/procesado/      dataset limpio con los clusters asignados
codigo/               scripts de exploracion, limpieza y K-Means
dashboard/            dashboard de Streamlit
documentacion/        documento PDF del proyecto
imagenes/             graficas generadas
requirements.txt      bibliotecas necesarias
```

## Como ejecutar

Se completara al terminar el proyecto.

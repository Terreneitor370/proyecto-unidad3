"""Rutas del proyecto (relativas a la raiz del repositorio) para no depender de la computadora."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ORIGINAL = RAIZ / "datos" / "original" / "winequality-red.csv"
LIMPIO = RAIZ / "datos" / "procesado" / "vino_limpio.csv"
CLUSTERS = RAIZ / "datos" / "procesado" / "vino_clusters.csv"
IMAGENES = RAIZ / "imagenes"

# Variables elegidas para el agrupamiento (configuracion principal)
VARIABLES = ["alcohol", "volatile acidity", "sulphates"]
# Variables adicionales usadas solo en la segunda configuracion del modelo
VARIABLES_AMPLIADAS = VARIABLES + ["citric acid", "residual sugar"]
SEMILLA = 42

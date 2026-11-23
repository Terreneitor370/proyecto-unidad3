# ---------------------------------------------------------------
# Parte 3. Limpieza y preparacion de datos
# Genera datos/procesado/vino_limpio.csv (sin escalar; el escalado
# se aplica en 03_kmeans.py justo antes de entrenar el modelo)
# ---------------------------------------------------------------
import pandas as pd
from rutas import ORIGINAL, LIMPIO, VARIABLES_AMPLIADAS

pd.set_option("display.width", 120)
df = pd.read_csv(ORIGINAL, sep=";")
print("Registros originales:", len(df))

# 1. Inconsistencias: nombres de columna y rangos validos
df.columns = df.columns.str.strip()
print("\n=== REVISION DE INCONSISTENCIAS ===")
print("Valores negativos:", int((df < 0).sum().sum()))
print("pH fuera de 0-14:", int((~df["pH"].between(0, 14)).sum()))
print("Alcohol fuera de 0-20 %:", int((~df["alcohol"].between(0, 20)).sum()))
print("Tipos de datos:", df.dtypes.value_counts().to_dict())

# 2. Valores nulos (el dataset no trae; se deja la verificacion)
print("\n=== VALORES NULOS ===")
print("Total de nulos:", int(df.isnull().sum().sum()))

# 3. Registros duplicados: se eliminan porque un mismo vino contaria
#    dos veces y daria mas peso a ese perfil en los centroides
duplicados = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("\n=== DUPLICADOS ===")
print("Duplicados eliminados:", duplicados)
print("Registros despues de eliminar duplicados:", len(df))

# 4. Valores atipicos con el criterio del rango intercuartilico (IQR)
print("\n=== VALORES ATIPICOS (IQR) EN LAS VARIABLES DEL MODELO ===")
limites = {}
for var in VARIABLES_AMPLIADAS:
    q1, q3 = df[var].quantile([0.25, 0.75])
    iqr = q3 - q1
    inf, sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    limites[var] = (inf, sup)
    n = int(((df[var] < inf) | (df[var] > sup)).sum())
    print(f"{var:18s} limites [{inf:6.3f}, {sup:6.3f}]  atipicos: {n}")

# Tratamiento: se limitan (winsorizacion) al limite del IQR en lugar de
# borrar registros. K-Means es sensible a valores extremos porque usa
# distancias, pero cada fila es un vino real y no queremos perderlos.
for var, (inf, sup) in limites.items():
    df[var] = df[var].clip(lower=inf, upper=sup).round(4)

print("\nEstadisticas de las variables del modelo despues del tratamiento:")
print(df[VARIABLES_AMPLIADAS].describe().round(3).T[["min", "mean", "max"]])

# 5. Codificaciones: todas las variables son numericas; no se requiere
#    codificar. La calidad se conserva solo para validar los grupos.
print("\n=== CODIFICACION ===")
print("No hay variables de texto; no se requiere codificacion.")

# 6. Guardar dataset limpio
LIMPIO.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(LIMPIO, index=False)
print("\nDataset limpio guardado:", LIMPIO.name, df.shape)

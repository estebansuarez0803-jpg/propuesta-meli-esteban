# src/features.py
# -*- coding: utf-8 -*-
"""
Definición de columnas y limpieza para el modelo CatBoost.
"""

import re
import pandas as pd

# ------------------------------
# Definiciones del modelo
# ------------------------------
# ¡OJO! No incluir el target en NUM_COLS
NUM_COLS = [
    "edad",
    "frecuencia_de_compra",
    "promociones_utilizadas",
]
CAT_COLS = [
    "genero",
    "marca_preferida",
]
LABEL = "total_compras"
# Usaremos la cantidad de usuarios como peso de cada fila (si existe)
WEIGHT_COL = "cantidad_usuarios"

# ------------------------------
# Utilidades de normalización
# ------------------------------
def _normalize_colname(col: str) -> str:
    c = str(col).strip()
    c = c.replace('"', " ").replace("\n", " ").replace("\r", " ")
    c = re.sub(r"\s+", " ", c).lower()
    # reemplazo de acentos y mapeos del caso
    tr = (
        ("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),
    )
    for a,b in tr:
        c = c.replace(a,b)
    mapping = {
        "cantidad usuarios": "cantidad_usuarios",
        "edad": "edad",
        "genero": "genero",
        "marca preferida": "marca_preferida",
        "total compras": "total_compras",
        "frecuencia de compra": "frecuencia_de_compra",
        "promociones utilizadas": "promociones_utilizadas",
    }
    return mapping.get(c, c.replace(" ", "_"))

# ------------------------------
# Limpieza principal
# ------------------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normaliza encabezados problemáticos del Excel
    df.columns = [_normalize_colname(c) for c in df.columns]

    # Asegura tipos numéricos
    for c in set(NUM_COLS + [LABEL, WEIGHT_COL]) & set(df.columns):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Limpia categóricas
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()

    # Filtra filas inválidas (target y features esenciales)
    must_have = [LABEL] + NUM_COLS + CAT_COLS
    missing_cols = [c for c in must_have if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas requeridas después de normalizar: {missing_cols}")

    df = df.dropna(subset=[LABEL] + NUM_COLS + CAT_COLS)

    # Target no negativo
    df.loc[df[LABEL] < 0, LABEL] = 0

    return df

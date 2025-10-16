# src/ingest_excel.py
import argparse
import unicodedata
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, String
from dotenv import load_dotenv
import os
import re

# --- Config DB desde .env ---
load_dotenv()
DB_URI = os.getenv("DB_URI")
engine = create_engine(DB_URI, pool_pre_ping=True)

EXPECTED_COLS = {
    "cantidad_usuarios": "INT",
    "edad": "INT",
    "genero": "TEXT",
    "marca_preferida": "TEXT",
    "total_compras": "INT",
    "frecuencia_de_compra": "INT",
    "promociones_utilizadas": "INT",
}

def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def normalize_header(col: str) -> str:
    # limpia comillas, saltos de línea y espacios extra
    c = col.strip().replace('"', ' ').replace("\n", " ").replace("\r", " ")
    c = re.sub(r"\s+", " ", c)
    c = _strip_accents(c).lower()
    # mapeos conocidos del caso
    replacements = {
        "cantidad usuarios": "cantidad_usuarios",
        "edad": "edad",
        "genero": "genero",
        "marca preferida": "marca_preferida",
        "total compras": "total_compras",
        "frecuencia de compra": "frecuencia_de_compra",
        "promociones utilizadas": "promociones_utilizadas",
    }
    return replacements.get(c, c.replace(" ", "_"))

def load_excel_to_df(xlsx_path: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    # lee Excel
    xls = pd.ExcelFile(xlsx_path)
    sheet = sheet_name if sheet_name is not None else xls.sheet_names[0]
    df = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")

    # normaliza encabezados
    df.columns = [normalize_header(str(c)) for c in df.columns]

    # valida columnas mínimas
    missing = [c for c in EXPECTED_COLS.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el Excel: {missing}\nColumnas detectadas: {list(df.columns)}")

    # convierte tipos numéricos
    num_cols = [c for c, t in EXPECTED_COLS.items() if t == "INT"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # limpia filas completamente vacías en claves mínimas
    df = df.dropna(subset=["edad", "genero", "marca_preferida", "total_compras"], how="any")

    # normaliza texto
    for c in ["genero", "marca_preferida"]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    return df

def ensure_table():
    # crea la tabla si no existe, alineada al diccionario del caso
    ddl = """
    CREATE TABLE IF NOT EXISTS clientes (
      cantidad_usuarios INT,
      edad INT,
      genero TEXT,
      marca_preferida TEXT,
      total_compras INT,
      frecuencia_de_compra INT,
      promociones_utilizadas INT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def write_df(df: pd.DataFrame, mode: str):
    # mapeo de dtypes para to_sql
    dtypes = {
        "cantidad_usuarios": Integer(),
        "edad": Integer(),
        "genero": String(),
        "marca_preferida": String(),
        "total_compras": Integer(),
        "frecuencia_de_compra": Integer(),
        "promociones_utilizadas": Integer(),
    }
    if mode not in {"replace", "append"}:
        raise ValueError("mode debe ser 'replace' o 'append'.")

    df.to_sql(
        "clientes",
        engine,
        if_exists=mode,
        index=False,
        dtype=dtypes
    )

def quick_checks():
    # algunas verificaciones rápidas
    sql_sum = "SELECT COUNT(*) AS n, SUM(total_compras) AS tgmv FROM clientes;"
    sql_head = "SELECT * FROM clientes LIMIT 5;"
    with engine.begin() as conn:
        n, tgmv = conn.execute(text(sql_sum)).one()
        hdr = conn.execute(text(sql_head)).fetchall()
    return {"filas": n, "tgmv_sum": tgmv, "muestra": [dict(row._mapping) for row in hdr]}

def main():
    parser = argparse.ArgumentParser(description="Carga Excel a tabla clientes")
    parser.add_argument("--file", required=True, help="Ruta al archivo .xlsx")
    parser.add_argument("--sheet", default=0, help="Nombre o índice de hoja, por defecto 0")
    parser.add_argument("--mode", default="replace", choices=["replace", "append"], help="Estrategia de escritura")
    args = parser.parse_args()

    print("Leyendo Excel…")
    df = load_excel_to_df(args.file, args.sheet)
    print(f"Filas válidas: {len(df)}")

    print("Asegurando tabla…")
    ensure_table()

    print(f"Escribiendo en BD con mode={args.mode}…")
    write_df(df, args.mode)

    info = quick_checks()
    print("OK. Verificación rápida:")
    print(info)

if __name__ == "__main__":
    main()

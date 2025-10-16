# src/data_access.py
import pandas as pd
from sqlalchemy import create_engine, text
from .config import DB_URI  # Importa DB_URI ya cargado desde .env

# Crea el engine una sola vez
engine = create_engine(DB_URI, pool_pre_ping=True, future=True)

def load_clients():
    """
    Carga todos los registros de la tabla 'clientes'.
    """
    query = "SELECT * FROM clientes;"
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df

def upsert_predictions(df_pred: pd.DataFrame, table_name: str = "predicciones"):
    """
    Escribe/actualiza predicciones en la tabla indicada.
    if_exists='replace' para demo; cambia a 'append' si necesitas histórico.
    """
    with engine.begin() as conn:
        df_pred.to_sql(table_name, con=conn, if_exists="replace", index=False)

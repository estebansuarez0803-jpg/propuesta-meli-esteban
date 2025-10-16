# src/config.py
import os
from dotenv import load_dotenv

# Carga variables del .env en el entorno
load_dotenv()

# Lee la URI de la BD. Si no existe, usa SQLite local por defecto.
DB_URI = os.getenv("DB_URI", "sqlite:///meli_local.db")

# Validaciones y ayuda para depurar
if not DB_URI:
    raise RuntimeError(
        "No se encontró DB_URI. Define DB_URI en tu .env o usa SQLite por defecto: "
        "DB_URI=sqlite:///meli_local.db"
    )

# Opcional: imprime una vez para verificar (descomenta si necesitas debug)
# print(f"[config] Usando DB_URI: {DB_URI}")

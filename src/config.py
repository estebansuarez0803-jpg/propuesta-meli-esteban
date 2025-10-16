# -*- coding: utf-8 -*-
"""
Configuración global del proyecto MELI Predictive Dashboard.
Compatibilidad local (dotenv/.env) y Streamlit Cloud (st.secrets).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Intentar cargar secrets de Streamlit (en Cloud no hay .env)
try:
    import streamlit as st
    _secrets = st.secrets  # dict-like
except Exception:
    _secrets = {}

# -------------------------
# Paths base
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # raíz del proyecto (donde está /src)
ENV_PATH = BASE_DIR / ".env"

# Cargar .env solo si existe (entornos locales)
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print("ℹ️  .env no encontrado, se usarán st.secrets / variables de entorno / valores por defecto.")

# -------------------------
# DB_URI con prioridad: secrets → env → SQLite repo
# -------------------------
# Ruta por defecto: data/meli.db dentro del repo (válida en Cloud si lo incluyes en el repo)
default_sqlite_path = (BASE_DIR / "data" / "meli.db").as_posix()
default_sqlite_uri  = f"sqlite:///{default_sqlite_path}"

DB_URI = (
    _secrets.get("DB_URI")              # Streamlit Cloud: Settings → Secrets
    or os.getenv("DB_URI")              # Variables de entorno (local/CI)
    or default_sqlite_uri               # Fallback: SQLite en el repo
)

# -------------------------
# Directorios de trabajo
# -------------------------
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in (DATA_DIR, ARTIFACTS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------------------------
# Parámetros del modelo
# -------------------------
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
KFOLDS = int(os.getenv("KFOLDS", 5))
EARLY_STOPPING_ROUNDS = int(os.getenv("EARLY_STOPPING_ROUNDS", 100))
MAX_EVALS = int(os.getenv("MAX_EVALS", 25))

# -------------------------
# Logging
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# -------------------------
# Verificación manual
# -------------------------
if __name__ == "__main__":
    print("✅ Config cargada")
    print(f"DB_URI: {DB_URI}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"default_sqlite_uri: {default_sqlite_uri}")

# src/pipelines/predict_catboost.py
# -*- coding: utf-8 -*-
"""
Pipeline de predicción con el modelo CatBoost entrenado.
Carga el modelo más reciente, genera predicciones y sube resultados a la BD.
"""

import os
import json
import glob
import argparse
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor, Pool

from src.features import clean_cols, NUM_COLS, CAT_COLS
from src.data_access import load_clients, upsert_predictions
from src.powerbi_api import refresh_dataset

# Directorio de artefactos
ARTIFACTS_DIR = Path("artifacts")


def get_latest_model() -> Path:
    """Devuelve la ruta del modelo CatBoost más reciente."""
    runs = sorted(ARTIFACTS_DIR.glob("*/model_catboost.cbm"), key=os.path.getmtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No se encontró ningún modelo en artifacts/")
    return runs[0]


def load_model(model_path: Path) -> CatBoostRegressor:
    """Carga el modelo CatBoost desde un archivo .cbm"""
    print(f"[INFO] Cargando modelo desde: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def predict_dataset(model: CatBoostRegressor, df: pd.DataFrame) -> pd.DataFrame:
    """Genera predicciones para un DataFrame con las mismas columnas que el entrenamiento."""
    df = clean_cols(df)
    pool = Pool(df[NUM_COLS + CAT_COLS], cat_features=list(range(len(NUM_COLS), len(NUM_COLS) + len(CAT_COLS))))
    preds = model.predict(pool)
    df["pred_total_compras"] = preds
    return df


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones con el modelo CatBoost entrenado.")
    parser.add_argument("--input-csv", help="Ruta a CSV de nuevos clientes a predecir.")
    parser.add_argument("--use-db", action="store_true", help="Usar datos desde la base de datos en lugar de CSV.")
    parser.add_argument("--write-db", action="store_true", help="Guardar las predicciones en la BD (tabla 'predicciones').")
    parser.add_argument("--refresh-pbi", action="store_true", help="Refrescar Power BI al finalizar.")
    args = parser.parse_args()

    # 1) Cargar modelo más reciente
    model_path = get_latest_model()
    model = load_model(model_path)

    # 2) Cargar datos
    if args.use_db:
        df = load_clients()
        print(f"[DATA] Cargados {len(df)} registros desde BD.")
    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        print(f"[DATA] Cargados {len(df)} registros desde {args.input_csv}.")
    else:
        raise ValueError("Debes especificar --use-db o --input-csv")

    # 3) Predecir
    df_pred = predict_dataset(model, df)
    print("[INFO] Predicciones generadas:")
    print(df_pred.head())

    # 4) Guardar resultados
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"predicciones_{Path(model_path).parent.name}.csv"
    df_pred.to_csv(out_path, index=False)
    print(f"[INFO] Archivo de salida: {out_path}")

    # 5) Subir a BD si se indica
    if args.write_db:
        print("[DB] Subiendo predicciones a la base de datos (tabla 'predicciones')...")
        upsert_predictions(df_pred, table_name="predicciones")
        print("[DB] Predicciones guardadas correctamente.")

    # 6) Refrescar Power BI si se indica
    if args.refresh_pbi:
        try:
            print("[PowerBI] Refrescando dataset...")
            result = refresh_dataset()
            print("[PowerBI] Refresh iniciado:", json.dumps(result, indent=2))
        except Exception as e:
            print(f"[PowerBI] Error al refrescar dataset: {e}")

    print("\n✅ Proceso de scoring finalizado.\n")


if __name__ == "__main__":
    main()

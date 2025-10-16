# src/tools/train_exhaustive.py
# -*- coding: utf-8 -*-
"""
Entrenamiento exhaustivo (CatBoost Regressor) con:
- Limpieza y validaciones
- KFold CV + early stopping
- Random Search reproducible
- Métricas globales, por segmento y por deciles
- Importancias (PredictionValuesChange y Gain-like)
- Guardado de artefactos con metadatos de la corrida
- CLI para controlar snapshot/BD, semillas, K, MAX_EVALS, etc.

Requiere:
  - src.data_access: load_clients, upsert_predictions
  - src.features: clean_cols, NUM_COLS, CAT_COLS, LABEL, WEIGHT_COL
"""

import argparse
import json
import math
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from src.data_access import load_clients, upsert_predictions
from src.features import clean_cols, NUM_COLS, CAT_COLS, LABEL, WEIGHT_COL

# ---------------------------------------------------------------------
# Configs por defecto (tuneables via CLI)
# ---------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEED = 42
DEFAULT_KFOLDS = 5
DEFAULT_EARLY_STOP = 100
DEFAULT_MAX_EVALS = 25
DEFAULT_TEST_SIZE = 0.2
PRIMARY_METRIC = "R2"  # métrica para elegir mejor configuración (R2 | -MAE | -RMSE | -MAPE)

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def now_id():
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

def as_float(x):
    return float(x) if x is not None else None

def safe_clip_nonnegative(arr):
    # Para evitar predicciones negativas en un target que debería ser >=0
    return np.maximum(arr, 0.0)

# ---------------------------------------------------------------------
# Espacio de hiperparámetros para Random Search
# ---------------------------------------------------------------------
def base_param_space(seed: int, max_evals: int) -> List[Dict]:
    """
    Espacio aleatorio reproducible de hiperparámetros CatBoost.
    """
    rng = np.random.default_rng(seed)
    space = []
    for _ in range(max_evals):
        space.append(dict(
            depth=int(rng.integers(4, 9)),                      # 4-8
            learning_rate=float(rng.uniform(0.03, 0.2)),        # 0.03-0.2
            l2_leaf_reg=float(10 ** rng.uniform(-1, 2)),        # 0.1 - 100
            bagging_temperature=float(rng.uniform(0.0, 1.0)),   # 0-1
            random_strength=float(rng.uniform(0.0, 1.0)),       # 0-1
            border_count=int(rng.integers(32, 256)),            # 32-255
            n_estimators=int(rng.integers(400, 1401)),          # 400-1400
            # parámetros adicionales útiles:
            # grow_policy="SymmetricTree",  # default
            # thread_count=-1,              # usar todos los cores disponibles
        ))
    return space

# ---------------------------------------------------------------------
# Preparación de datos
# ---------------------------------------------------------------------
def make_pools(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[int]]:
    """
    Devuelve X (features), y (label), w (pesos) y cat_idx (índices categóricas).
    """
    df = clean_cols(df)

    # Validaciones mínimas
    cols_needed = set(NUM_COLS + CAT_COLS + [LABEL])
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Etiqueta no negativa
    if (df[LABEL] < 0).any():
        # si hubiera valores negativos (no esperado), cap a 0
        df.loc[df[LABEL] < 0, LABEL] = 0

    X = df[NUM_COLS + CAT_COLS].reset_index(drop=True)
    y = df[LABEL].values.astype(float)
    w = df[WEIGHT_COL].values.astype(float) if WEIGHT_COL in df.columns else np.ones(len(df), dtype=float)
    cat_idx = list(range(len(NUM_COLS), len(NUM_COLS) + len(CAT_COLS)))
    return X, y, w, cat_idx

# ---------------------------------------------------------------------
# Entrenamiento con CV
# ---------------------------------------------------------------------
def fit_cv(
    params: Dict,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    cat_idx: List[int],
    n_splits: int,
    early_stopping_rounds: int,
    seed: int,
) -> Dict:
    """
    KFold CV con early stopping. Devuelve métricas promedio y modelos por fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s, maes, rmses, mapes = [], [], [], []
    models, oof_preds = [], np.zeros_like(y, dtype=float)

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        w_tr, w_va = w[tr], w[va]

        model = CatBoostRegressor(
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            **params
        )

        pool_tr = Pool(X_tr, label=y_tr, weight=w_tr, cat_features=cat_idx)
        pool_va = Pool(X_va, label=y_va, weight=w_va, cat_features=cat_idx)

        model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=early_stopping_rounds)

        pred_va = safe_clip_nonnegative(model.predict(pool_va))
        oof_preds[va] = pred_va

        r2s.append(r2_score(y_va, pred_va, sample_weight=w_va))
        maes.append(mean_absolute_error(y_va, pred_va, sample_weight=w_va))
        rmses.append(math.sqrt(mean_squared_error(y_va, pred_va, sample_weight=w_va)))
        mapes.append(mape(y_va, pred_va))

        models.append(model)

    metrics = {
        "R2_mean":  as_float(np.mean(r2s)),
        "R2_std":   as_float(np.std(r2s)),
        "MAE_mean": as_float(np.mean(maes)),
        "MAE_std":  as_float(np.std(maes)),
        "RMSE_mean":as_float(np.mean(rmses)),
        "RMSE_std": as_float(np.std(rmses)),
        "MAPE_mean":as_float(np.mean(mapes)),
        "MAPE_std": as_float(np.std(mapes)),
        "oof_preds": oof_preds,
        "folds": len(r2s),
    }
    return {"metrics": metrics, "models": models}

# ---------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------
def pick_best(row_a: Dict, row_b: Dict, primary: str = PRIMARY_METRIC) -> Dict:
    """
    Selecciona la mejor fila según métrica primaria:
      - 'R2' (maximizar)
      - '-MAE' / '-RMSE' / '-MAPE' (minimizar)
    """
    if row_b is None:
        return row_a
    a, b = row_a[primary], row_b[primary]
    return row_a if a > b else row_b

def random_search(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    cat_idx: List[int],
    space: List[Dict],
    max_trials: int,
    n_splits: int,
    early_stopping_rounds: int,
    seed: int,
    primary_metric: str = PRIMARY_METRIC,
):
    results = []
    best = None
    for i, params in enumerate(space[:max_trials], 1):
        start = time.time()
        cv_res = fit_cv(params, X, y, w, cat_idx, n_splits, early_stopping_rounds, seed)
        duration = time.time() - start

        row = {
            "trial": i,
            "params": params,
            "R2":   cv_res["metrics"]["R2_mean"],
            "-MAE": -cv_res["metrics"]["MAE_mean"],
            "-RMSE":-cv_res["metrics"]["RMSE_mean"],
            "-MAPE":-cv_res["metrics"]["MAPE_mean"],
            "time_sec": duration,
        }
        results.append(row)
        best = pick_best(row, best, primary=primary_metric)

        print(f"[{i}/{max_trials}] "
              f"R2={row['R2']:.4f} | MAE={-row['-MAE']:.2f} | "
              f"RMSE={-row['-RMSE']:.2f} | MAPE={-row['-MAPE']:.4f} | {duration:.1f}s")

    results_df = pd.DataFrame(results).sort_values(primary_metric, ascending=False).reset_index(drop=True)
    return best, results_df

# ---------------------------------------------------------------------
# Entrenamiento final + diagnósticos
# ---------------------------------------------------------------------
def train_final_model(X, y, w, cat_idx, best_params, seed: int) -> CatBoostRegressor:
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        **best_params
    )
    pool = Pool(X, label=y, weight=w, cat_features=cat_idx)
    model.fit(pool)
    return model

def segment_metrics(df_feats, y_true, y_pred, w, seg_cols: List[str]) -> pd.DataFrame:
    tmp = df_feats.copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    tmp["w"] = w
    out = []
    for cols, grp in tmp.groupby(seg_cols):
        if not isinstance(cols, tuple):
            cols = (cols,)
        r2  = r2_score(grp["y_true"], grp["y_pred"], sample_weight=grp["w"])
        mae = mean_absolute_error(grp["y_true"], grp["y_pred"], sample_weight=grp["w"])
        rmse= math.sqrt(mean_squared_error(grp["y_true"], grp["y_pred"], sample_weight=grp["w"]))
        out.append(dict(**{f"seg_{i}": v for i, v in enumerate(cols)},
                        R2=r2, MAE=mae, RMSE=rmse, n=int(len(grp))))
    return pd.DataFrame(out).sort_values("R2", ascending=False)

def decile_diagnostics(y_true, y_pred, w, n_deciles=10) -> pd.DataFrame:
    """Diagnóstico por deciles de predicción para evaluar calibración (sin warnings de pandas)."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "w": w})
    # Orden estable y corte en deciles por score
    ranks = df["y_pred"].rank(method="first")
    df["decile"] = pd.qcut(ranks, q=n_deciles, labels=list(range(1, n_deciles + 1)))

    # Agregación con medias ponderadas (sin .apply)
    def wmean(xv, wv):
        xv = np.asarray(xv, dtype=float)
        wv = np.asarray(wv, dtype=float)
        return float((xv * wv).sum() / max(wv.sum(), 1.0))

    agg = df.groupby("decile", observed=True).apply(
        lambda g: pd.Series({
            "n": int(len(g)),
            "y_true_mean": wmean(g["y_true"], g["w"]),
            "y_pred_mean": wmean(g["y_pred"], g["w"]),
        }),
        include_groups=False  # Pandas >= 2.2
    ).reset_index()

    agg["bias"] = agg["y_pred_mean"] - agg["y_true_mean"]
    return agg.sort_values("decile")


def feature_importances_catboost(model: CatBoostRegressor, X, y, w, cat_idx) -> pd.DataFrame:
    pool = Pool(X, label=y, weight=w, cat_features=cat_idx)
    pvc  = model.get_feature_importance(type="PredictionValuesChange", data=pool)
    gain = model.get_feature_importance(type="FeatureImportance",        data=pool)
    cols = list(X.columns)
    return pd.DataFrame({"feature": cols,
                         "importance_pvc": pvc,
                         "importance_gain": gain}).sort_values("importance_pvc", ascending=False)

# ---------------------------------------------------------------------
# Guardado de artefactos
# ---------------------------------------------------------------------
def save_artifacts(
    run_id: str,
    cfg: Dict,
    best: Dict,
    cv_table: pd.DataFrame,
    global_metrics: Dict,
    seg_brand: pd.DataFrame,
    seg_gender: pd.DataFrame,
    deciles: pd.DataFrame,
    fi: pd.DataFrame,
    model: CatBoostRegressor,
):
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Modelo
    model_path = run_dir / "model_catboost.cbm"
    model.save_model(str(model_path))

    # Tablas
    cv_table.to_csv(run_dir / "cv_results.csv", index=False)
    seg_brand.to_csv(run_dir / "segment_metrics_brand.csv", index=False)
    seg_gender.to_csv(run_dir / "segment_metrics_gender.csv", index=False)
    deciles.to_csv(run_dir / "deciles_diagnostics.csv", index=False)
    fi.to_csv(run_dir / "feature_importances.csv", index=False)

    # Métricas + config + entorno
    payload = {
        "run_id": run_id,
        "best_params": best["params"],
        "best_cv_row": {k: v for k, v in best.items() if k != "params"},
        "global_metrics": global_metrics,
        "config": cfg,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": cfg["seed"],
            "kfolds": cfg["kfolds"],
        }
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nArtefactos guardados en: {run_dir}")
    print("• Modelo:", model_path)
    print("• CV results:", run_dir / "cv_results.csv")
    print("• Importancias:", run_dir / "feature_importances.csv")
    print("• Métricas x marca:", run_dir / "segment_metrics_brand.csv")
    print("• Métricas x género:", run_dir / "segment_metrics_gender.csv")
    print("• Deciles:", run_dir / "deciles_diagnostics.csv")
    print("• metrics.json:", run_dir / "metrics.json")

# ---------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Entrenamiento exhaustivo CatBoost (MELI).")
    p.add_argument("--use-snapshot", action="store_true", help="Usar snapshot CSV en vez de cargar desde BD.")
    p.add_argument("--snapshot-path", default="data/train_snapshot.csv", help="Ruta al snapshot CSV.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--kfolds", type=int, default=DEFAULT_KFOLDS)
    p.add_argument("--early-stop", type=int, default=DEFAULT_EARLY_STOP)
    p.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--primary", choices=["R2", "-MAE", "-RMSE", "-MAPE"], default=PRIMARY_METRIC,
                   help="Métrica primaria para elegir la mejor configuración.")
    p.add_argument("--no-upload-test", action="store_true", help="No subir tabla de predicciones de test a BD.")
    args = p.parse_args()

    cfg = dict(
        seed=args.seed,
        kfolds=args.kfolds,
        early_stop=args.early_stop,
        max_evals=args.max_evals,
        test_size=args.test_size,
        primary_metric=args.primary,
        use_snapshot=args.use_snapshot,
        snapshot_path=args.snapshot_path,
    )

    # ------------------ 1) Carga ------------------
    if args.use_snapshot and Path(args.snapshot_path).exists():
        df = pd.read_csv(args.snapshot_path)
        print(f"[DATA] Snapshot: {args.snapshot_path} ({len(df)} filas)")
    else:
        df = load_clients()
        print(f"[DATA] BD: {len(df)} filas")

    # ------------------ 2) Preparación ------------------
    X, y, w, cat_idx = make_pools(df)

    # ------------------ 3) Random Search + CV ------------------
    space = base_param_space(seed=args.seed, max_evals=args.max_evals)
    best, cv_df = random_search(
        X, y, w, cat_idx, space,
        max_trials=args.max_evals,
        n_splits=args.kfolds,
        early_stopping_rounds=args.early_stop,
        seed=args.seed,
        primary_metric=args.primary
    )

    print("\n[MEJOR CONFIG]:")
    print(json.dumps(best, indent=2, ensure_ascii=False))

        # ------------------ 4) Split Train/Test antes del entrenamiento final ------------------
    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, w, test_size=args.test_size, random_state=args.seed
    )

    # Hacemos un pequeño split interno de validación para early stopping (no usar test para early stop)
    from sklearn.model_selection import train_test_split as _tts
    X_tr2, X_va, y_tr2, y_va, w_tr2, w_va = _tts(
        X_tr, y_tr, w_tr, test_size=0.15, random_state=args.seed
    )

    pool_tr = Pool(X_tr2, label=y_tr2, weight=w_tr2, cat_features=cat_idx)
    pool_va = Pool(X_va,  label=y_va,  weight=w_va,  cat_features=cat_idx)

    # ------------------ 5) Entrenamiento final SOLO con TRAIN ------------------
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=args.seed,
        verbose=False,
        **best["params"]
    )
    model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=args.early_stop)

    # Evaluación en TEST (nunca visto)
    pool_te = Pool(X_te, label=y_te, weight=w_te, cat_features=cat_idx)
    y_hat = model.predict(pool_te)
    y_hat = np.maximum(y_hat, 0.0)  # si tu target no puede ser negativo

    global_metrics = {
        "R2_test":   float(r2_score(y_te, y_hat, sample_weight=w_te)),
        "MAE_test":  float(mean_absolute_error(y_te, y_hat, sample_weight=w_te)),
        "RMSE_test": float(math.sqrt(mean_squared_error(y_te, y_hat, sample_weight=w_te))),
        "MAPE_test": float(mape(y_te, y_hat)),
    }
    print("\n[MÉTRICAS TEST]:", json.dumps(global_metrics, indent=2, ensure_ascii=False))


    # ------------------ 6) Métricas por segmento ------------------
    seg_brand  = segment_metrics(X_te.copy(), y_te, y_hat, w_te, seg_cols=["marca_preferida"])
    seg_gender = segment_metrics(X_te.copy(), y_te, y_hat, w_te, seg_cols=["genero"])

    # ------------------ 7) Deciles ------------------
    deciles = decile_diagnostics(y_te, y_hat, w_te, n_deciles=10)

    # ------------------ 8) Importancias ------------------
    fi = feature_importances_catboost(model, X, y, w, cat_idx)

    # ------------------ 9) Guardado de artefactos ------------------
    run_id = now_id()
    save_artifacts(run_id, cfg, best, cv_df, global_metrics, seg_brand, seg_gender, deciles, fi, model)

    # ------------------ 10) (Opcional) Subir predicciones de test a BD ------------------
    if not args.no_upload_test:
        preds_out = X_te.copy()
        preds_out[LABEL] = y_te
        preds_out["pred_total_compras"] = y_hat
        preds_out["weight"] = w_te
        upsert_predictions(preds_out, table_name="predicciones_test")
        print("Predicciones de test subidas a BD (tabla: predicciones_test).")

if __name__ == "__main__":
    main()

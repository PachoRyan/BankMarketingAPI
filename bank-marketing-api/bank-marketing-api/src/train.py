"""
src/train.py
============
PARTE 1 (análisis y modelado) + PARTE 2 (entrenamiento final) de la actividad.

Qué hace este script, en orden:

  1. Carga el Bank Marketing Dataset (UCI) y hace una exploración básica.
  2. ELIMINA `duration` (restricción crítica: es data leakage).
  3. Separa train/test de forma estratificada (80/20).
  4. Construye un pipeline de preprocesamiento (imputación no necesaria: no hay
     nulos, pero sí escalado + one-hot) y entrena 3 modelos con búsqueda de
     hiperparámetros por validación cruzada.
  5. Selecciona el mejor modelo por PR-AUC (average precision) en CV y ajusta el
     UMBRAL de decisión para maximizar F1 sobre predicciones out-of-fold.
  6. Evalúa el mejor modelo en el test set (F1, balanced accuracy, ROC-AUC,
     PR-AUC, matriz de confusión).
  7. REENTRENA el modelo ganador con TODOS los datos y los mejores
     hiperparámetros, y guarda `models/model.joblib` + metadatos + métricas.

Uso:
    python src/train.py
    python src/train.py --data data/bank-additional-full.csv --seed 42
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Permite ejecutar `python src/train.py` desde la raíz del repositorio.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RAW_FEATURES,
    TARGET,
    TARGET_MAP,
    build_features,
)

MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def cargar_datos(path: Path) -> pd.DataFrame:
    """Lee el CSV de UCI (separador ';') y hace una limpieza mínima."""
    df = pd.read_csv(path, sep=";")
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Se eliminaron {n_dup} filas duplicadas exactas.")
    return df


def explorar(df: pd.DataFrame) -> None:
    """Exploración rápida del dataset (Parte 1)."""
    titulo("1. EXPLORACIÓN DEL DATASET")
    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    print(f"Valores nulos totales: {int(df.isna().sum().sum())}")

    dist = df[TARGET].value_counts(normalize=True).mul(100).round(2)
    print("\nDistribución de la variable objetivo (%):")
    print(dist.to_string())
    print(
        "-> El dataset está MUY desbalanceado (~11% de 'yes'). Por eso no usamos "
        "accuracy como métrica principal, sino F1 / balanced accuracy / PR-AUC."
    )

    # 'unknown' funciona como valor faltante enmascarado en varias categóricas.
    print("\nPorcentaje de 'unknown' por variable categórica:")
    for col in CATEGORICAL_FEATURES:
        pct = (df[col] == "unknown").mean() * 100
        if pct > 0:
            print(f"  {col:<14} {pct:5.2f}%")
    print(
        "-> Tratamos 'unknown' como una CATEGORÍA más (no lo imputamos): que un "
        "dato falte suele ser informativo en campañas telefónicas."
    )

    print("\nTasa de aceptación por algunas variables clave:")
    for col in ["poutcome", "contact", "month", "job"]:
        tasa = (
            df.groupby(col, observed=True)[TARGET]
            .apply(lambda s: (s == "yes").mean() * 100)
            .sort_values(ascending=False)
            .round(2)
        )
        print(f"\n  -- {col} --")
        print(tasa.to_string())


def analizar_leakage(df: pd.DataFrame) -> None:
    """Parte 1: análisis explícito de posibles fuentes de data leakage."""
    titulo("2. ANÁLISIS DE FUENTES DE LEAKAGE")
    corr = df["duration"].corr((df[TARGET] == "yes").astype(int))
    print(f"Correlación de `duration` con el target: {corr:.3f}")
    print(
        "\n(a) `duration` -> LEAKAGE CLARO. Solo se conoce cuando la llamada ya\n"
        "    terminó; si la llamada dura 0 segundos el resultado es siempre 'no'.\n"
        "    En producción, al decidir a quién llamar, ese valor no existe.\n"
        "    ACCIÓN: se elimina del dataset antes de cualquier split.\n"
        "\n(b) Variables macroeconómicas (emp.var.rate, cons.price.idx,\n"
        "    cons.conf.idx, euribor3m, nr.employed) -> NO son leakage: se publican\n"
        "    antes de la llamada y en producción se conocen. Sí introducen\n"
        "    dependencia temporal: el modelo aprende el contexto económico\n"
        "    2008-2010, así que debe reentrenarse periódicamente.\n"
        "    ACCIÓN: se conservan, pero se documenta el riesgo de drift.\n"
        "\n(c) `pdays` / `previous` / `poutcome` -> historial de campañas ANTERIORES,\n"
        "    conocido antes de llamar. No es leakage.\n"
        "\n(d) Split: se estratifica por el target y todo el preprocesamiento\n"
        "    (escalado, one-hot) va DENTRO del Pipeline, de modo que se ajusta solo\n"
        "    con los datos de entrenamiento de cada fold. Esto evita el leakage\n"
        "    silencioso de escalar con estadísticas del test."
    )


def construir_preprocesador() -> ColumnTransformer:
    """Escalado para numéricas + One-Hot para categóricas.

    `handle_unknown='ignore'` hace que una categoría nunca vista en producción
    no rompa la API (se codifica como todo ceros).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def definir_modelos(seed: int) -> dict:
    """Modelos candidatos + su grid de hiperparámetros.

    Se entrenan 3 familias distintas a propósito:
      * Regresión logística: baseline lineal, interpretable.
      * Random Forest: no lineal, robusto, poco sensible a escalas.
      * HistGradientBoosting: boosting moderno, suele ganar en datos tabulares.
    """
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
            "grid": {"clf__C": [0.1, 1.0, 10.0]},
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            "grid": {
                "clf__max_depth": [10, 20],
                "clf__min_samples_leaf": [5, 20],
            },
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            ),
            "grid": {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_leaf_nodes": [31, 63],
            },
        },
    }


def umbral_optimo(y_true: np.ndarray, p_yes: np.ndarray) -> tuple[float, float]:
    """Busca el umbral que maximiza F1 sobre predicciones out-of-fold."""
    mejores = (0.5, -1.0)
    for thr in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (p_yes >= thr).astype(int))
        if f1 > mejores[1]:
            mejores = (float(thr), float(f1))
    return mejores


def metricas(y_true, y_pred, p_yes) -> dict:
    return {
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, p_yes)), 4),
        "pr_auc": round(float(average_precision_score(y_true, p_yes)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento Bank Marketing (UCI)")
    parser.add_argument("--data", default=str(ROOT / "data" / "bank-additional-full.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=3, help="folds de validación cruzada")
    args = parser.parse_args()

    t0 = time.time()
    rng = args.seed
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    # ---- Carga + exploración -------------------------------------------------
    df = cargar_datos(Path(args.data))
    explorar(df)
    analizar_leakage(df)

    # ---- Restricción crítica: fuera `duration` -------------------------------
    titulo("3. PREPARACIÓN DE DATOS")
    df = df.drop(columns=[c for c in FORBIDDEN_FEATURES if c in df.columns])
    assert not any(c in df.columns for c in FORBIDDEN_FEATURES), "¡duration sigue presente!"
    print(f"Variables prohibidas eliminadas: {FORBIDDEN_FEATURES}")

    X = build_features(df[RAW_FEATURES])
    y = df[TARGET].map(TARGET_MAP).astype(int)
    print(f"Features finales ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=rng
    )
    print(f"Train: {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas (estratificado)")

    # ---- Entrenamiento y comparación ----------------------------------------
    titulo("4. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS")
    print(
        "Métrica de selección: PR-AUC (average precision).\n"
        "Motivo: con 11% de positivos, el ROC-AUC es optimista y el F1 depende del\n"
        "umbral. El PR-AUC resume la calidad del ranking sobre la clase minoritaria\n"
        "sin depender de un umbral fijo; el umbral se ajusta después."
    )

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=rng)
    preprocesador = construir_preprocesador()
    resultados = []
    busquedas = {}

    for nombre, cfg in definir_modelos(rng).items():
        print(f"\n--- {nombre} ---")
        pipe = Pipeline([("prep", preprocesador), ("clf", cfg["estimator"])])
        gs = GridSearchCV(
            pipe,
            param_grid=cfg["grid"],
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        t = time.time()
        gs.fit(X_train, y_train)
        print(f"Mejores hiperparámetros: {gs.best_params_}")
        print(f"PR-AUC (CV): {gs.best_score_:.4f}  [{time.time() - t:.1f}s]")
        busquedas[nombre] = gs
        resultados.append(
            {"modelo": nombre, "pr_auc_cv": round(float(gs.best_score_), 4),
             "mejores_params": gs.best_params_}
        )

    tabla = pd.DataFrame(resultados).sort_values("pr_auc_cv", ascending=False)
    print("\nComparación (PR-AUC en validación cruzada):")
    print(tabla.to_string(index=False))

    mejor_nombre = str(tabla.iloc[0]["modelo"])
    mejor_gs = busquedas[mejor_nombre]
    print(f"\n>>> Modelo seleccionado: {mejor_nombre}")

    # ---- Ajuste del umbral con predicciones out-of-fold ----------------------
    titulo("5. AJUSTE DEL UMBRAL DE DECISIÓN")
    p_oof = cross_val_predict(
        mejor_gs.best_estimator_, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    thr, f1_oof = umbral_optimo(y_train.to_numpy(), p_oof)
    print(
        f"Umbral óptimo (maximiza F1 out-of-fold): {thr:.2f}  ->  F1 = {f1_oof:.4f}\n"
        f"(el umbral por defecto 0.50 daría F1 = {f1_score(y_train, (p_oof >= 0.5).astype(int)):.4f})\n"
        "Se elige el umbral con datos out-of-fold, NUNCA con el test set."
    )

    # ---- Evaluación en test --------------------------------------------------
    titulo("6. EVALUACIÓN EN EL CONJUNTO DE TEST (datos nunca vistos)")
    p_test = mejor_gs.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= thr).astype(int)
    m_test = metricas(y_test, y_pred, p_test)
    for k, v in m_test.items():
        print(f"  {k:<18} {v}")
    print("\nMatriz de confusión [filas=real, columnas=predicho]:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=["real_no", "real_yes"], columns=["pred_no", "pred_yes"]).to_string())
    print("\n" + classification_report(y_test, y_pred, target_names=["no", "yes"], digits=3))

    # Línea base tonta, para dimensionar la mejora.
    base_f1 = f1_score(y_test, np.ones_like(y_test))
    print(f"Referencia: predecir siempre 'yes' daría F1 = {base_f1:.4f} (y sería inútil).")

    # ---- PARTE 2: reentrenamiento final con TODOS los datos ------------------
    titulo("7. ENTRENAMIENTO FINAL (PARTE 2): reentrenar con el 100% de los datos")
    mejores_params = {k.replace("clf__", ""): v for k, v in mejor_gs.best_params_.items()}
    estimador_final = definir_modelos(rng)[mejor_nombre]["estimator"].set_params(**mejores_params)
    pipeline_final = Pipeline([("prep", construir_preprocesador()), ("clf", estimador_final)])
    pipeline_final.fit(X, y)
    print(
        f"Pipeline final entrenado con {X.shape[0]:,} filas y los hiperparámetros "
        f"{mejores_params}.\nSe reutiliza el umbral {thr:.2f} hallado out-of-fold."
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "model_name": mejor_nombre,
        "best_params": mejores_params,
        "raw_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "excluded_features": FORBIDDEN_FEATURES,
        "target_map": TARGET_MAP,
        "n_rows_final_fit": int(X.shape[0]),
        "decision_threshold": round(thr, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset": "Bank Marketing (UCI) - bank-additional-full.csv",
    }
    metricas_json = {
        "seleccion_cv": tabla.to_dict(orient="records"),
        "umbral": {"valor": round(thr, 4), "f1_out_of_fold": round(f1_oof, 4)},
        "test_holdout": m_test,
        "matriz_confusion_test": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }

    artefacto = {
        "pipeline": pipeline_final,
        "threshold": float(thr),
        "metadata": metadata,
        "metrics": metricas_json,
    }
    ruta_modelo = ROOT / "models" / "model.joblib"
    joblib.dump(artefacto, ruta_modelo, compress=3)
    (ROOT / "models" / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "models" / "metrics.json").write_text(
        json.dumps(metricas_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tabla.to_csv(ROOT / "reports" / "comparacion_modelos.csv", index=False)

    titulo("8. ARTEFACTOS GUARDADOS")
    print(f"  models/model.joblib    ({ruta_modelo.stat().st_size / 1e6:.2f} MB)")
    print("  models/metadata.json")
    print("  models/metrics.json")
    print("  reports/comparacion_modelos.csv")
    print(f"\nTiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

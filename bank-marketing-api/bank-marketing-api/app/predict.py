"""
app/predict.py
==============
PARTE 3 de la actividad — "Pipeline reproducible".

Expone la función:

    predict(input_dict) -> dict

que:
  * recibe los datos de un cliente en formato diccionario,
  * VALIDA los inputs (campos faltantes, tipos, rangos, categorías desconocidas),
  * devuelve la predicción, las probabilidades y un nivel de confianza.

Esta función es el único punto de entrada al modelo: la API (app/main.py) la
llama tal cual, y el script cliente prueba exactamente la misma lógica. Si
mañana cambia el preprocesamiento, cambia en un solo lugar.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib

from app.features import (
    CATEGORY_VALUES,
    NUMERIC_RANGES,
    NUMERIC_RAW,
    RAW_FEATURES,
    frame_from_dict,
)

# Ruta al artefacto serializado. Se puede sobrescribir con la variable de
# entorno MODEL_PATH (útil en Docker / Cloud Run).
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "model.joblib"))


class InputValidationError(ValueError):
    """Error de validación de entrada. `errors` contiene el detalle por campo."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


# ---------------------------------------------------------------------------
# Carga del modelo (una sola vez por proceso)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model() -> Dict[str, Any]:
    """Carga el artefacto .joblib. Se cachea para no leer el disco en cada request."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta primero: python src/train.py"
        )
    artifact = joblib.load(MODEL_PATH)
    required = {"pipeline", "threshold", "metadata"}
    if not required.issubset(artifact):
        raise ValueError(f"Artefacto inválido; faltan claves: {required - set(artifact)}")
    return artifact


def model_info() -> Dict[str, Any]:
    """Metadatos del modelo cargado (para el endpoint /model-info)."""
    artifact = load_model()
    info = dict(artifact["metadata"])
    info["threshold"] = artifact["threshold"]
    return info


# ---------------------------------------------------------------------------
# Validación de inputs
# ---------------------------------------------------------------------------

def validate_input(input_dict: Any) -> Dict[str, Any]:
    """Valida el diccionario de entrada y devuelve una copia limpia y tipada.

    Reglas aplicadas:
      1. La entrada debe ser un diccionario.
      2. Deben estar TODAS las 19 features (no se imputan campos faltantes:
         preferimos fallar rápido antes que inventar datos en producción).
      3. `duration` (y cualquier columna extra) se ignora y se reporta como aviso.
      4. Las numéricas deben ser convertibles a float y caer dentro de un rango
         plausible.
      5. Las categóricas deben pertenecer al conjunto de valores conocidos
         (se normalizan a minúsculas y sin espacios antes de comparar).
    """
    if not isinstance(input_dict, dict):
        raise InputValidationError(["La entrada debe ser un diccionario (JSON object)."])

    errors: List[str] = []
    clean: Dict[str, Any] = {}

    faltantes = [c for c in RAW_FEATURES if c not in input_dict]
    if faltantes:
        errors.append(f"Faltan campos obligatorios: {faltantes}")

    for col in RAW_FEATURES:
        if col not in input_dict:
            continue
        value = input_dict[col]

        if value is None:
            errors.append(f"'{col}': no se admiten valores nulos.")
            continue

        if col in NUMERIC_RAW:
            if isinstance(value, bool):
                errors.append(f"'{col}': se esperaba un número, se recibió un booleano.")
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                errors.append(f"'{col}': se esperaba un número, se recibió {value!r}.")
                continue
            lo, hi = NUMERIC_RANGES[col]
            if not (lo <= num <= hi):
                errors.append(f"'{col}': {num} fuera del rango admitido [{lo}, {hi}].")
                continue
            clean[col] = num
        else:
            if not isinstance(value, str):
                errors.append(f"'{col}': se esperaba texto, se recibió {type(value).__name__}.")
                continue
            norm = value.strip().lower()
            permitidos = CATEGORY_VALUES[col]
            if norm not in permitidos:
                errors.append(f"'{col}': valor '{value}' no válido. Admitidos: {permitidos}")
                continue
            clean[col] = norm

    if errors:
        raise InputValidationError(errors)

    return clean


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def _confidence_level(prob: float) -> str:
    """Traduce la probabilidad de la clase predicha a un nivel legible."""
    if prob >= 0.80:
        return "alta"
    if prob >= 0.60:
        return "media"
    return "baja"


def predict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predice si un cliente aceptará el depósito a plazo.

    Parámetros
    ----------
    input_dict : dict
        Diccionario con las 19 features de entrada (ver app/features.RAW_FEATURES).

    Retorna
    -------
    dict con:
        prediction        : "yes" | "no"
        prediction_label  : descripción en español
        probabilities     : {"no": float, "yes": float}
        confidence        : probabilidad de la clase predicha
        confidence_level  : "alta" | "media" | "baja"
        threshold         : umbral de decisión usado
        model_version     : versión del artefacto
        warnings          : lista de avisos (campos ignorados, p. ej. `duration`)

    Lanza
    -----
    InputValidationError si la entrada no es válida.
    """
    artifact = load_model()
    clean = validate_input(input_dict)

    warnings: List[str] = []
    extras = [k for k in input_dict if k not in RAW_FEATURES]
    if extras:
        warnings.append(
            f"Campos ignorados (no forman parte del modelo): {extras}"
        )
    if "duration" in extras:
        warnings.append(
            "'duration' se ignora de forma intencional: solo se conoce después de "
            "la llamada y usarla sería data leakage."
        )

    X = frame_from_dict(clean)
    proba = artifact["pipeline"].predict_proba(X)[0]

    # El pipeline se entrenó con y ∈ {0, 1}; classes_ = [0, 1].
    classes = list(artifact["pipeline"].classes_)
    p_yes = float(proba[classes.index(1)])
    p_no = float(proba[classes.index(0)])

    threshold = float(artifact["threshold"])
    pred = "yes" if p_yes >= threshold else "no"
    confidence = p_yes if pred == "yes" else p_no

    return {
        "prediction": pred,
        "prediction_label": (
            "Contratará el depósito a plazo" if pred == "yes"
            else "No contratará el depósito a plazo"
        ),
        "probabilities": {"no": round(p_no, 4), "yes": round(p_yes, 4)},
        "confidence": round(float(confidence), 4),
        "confidence_level": _confidence_level(confidence),
        "threshold": round(threshold, 4),
        "model_version": artifact["metadata"].get("model_version", "n/a"),
        "warnings": warnings,
    }


def predict_batch(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Versión por lotes: aplica `predict` a una lista de diccionarios."""
    return [predict(r) for r in rows]


if __name__ == "__main__":  # prueba rápida desde la terminal
    ejemplo = {
        "age": 35, "job": "technician", "marital": "single", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "campaign": 1, "pdays": 999, "previous": 0,
        "poutcome": "nonexistent", "emp.var.rate": -1.8, "cons.price.idx": 92.893,
        "cons.conf.idx": -46.2, "euribor3m": 1.313, "nr.employed": 5099.1,
    }
    print(json.dumps(predict(ejemplo), indent=2, ensure_ascii=False))

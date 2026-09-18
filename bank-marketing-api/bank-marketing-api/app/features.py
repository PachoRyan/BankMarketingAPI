"""
app/features.py
===============
Contrato de datos ÚNICO del proyecto.

Este módulo es la "fuente de la verdad" sobre qué variables entran al modelo,
qué valores admite cada una y qué transformaciones deterministas se aplican
antes del pipeline de scikit-learn.

Lo usan TANTO el entrenamiento (src/train.py) COMO la API (app/predict.py),
de modo que es imposible que el modelo se entrene con un conjunto de columnas
y la API envíe otro (una de las causas más comunes de training/serving skew).

IMPORTANTE (restricción crítica de la actividad):
    La variable `duration` NO aparece en ninguna lista de este archivo.
    Solo se conoce DESPUÉS de terminar la llamada, así que usarla sería
    data leakage: en producción, en el momento de decidir a quién llamar,
    ese valor todavía no existe.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Columnas que la API recibe del usuario (features "crudas")
# ---------------------------------------------------------------------------

# Variables numéricas tal como vienen en el CSV de UCI.
NUMERIC_RAW: List[str] = [
    "age",             # edad del cliente
    "campaign",        # nº de contactos en esta campaña (incluye el actual)
    "pdays",           # días desde el último contacto de una campaña previa (999 = nunca)
    "previous",        # nº de contactos previos a esta campaña
    "emp.var.rate",    # tasa de variación del empleo (indicador trimestral)
    "cons.price.idx",  # índice de precios al consumidor (mensual)
    "cons.conf.idx",   # índice de confianza del consumidor (mensual)
    "euribor3m",       # euríbor a 3 meses (diario)
    "nr.employed",     # nº de empleados (indicador trimestral)
]

# Variables categóricas tal como vienen en el CSV de UCI.
CATEGORICAL_RAW: List[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

# Orden canónico de las 19 features de entrada (20 del dataset - `duration`).
RAW_FEATURES: List[str] = NUMERIC_RAW + CATEGORICAL_RAW

# Variable objetivo y su codificación.
TARGET = "y"
TARGET_MAP = {"no": 0, "yes": 1}   # 1 = el cliente SÍ contrata el depósito a plazo

# Variable prohibida: se documenta explícitamente para poder verificarlo en tests.
FORBIDDEN_FEATURES: List[str] = ["duration"]

# ---------------------------------------------------------------------------
# 2. Valores admitidos / rangos plausibles (se usan para validar la entrada)
# ---------------------------------------------------------------------------

CATEGORY_VALUES: Dict[str, List[str]] = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": [
        "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
        "professional.course", "university.degree", "unknown",
    ],
    "default": ["no", "unknown", "yes"],
    "housing": ["no", "unknown", "yes"],
    "loan": ["no", "unknown", "yes"],
    "contact": ["cellular", "telephone"],
    "month": ["apr", "aug", "dec", "jul", "jun", "mar", "may", "nov", "oct", "sep"],
    "day_of_week": ["fri", "mon", "thu", "tue", "wed"],
    "poutcome": ["failure", "nonexistent", "success"],
}

# Rangos plausibles (min, max) para cada numérica. Son algo más amplios que el
# rango observado en el dataset para no rechazar clientes reales atípicos,
# pero sirven para detectar errores obvios (edad negativa, euríbor = 900, ...).
NUMERIC_RANGES: Dict[str, tuple] = {
    "age": (17, 120),
    "campaign": (1, 100),
    "pdays": (0, 999),
    "previous": (0, 50),
    "emp.var.rate": (-5.0, 5.0),
    "cons.price.idx": (85.0, 105.0),
    "cons.conf.idx": (-60.0, 10.0),
    "euribor3m": (0.0, 10.0),
    "nr.employed": (4500.0, 5500.0),
}

# ---------------------------------------------------------------------------
# 3. Feature engineering determinista (sin ajuste / sin fit)
# ---------------------------------------------------------------------------
#
# Se hace FUERA del pipeline de scikit-learn y con funciones puras de pandas
# a propósito: así el artefacto .joblib solo contiene objetos estándar de
# scikit-learn y se puede cargar en el contenedor sin depender de que este
# módulo esté en el mismo path que durante el entrenamiento.

PDAYS_NEVER = 999  # el dataset codifica "nunca contactado antes" como 999

# Columnas que realmente entran al ColumnTransformer:
NUMERIC_FEATURES: List[str] = [
    c for c in NUMERIC_RAW if c != "pdays"
] + ["pdays_clean", "was_contacted_before"]
CATEGORICAL_FEATURES: List[str] = list(CATEGORICAL_RAW)
MODEL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el feature engineering determinista y devuelve el DataFrame final.

    Transformaciones:
      1. `was_contacted_before`: 1 si el cliente ya había sido contactado en una
         campaña anterior, 0 si no. Es la señal útil escondida dentro de `pdays`.
      2. `pdays_clean`: igual que `pdays`, pero el centinela 999 se reemplaza por
         -1. Dejar el 999 mezclaría una "categoría" con una escala de días y
         distorsiona cualquier modelo lineal (y las medias/escalados).
      3. Se elimina `pdays` original y se ordenan las columnas de forma canónica.

    El resultado tiene SIEMPRE las mismas columnas y en el mismo orden, que es
    lo que el pipeline entrenado espera.
    """
    out = df.copy()

    out["was_contacted_before"] = (out["pdays"] != PDAYS_NEVER).astype(int)
    out["pdays_clean"] = out["pdays"].where(out["pdays"] != PDAYS_NEVER, -1)
    out = out.drop(columns=["pdays"])

    # Normalizamos texto de las categóricas (minúsculas y sin espacios sobrantes)
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype(str).str.strip().str.lower()

    return out[MODEL_FEATURES]


def frame_from_dict(data: Dict[str, Any]) -> pd.DataFrame:
    """Convierte un diccionario ya validado en el DataFrame de 1 fila del modelo."""
    row = {k: data[k] for k in RAW_FEATURES}
    return build_features(pd.DataFrame([row]))

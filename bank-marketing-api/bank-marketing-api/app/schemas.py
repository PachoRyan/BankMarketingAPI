"""
app/schemas.py
==============
Contratos de entrada/salida de la API (Pydantic v2).

Detalle importante: tres columnas del dataset original tienen puntos en el
nombre (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `nr.employed`), y un
punto no es válido como nombre de atributo en Python. Se resuelve con `alias`:

  * el JSON puede enviarse con el nombre original  -> "emp.var.rate": -1.8
  * o con el nombre "pythonizado"                  -> "emp_var_rate": -1.8

Ambos funcionan gracias a `populate_by_name=True`.
"""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

EJEMPLO_CLIENTE = {
    "age": 35,
    "job": "technician",
    "marital": "single",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.313,
    "nr.employed": 5099.1,
}


class ClienteInput(BaseModel):
    """Las 19 features que necesita el modelo. `duration` NO se pide a propósito."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",  # si mandan `duration` u otra columna, se acepta y se avisa
        json_schema_extra={"example": EJEMPLO_CLIENTE},
    )

    # --- datos del cliente ---
    age: int = Field(..., ge=17, le=120, description="Edad del cliente")
    job: str = Field(..., description="Tipo de trabajo")
    marital: str = Field(..., description="Estado civil")
    education: str = Field(..., description="Nivel educativo")
    default: str = Field(..., description="¿Tiene crédito en mora? (yes/no/unknown)")
    housing: str = Field(..., description="¿Tiene préstamo hipotecario?")
    loan: str = Field(..., description="¿Tiene préstamo personal?")

    # --- datos del contacto actual ---
    contact: str = Field(..., description="Medio de contacto (cellular/telephone)")
    month: str = Field(..., description="Mes del último contacto (mar..dec, en inglés)")
    day_of_week: str = Field(..., description="Día de la semana (mon..fri)")
    campaign: int = Field(..., ge=1, le=100, description="Contactos en esta campaña")

    # --- historial de campañas anteriores ---
    pdays: int = Field(..., ge=0, le=999, description="Días desde el último contacto (999 = nunca)")
    previous: int = Field(..., ge=0, le=50, description="Contactos en campañas previas")
    poutcome: str = Field(..., description="Resultado de la campaña previa")

    # --- contexto macroeconómico ---
    emp_var_rate: float = Field(..., alias="emp.var.rate", description="Tasa de variación del empleo")
    cons_price_idx: float = Field(..., alias="cons.price.idx", description="Índice de precios al consumidor")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", description="Índice de confianza del consumidor")
    euribor3m: float = Field(..., description="Euríbor a 3 meses")
    nr_employed: float = Field(..., alias="nr.employed", description="Número de empleados (miles)")

    def to_model_dict(self) -> dict:
        """Devuelve el diccionario con los nombres originales del dataset."""
        return self.model_dump(by_alias=True)


class BatchInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"clientes": [EJEMPLO_CLIENTE]}})
    clientes: List[ClienteInput] = Field(..., min_length=1, max_length=500)


class PredictionOutput(BaseModel):
    prediction: Literal["yes", "no"]
    prediction_label: str
    probabilities: Dict[str, float]
    confidence: float
    confidence_level: Literal["alta", "media", "baja"]
    threshold: float
    model_version: str
    warnings: List[str] = []


class BatchOutput(BaseModel):
    n: int
    resultados: List[PredictionOutput]


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

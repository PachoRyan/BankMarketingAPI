"""
app/main.py
===========
PARTE 4 de la actividad — API REST de despliegue (FastAPI).

Endpoints
---------
GET  /                -> información del servicio
GET  /health          -> healthcheck (lo usa Cloud Run y el script cliente)
GET  /model-info      -> metadatos y métricas del modelo en producción
GET  /schema          -> valores admitidos por cada campo (útil para probar a mano)
GET  /ejemplo         -> un payload de ejemplo listo para copiar/pegar
POST /predict         -> OBLIGATORIO: predicción de un cliente
POST /predict/batch   -> predicción de varios clientes en una sola llamada

La documentación interactiva (Swagger UI) queda en /docs: desde ahí el docente
puede hacer pruebas manuales sin instalar nada.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.features import CATEGORY_VALUES, NUMERIC_RANGES, RAW_FEATURES
from app.predict import InputValidationError, load_model, model_info, predict, predict_batch
from app.schemas import (
    BatchInput,
    BatchOutput,
    ClienteInput,
    EJEMPLO_CLIENTE,
    HealthOutput,
    PredictionOutput,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar (evita pagar el coste en el primer request)."""
    load_model()
    print("[startup] Modelo cargado correctamente.")
    yield
    print("[shutdown] Cerrando servicio.")


app = FastAPI(
    title="Bank Marketing API - Predicción de depósito a plazo - Grupo Mitsuo :D",
    description=(
        "API de predicción entrenada con el Bank Marketing Dataset (UCI).\n\n"
        "**La variable `duration` no se usa**: solo se conoce después de la llamada "
        "y utilizarla sería data leakage. Si se envía en el JSON, se ignora."

        "> Oscar Ryan Chu Lao Orrego"
        "> Mitsuo Murakami"
        "> Christian Frisancho"
        "> Gonzalo Rodriguez"
        "> Luis Huaregui"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS abierto: la API es pública y de uso académico.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Manejo de errores de validación propios
# ---------------------------------------------------------------------------

@app.exception_handler(InputValidationError)
async def input_validation_handler(request: Request, exc: InputValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Entrada inválida", "errors": exc.errors},
    )


# ---------------------------------------------------------------------------
# Endpoints informativos
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
def root():
    return {
        "servicio": "Bank Marketing API",
        "estado": "ok",
        "docs": "/docs",
        "endpoint_principal": "POST /predict",
        "nota": "La variable `duration` está excluida del modelo (data leakage).",
    }


@app.get("/health", response_model=HealthOutput, tags=["info"])
def health():
    try:
        info = model_info()
        return HealthOutput(
            status="ok", model_loaded=True, model_version=info.get("model_version", "n/a")
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {exc}")


@app.get("/model-info", tags=["info"])
def info_modelo():
    """Metadatos y métricas del modelo desplegado."""
    from app.predict import load_model as _lm

    artefacto = _lm()
    return {"metadata": artefacto["metadata"], "metrics": artefacto.get("metrics", {})}


@app.get("/schema", tags=["info"])
def esquema():
    """Campos requeridos, valores categóricos admitidos y rangos numéricos."""
    return {
        "campos_requeridos": RAW_FEATURES,
        "valores_categoricos": CATEGORY_VALUES,
        "rangos_numericos": {k: {"min": v[0], "max": v[1]} for k, v in NUMERIC_RANGES.items()},
        "excluidas": ["duration"],
    }


@app.get("/ejemplo", tags=["info"])
def ejemplo():
    """Payload de ejemplo listo para pegar en POST /predict."""
    return EJEMPLO_CLIENTE


# ---------------------------------------------------------------------------
# Endpoints de predicción
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionOutput, tags=["predicción"])
def predecir(cliente: ClienteInput):
    """Predice si un cliente aceptará el depósito a plazo.

    Devuelve la clase predicha, las probabilidades de ambas clases y un nivel de
    confianza. La decisión usa el umbral calibrado durante el entrenamiento
    (no el 0.5 por defecto).
    """
    return predict(cliente.to_model_dict())


@app.post("/predict/batch", response_model=BatchOutput, tags=["predicción"])
def predecir_lote(payload: BatchInput):
    """Predicción para una lista de clientes (máx. 500 por llamada)."""
    resultados = predict_batch([c.to_model_dict() for c in payload.clientes])
    return {"n": len(resultados), "resultados": resultados}


if __name__ == "__main__":
    # Ejecución local: python -m app.main  (Cloud Run inyecta la variable PORT)
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True)

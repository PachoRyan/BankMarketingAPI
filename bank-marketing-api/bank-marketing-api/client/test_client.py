"""
client/test_client.py
=====================
PARTE 5 de la actividad — Cliente de prueba automático.

Envía requests reales a la API y muestra los resultados por consola:

  1. GET  /health          -> el servicio responde y el modelo está cargado
  2. GET  /model-info      -> qué modelo hay desplegado
  3. POST /predict         -> 6 perfiles de cliente MUY distintos entre sí
  4. POST /predict         -> 2 casos inválidos (debe responder 422)
  5. POST /predict         -> un caso que incluye `duration` (debe ignorarla)
  6. POST /predict/batch   -> los 6 perfiles en una sola llamada

Uso
---
    # contra la API local
    python client/test_client.py

    # contra la API desplegada en Cloud Run
    python client/test_client.py --url https://bank-marketing-api-XXXX.run.app

Devuelve código de salida 0 si todas las pruebas pasan, 1 si alguna falla.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Tuple

import requests

# ---------------------------------------------------------------------------
# 6 perfiles de prueba, elegidos para cubrir situaciones distintas
# ---------------------------------------------------------------------------

EJEMPLOS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "1) Estudiante joven, ya contactado con ÉXITO en una campaña previa, "
        "contexto de tasas bajas -> perfil de ALTA probabilidad",
        {
            "age": 24, "job": "student", "marital": "single", "education": "high.school",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "mar", "day_of_week": "tue", "campaign": 1, "pdays": 6, "previous": 2,
            "poutcome": "success", "emp.var.rate": -1.8, "cons.price.idx": 92.843,
            "cons.conf.idx": -50.0, "euribor3m": 1.266, "nr.employed": 5099.1,
        },
    ),
    (
        "2) Jubilado, contactado por celular en octubre, campaña previa exitosa",
        {
            "age": 68, "job": "retired", "marital": "married", "education": "professional.course",
            "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
            "month": "oct", "day_of_week": "thu", "campaign": 1, "pdays": 3, "previous": 1,
            "poutcome": "success", "emp.var.rate": -3.4, "cons.price.idx": 92.431,
            "cons.conf.idx": -26.9, "euribor3m": 0.742, "nr.employed": 5017.5,
        },
    ),
    (
        "3) Obrero contactado por teléfono fijo en mayo, sin historial "
        "-> perfil de BAJA probabilidad (caso más común del dataset)",
        {
            "age": 45, "job": "blue-collar", "marital": "married", "education": "basic.9y",
            "default": "unknown", "housing": "yes", "loan": "no", "contact": "telephone",
            "month": "may", "day_of_week": "mon", "campaign": 3, "pdays": 999, "previous": 0,
            "poutcome": "nonexistent", "emp.var.rate": 1.1, "cons.price.idx": 93.994,
            "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0,
        },
    ),
    (
        "4) Administrativa con estudios universitarios, primera campaña, "
        "contexto económico intermedio",
        {
            "age": 33, "job": "admin.", "marital": "single", "education": "university.degree",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "aug", "day_of_week": "wed", "campaign": 2, "pdays": 999, "previous": 0,
            "poutcome": "nonexistent", "emp.var.rate": 1.4, "cons.price.idx": 93.444,
            "cons.conf.idx": -36.1, "euribor3m": 4.964, "nr.employed": 5228.1,
        },
    ),
    (
        "5) Cliente 'quemado': 17 llamadas en la misma campaña y un fracaso previo",
        {
            "age": 52, "job": "technician", "marital": "divorced", "education": "basic.4y",
            "default": "no", "housing": "no", "loan": "yes", "contact": "telephone",
            "month": "jul", "day_of_week": "fri", "campaign": 17, "pdays": 999, "previous": 1,
            "poutcome": "failure", "emp.var.rate": 1.4, "cons.price.idx": 93.918,
            "cons.conf.idx": -42.7, "euribor3m": 4.963, "nr.employed": 5228.1,
        },
    ),
    (
        "6) Valores 'unknown' en varias variables (caso realista de datos sucios)",
        {
            "age": 41, "job": "unknown", "marital": "unknown", "education": "unknown",
            "default": "unknown", "housing": "unknown", "loan": "unknown", "contact": "cellular",
            "month": "nov", "day_of_week": "tue", "campaign": 1, "pdays": 999, "previous": 0,
            "poutcome": "nonexistent", "emp.var.rate": -0.1, "cons.price.idx": 93.2,
            "cons.conf.idx": -42.0, "euribor3m": 4.076, "nr.employed": 5195.8,
        },
    ),
]

# Casos que DEBEN ser rechazados por la validación (esperamos HTTP 422)
EJEMPLOS_INVALIDOS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "A) Categoría inexistente en `job`",
        {**EJEMPLOS[3][1], "job": "astronauta"},
    ),
    (
        "B) Falta el campo obligatorio `age`",
        {k: v for k, v in EJEMPLOS[3][1].items() if k != "age"},
    ),
]


# ---------------------------------------------------------------------------
# Helpers de impresión
# ---------------------------------------------------------------------------

VERDE, ROJO, AMARILLO, GRIS, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"


def titulo(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def ok(msg: str) -> None:
    print(f"{VERDE}[OK]{RESET} {msg}")


def fallo(msg: str) -> None:
    print(f"{ROJO}[FALLO]{RESET} {msg}")


def mostrar_prediccion(r: Dict[str, Any]) -> None:
    color = VERDE if r["prediction"] == "yes" else AMARILLO
    print(
        f"     -> {color}{r['prediction'].upper()}{RESET} "
        f"(P(yes)={r['probabilities']['yes']:.3f} | P(no)={r['probabilities']['no']:.3f}) "
        f"| confianza: {r['confidence_level']} ({r['confidence']:.3f}) "
        f"| umbral: {r['threshold']}"
    )
    for w in r.get("warnings", []):
        print(f"     {GRIS}aviso: {w}{RESET}")


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Cliente de prueba de la Bank Marketing API")
    parser.add_argument("--url", default="http://localhost:8080", help="URL base de la API")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    fallos = 0

    print(f"Probando API en: {base}")

    # --- 1. Health -----------------------------------------------------------
    titulo("1. HEALTHCHECK  (GET /health)")
    try:
        t = time.time()
        r = requests.get(f"{base}/health", timeout=args.timeout)
        r.raise_for_status()
        ok(f"{r.json()}  [{(time.time() - t) * 1000:.0f} ms]")
    except Exception as exc:
        fallo(f"No se pudo contactar la API: {exc}")
        print("Sugerencia: ¿está levantada? -> uvicorn app.main:app --port 8080")
        return 1

    # --- 2. Info del modelo --------------------------------------------------
    titulo("2. MODELO DESPLEGADO  (GET /model-info)")
    try:
        meta = requests.get(f"{base}/model-info", timeout=args.timeout).json()["metadata"]
        ok(
            f"{meta['model_name']} v{meta['model_version']} | umbral={meta['decision_threshold']} "
            f"| entrenado: {meta['trained_at']}"
        )
        print(f"     Variables excluidas: {meta['excluded_features']}")
    except Exception as exc:
        fallo(f"/model-info falló: {exc}")
        fallos += 1

    # --- 3. Predicciones individuales ---------------------------------------
    titulo("3. PREDICCIONES INDIVIDUALES  (POST /predict)")
    for desc, payload in EJEMPLOS:
        print(f"\n{desc}")
        try:
            t = time.time()
            r = requests.post(f"{base}/predict", json=payload, timeout=args.timeout)
            ms = (time.time() - t) * 1000
            if r.status_code != 200:
                fallo(f"HTTP {r.status_code}: {r.text[:200]}")
                fallos += 1
                continue
            mostrar_prediccion(r.json())
            print(f"     {GRIS}latencia: {ms:.0f} ms{RESET}")
        except Exception as exc:
            fallo(str(exc))
            fallos += 1

    # --- 4. Validación de entradas inválidas ---------------------------------
    titulo("4. VALIDACIÓN DE ENTRADAS INVÁLIDAS  (se espera HTTP 422)")
    for desc, payload in EJEMPLOS_INVALIDOS:
        print(f"\n{desc}")
        r = requests.post(f"{base}/predict", json=payload, timeout=args.timeout)
        if r.status_code == 422:
            detalle = json.dumps(r.json(), ensure_ascii=False)
            ok(f"rechazado correctamente -> {detalle[:160]}...")
        else:
            fallo(f"se esperaba 422 y se recibió {r.status_code}")
            fallos += 1

    # --- 5. `duration` debe ignorarse ----------------------------------------
    titulo("5. LA API IGNORA `duration`  (restricción crítica)")
    payload = {**EJEMPLOS[2][1], "duration": 600}
    r = requests.post(f"{base}/predict", json=payload, timeout=args.timeout)
    if r.status_code == 200 and any("duration" in w for w in r.json().get("warnings", [])):
        ok("la API acepta el request, ignora `duration` y lo reporta como aviso")
        mostrar_prediccion(r.json())
    else:
        fallo(f"comportamiento inesperado: HTTP {r.status_code} -> {r.text[:200]}")
        fallos += 1

    # --- 6. Batch ------------------------------------------------------------
    titulo("6. PREDICCIÓN POR LOTES  (POST /predict/batch)")
    try:
        t = time.time()
        r = requests.post(
            f"{base}/predict/batch",
            json={"clientes": [p for _, p in EJEMPLOS]},
            timeout=args.timeout,
        )
        r.raise_for_status()
        data = r.json()
        resumen = {}
        for res in data["resultados"]:
            resumen[res["prediction"]] = resumen.get(res["prediction"], 0) + 1
        ok(
            f"{data['n']} clientes procesados en una sola llamada "
            f"[{(time.time() - t) * 1000:.0f} ms] -> {resumen}"
        )
    except Exception as exc:
        fallo(str(exc))
        fallos += 1

    # --- Resumen -------------------------------------------------------------
    titulo("RESUMEN")
    if fallos == 0:
        print(f"{VERDE}Todas las pruebas pasaron correctamente.{RESET}")
        return 0
    print(f"{ROJO}{fallos} prueba(s) fallaron.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

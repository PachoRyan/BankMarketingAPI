# Hackathon 6 — Bank Marketing API

Sistema de Machine Learning end-to-end sobre el **Bank Marketing Dataset (UCI)**:
entrena un modelo, lo empaqueta en una API REST, lo containeriza y lo despliega
en **Google Cloud Run**.

> **Restricción crítica cumplida:** la variable `duration` **no** se usa en el
> modelo final. Solo se conoce después de la llamada, así que usarla sería data
> leakage. Si un request la incluye, la API la ignora y lo reporta en `warnings`.

**URL pública del servicio:** `https://...run.app` ← *(pegar aquí tras el despliegue)*
**Documentación interactiva:** `https://...run.app/docs`

---

## 1. Qué hay en este repositorio

```
.
├── app/                        # Código que va al contenedor
│   ├── features.py             # Contrato de datos: features, categorías, feature engineering
│   ├── predict.py              # PARTE 3: predict(input_dict) + validación
│   ├── schemas.py              # Contratos Pydantic de la API
│   └── main.py                 # PARTE 4: API FastAPI
├── src/
│   ├── download_data.py        # Descarga el dataset de UCI
│   └── train.py                # PARTES 1 y 2: análisis, comparación y modelo final
├── notebooks/
│   └── 01_analisis_y_modelado.ipynb   # PARTE 1 documentada, con gráficos
├── client/
│   ├── test_client.py          # PARTE 5: pruebas automáticas
│   ├── PRUEBAS_MANUALES.md     # PARTE 5: pruebas manuales (curl / Swagger)
│   └── ejemplos.json           # Payloads listos para copiar
├── docs/
│   ├── parte6_nivel_sistema.md # PARTE 6: nivel del sistema y riesgos
│   └── despliegue_gcp.md       # Guía completa de despliegue en GCP
├── models/
│   ├── model.joblib            # Modelo serializado (pipeline + umbral + metadatos)
│   ├── metadata.json
│   └── metrics.json
├── scripts/
│   ├── check_local.sh          # Verificación local antes de desplegar
│   └── deploy.sh               # Despliegue completo en Cloud Run (un solo comando)
├── Dockerfile
├── cloudbuild.yaml             # CI/CD: GitHub → Cloud Build → Artifact Registry → Cloud Run
└── requirements.txt
```

---

## 2. Resultados del modelo

Comparación en validación cruzada estratificada (3 folds), métrica de selección
**PR-AUC** (independiente del umbral y adecuada para clases desbalanceadas):

| Modelo | PR-AUC (CV) | Mejores hiperparámetros |
|---|---|---|
| **Random Forest** ← seleccionado | **0.4645** | `max_depth=10`, `min_samples_leaf=5` |
| HistGradientBoosting | 0.4598 | `learning_rate=0.05`, `max_leaf_nodes=31` |
| Regresión logística | 0.4439 | `C=1.0` |

Umbral de decisión calibrado con predicciones *out-of-fold*: **0.63**
(el 0.5 por defecto daba F1 = 0.480; el calibrado da 0.501).

Métricas en el conjunto de test (8.236 filas nunca vistas):

| Métrica | Valor |
|---|---|
| F1 (clase `yes`) | **0.524** |
| Balanced accuracy | **0.752** |
| Precision | 0.472 |
| Recall | 0.588 |
| ROC-AUC | 0.813 |
| PR-AUC | 0.483 |

Matriz de confusión: TN 6.698 · FP 610 · FN 382 · TP 546.

**Lectura de negocio:** la tasa base de aceptación es 11.3%. De cada 100 clientes
que el modelo recomienda llamar, ~47 contratan: **multiplica por ~4 la eficiencia
del contacto**. Sin `duration` no se puede aspirar a los AUC > 0.93 que se ven en
internet — esos modelos no son utilizables en producción.

---

## 3. Cómo ejecutarlo en local

### Opción A — Entorno virtual

```bash
# 1. Entorno con Python 3.11
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 2. Dependencias
pip install -r requirements.txt

# 3. Levantar la API (el modelo ya está en models/model.joblib)
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Abre <http://localhost:8080/docs>.

### Opción B — Docker

```bash
docker build -t bank-marketing-api .
docker run -p 8080:8080 bank-marketing-api
```

### Reentrenar el modelo desde cero

```bash
pip install -r requirements-train.txt
python src/download_data.py     # descarga data/bank-additional-full.csv
python src/train.py             # ~2 min; regenera models/model.joblib
```

El script imprime la exploración, el análisis de leakage, la comparación de
modelos, la calibración del umbral y la evaluación final.

---

## 4. Cómo probarlo

```bash
# Pruebas automáticas (6 perfiles + validaciones + batch)
python client/test_client.py --url http://localhost:8080

# Contra el servicio desplegado
python client/test_client.py --url https://TU-SERVICIO.run.app
```

Pruebas manuales (curl y Swagger): ver [`client/PRUEBAS_MANUALES.md`](client/PRUEBAS_MANUALES.md).

### Ejemplo de request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 24, "job": "student", "marital": "single", "education": "high.school",
    "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
    "month": "mar", "day_of_week": "tue", "campaign": 1, "pdays": 6, "previous": 2,
    "poutcome": "success", "emp.var.rate": -1.8, "cons.price.idx": 92.843,
    "cons.conf.idx": -50.0, "euribor3m": 1.266, "nr.employed": 5099.1
  }'
```

Respuesta:

```json
{
  "prediction": "yes",
  "prediction_label": "Contratará el depósito a plazo",
  "probabilities": {"no": 0.2088, "yes": 0.7912},
  "confidence": 0.7912,
  "confidence_level": "media",
  "threshold": 0.63,
  "model_version": "1.0.0",
  "warnings": []
}
```

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Healthcheck (servicio + modelo cargado) |
| GET | `/model-info` | Modelo, hiperparámetros y métricas desplegadas |
| GET | `/schema` | Campos obligatorios, categorías y rangos admitidos |
| GET | `/ejemplo` | Payload de ejemplo |
| **POST** | **`/predict`** | **Predicción individual (obligatorio)** |
| POST | `/predict/batch` | Predicción por lotes (máx. 500) |
| GET | `/docs` | Swagger UI |

---

## 5. Despliegue en GCP

Guía completa paso a paso: [`docs/despliegue_gcp.md`](docs/despliegue_gcp.md).

**Atajo — dos comandos:**

```bash
chmod +x scripts/*.sh
./scripts/check_local.sh                    # verifica que todo funciona en local
./scripts/deploy.sh tu-proyecto-gcp         # habilita APIs, crea el repo, construye y despliega
```

`deploy.sh` es idempotente (se puede repetir) y al terminar imprime la URL pública.

Resumen de lo que hace, comando a comando:

```bash
export PROJECT_ID="tu-proyecto-gcp"; export REGION="us-central1"
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create bank-repo --repository-format=docker --location=$REGION

export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/bank-repo/bank-marketing-api:v1"
gcloud builds submit --tag $IMAGE
gcloud run deploy bank-marketing-api --image $IMAGE --region $REGION \
  --platform managed --allow-unauthenticated --port 8080 --memory 1Gi --cpu 1
```

Despliegue continuo desde GitHub: Cloud Run → *Configurar despliegue continuo* →
GitHub → rama `main` → Dockerfile. A partir de ahí, cada push redespliega
(ver sección 6 de la guía; el repo ya incluye `cloudbuild.yaml`).

---

## 6. Decisiones de diseño

- **Un único contrato de features** (`app/features.py`) usado por el
  entrenamiento y por la API: elimina el riesgo de *training/serving skew*.
- **Todo el preprocesamiento dentro del `Pipeline`** de scikit-learn: se ajusta
  solo con los datos de entrenamiento de cada fold, evitando leakage silencioso.
- **`unknown` se trata como categoría**, no se imputa: en campañas telefónicas
  la ausencia de dato es informativa.
- **Umbral calibrado, no 0.5**: con 11% de positivos, el umbral por defecto
  sesga el modelo hacia la clase mayoritaria. Se elige con predicciones
  out-of-fold, nunca con el test.
- **`OneHotEncoder(handle_unknown="ignore")`**: una categoría nueva en
  producción no tumba la API.
- **El modelo va dentro de la imagen Docker** (6 MB): sin dependencias externas
  en tiempo de arranque, sin latencia de descarga, despliegue inmutable.
- **Carga del modelo en el `lifespan` de FastAPI**: el *cold start* se paga al
  arrancar el contenedor, no en el primer request del usuario.

---

## 7. Nivel del sistema (Parte 6)

**Nivel 2 — Recomendación.** La API devuelve una predicción con probabilidad y
nivel de confianza; no ejecuta ninguna acción comercial. El destinatario es una
persona que prioriza su lista de llamadas.

Justificación completa y análisis de riesgos (rendimiento, sesgo, drift,
seguridad y qué haría falta para subir de nivel):
[`docs/parte6_nivel_sistema.md`](docs/parte6_nivel_sistema.md).

---

## 8. Correspondencia con la rúbrica

| Parte | Dónde está |
|---|---|
| 1 — Análisis y modelado | `notebooks/01_analisis_y_modelado.ipynb`, `src/train.py` |
| 2 — Entrenamiento final | `src/train.py` (sección 7), `models/model.joblib`, `models/metrics.json` |
| 3 — Pipeline reproducible | `app/predict.py` → `predict(input_dict)` |
| 4 — API de despliegue | `app/main.py` → `POST /predict` |
| 5 — Cliente de prueba | `client/test_client.py` (automático) + `client/PRUEBAS_MANUALES.md` (manual) |
| 6 — Nivel del sistema | `docs/parte6_nivel_sistema.md` |
| Despliegue en GCP | `Dockerfile`, `cloudbuild.yaml`, `docs/despliegue_gcp.md` |

---

## Dataset

Moro, S., Rita, P. & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning
Repository. <https://archive.ics.uci.edu/ml/datasets/bank+marketing>

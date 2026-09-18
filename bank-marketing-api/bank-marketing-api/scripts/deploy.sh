#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/deploy.sh
# Despliegue completo en Google Cloud Run: habilita APIs, crea el repositorio
# en Artifact Registry, construye la imagen con Cloud Build y despliega.
#
# Uso:
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh mi-proyecto-gcp            # región por defecto: us-central1
#   ./scripts/deploy.sh mi-proyecto-gcp us-east1
#
# Es idempotente: se puede volver a ejecutar las veces que haga falta.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
REPO="bank-repo"
SERVICE="bank-marketing-api"
TAG="$(date +%Y%m%d-%H%M%S)"

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "ERROR: indica el PROJECT_ID -> ./scripts/deploy.sh tu-proyecto-gcp"
  exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

echo "=============================================="
echo " PROJECT_ID : ${PROJECT_ID}"
echo " REGION     : ${REGION}"
echo " IMAGEN     : ${IMAGE}:${TAG}"
echo "=============================================="

# --- 0. Comprobaciones previas ---------------------------------------------
command -v gcloud >/dev/null || { echo "ERROR: gcloud no está instalado."; exit 1; }
[[ -f "Dockerfile" ]] || { echo "ERROR: ejecuta el script desde la raíz del repo."; exit 1; }
[[ -f "models/model.joblib" ]] || { echo "ERROR: falta models/model.joblib (corre python src/train.py)."; exit 1; }

gcloud config set project "${PROJECT_ID}" >/dev/null

# --- 1. Habilitar APIs ------------------------------------------------------
echo ">> [1/4] Habilitando APIs (puede tardar ~1 min la primera vez)..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# --- 2. Repositorio en Artifact Registry ------------------------------------
echo ">> [2/4] Repositorio en Artifact Registry..."
if gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  echo "   ya existe: ${REPO}"
else
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Imagenes de la API de Bank Marketing"
fi

# --- 3. Build en la nube ----------------------------------------------------
echo ">> [3/4] Construyendo la imagen con Cloud Build..."
gcloud builds submit --tag "${IMAGE}:${TAG}" .

# --- 4. Deploy en Cloud Run -------------------------------------------------
echo ">> [4/4] Desplegando en Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}:${TAG}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 5

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"

echo ""
echo "=============================================="
echo " DESPLIEGUE COMPLETADO"
echo " URL pública : ${URL}"
echo " Swagger UI  : ${URL}/docs"
echo "=============================================="
echo ""
echo "Verificando /health ..."
curl -s "${URL}/health" || true
echo ""
echo ""
echo "Prueba completa:"
echo "  python client/test_client.py --url ${URL}"

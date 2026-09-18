# ---------------------------------------------------------------------------
# Imagen de la API de predicción (Bank Marketing) para Cloud Run.
#
# Se parte de python:3.11-slim, igual que en el manual del taller: dentro del
# contenedor todo corre con Python 3.11, así que la versión que tengas instalada
# en tu máquina deja de ser relevante.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# - PYTHONDONTWRITEBYTECODE: no genera .pyc dentro del contenedor
# - PYTHONUNBUFFERED: los logs salen inmediatamente (se ven en Cloud Logging)
# - PORT: Cloud Run inyecta esta variable; 8080 es el valor por defecto
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# 1) Primero solo las dependencias: si el código cambia pero requirements.txt no,
#    Docker reutiliza esta capa cacheada y el build es mucho más rápido.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2) Después el código de la API y el modelo serializado.
COPY app/ ./app/
COPY models/model.joblib ./models/model.joblib

# Usuario sin privilegios (buena práctica de seguridad en contenedores)
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run escucha en $PORT. Un solo worker: el contenedor escala
# horizontalmente (más instancias), no verticalmente.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1

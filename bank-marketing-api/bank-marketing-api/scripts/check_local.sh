#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/check_local.sh
# Verificación previa al despliegue: levanta la API en local, corre el cliente
# de prueba y la apaga. Si esto pasa, el despliegue en Cloud Run también debería.
#
# Uso:
#   chmod +x scripts/check_local.sh
#   ./scripts/check_local.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PORT="${1:-8080}"

[[ -f "Dockerfile" ]] || { echo "ERROR: ejecuta el script desde la raíz del repo."; exit 1; }
[[ -f "models/model.joblib" ]] || { echo "ERROR: falta models/model.joblib."; exit 1; }

echo ">> Comprobando que el modelo carga y que predict() funciona..."
python -c "
from app.predict import predict, model_info
info = model_info()
print('   modelo:', info['model_name'], 'v' + info['model_version'], '| umbral:', info['threshold'])
assert 'duration' not in info['model_features'], 'duration sigue en el modelo!'
print('   OK: duration NO está entre las features del modelo')
"

echo ">> Levantando la API en el puerto ${PORT}..."
python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}" --log-level warning &
API_PID=$!
trap 'kill ${API_PID} 2>/dev/null || true' EXIT

# Espera a que el servicio responda (máx. 30 s)
for _ in $(seq 1 30); do
  if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 1
done

echo ">> Corriendo el cliente de prueba..."
python client/test_client.py --url "http://127.0.0.1:${PORT}"
RESULT=$?

echo ""
if [[ ${RESULT} -eq 0 ]]; then
  echo "TODO OK. Ya puedes desplegar: ./scripts/deploy.sh tu-proyecto-gcp"
else
  echo "Hubo fallos. Revísalos antes de desplegar."
fi
exit ${RESULT}

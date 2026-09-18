# Pruebas manuales de la API

> Reemplaza `$URL` por la URL pública del servicio en Cloud Run
> (o por `http://localhost:8080` si la estás probando en local).
>
> ```bash
> export URL="https://bank-marketing-api-XXXXXXXX-uc.a.run.app"
> ```

## Opción A — Sin instalar nada: Swagger UI

Abre en el navegador:

```
$URL/docs
```

1. Despliega `POST /predict` → **Try it out**.
2. El cuerpo ya viene precargado con un ejemplo válido; edítalo si quieres.
3. **Execute** → abajo aparece la respuesta con la predicción y las probabilidades.

Otros endpoints útiles desde la misma pantalla:

| Endpoint | Para qué sirve |
|---|---|
| `GET /health` | Verificar que el servicio y el modelo están arriba |
| `GET /model-info` | Ver qué modelo, qué hiperparámetros y qué métricas están desplegados |
| `GET /schema` | Lista de campos obligatorios y valores admitidos por cada uno |
| `GET /ejemplo` | Un payload de ejemplo listo para copiar |
| `POST /predict/batch` | Varios clientes en una sola llamada |

Más ejemplos listos para copiar/pegar: [`client/ejemplos.json`](ejemplos.json).

## Opción B — Con `curl`

### 1. Healthcheck

```bash
curl -s $URL/health
```

### 2. Cliente con ALTA probabilidad (estudiante joven, campaña previa exitosa)

```bash
curl -s -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 24, "job": "student", "marital": "single", "education": "high.school",
    "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
    "month": "mar", "day_of_week": "tue", "campaign": 1, "pdays": 6, "previous": 2,
    "poutcome": "success", "emp.var.rate": -1.8, "cons.price.idx": 92.843,
    "cons.conf.idx": -50.0, "euribor3m": 1.266, "nr.employed": 5099.1
  }'
```

### 3. Cliente con BAJA probabilidad (obrero, teléfono fijo, sin historial)

```bash
curl -s -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "job": "blue-collar", "marital": "married", "education": "basic.9y",
    "default": "unknown", "housing": "yes", "loan": "no", "contact": "telephone",
    "month": "may", "day_of_week": "mon", "campaign": 3, "pdays": 999, "previous": 0,
    "poutcome": "nonexistent", "emp.var.rate": 1.1, "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0
  }'
```

### 4. Validación: categoría inexistente → debe responder **422**

```bash
curl -s -i -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 33, "job": "astronauta", "marital": "single", "education": "university.degree",
    "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
    "month": "aug", "day_of_week": "wed", "campaign": 2, "pdays": 999, "previous": 0,
    "poutcome": "nonexistent", "emp.var.rate": 1.4, "cons.price.idx": 93.444,
    "cons.conf.idx": -36.1, "euribor3m": 4.964, "nr.employed": 5228.1
  }'
```

### 5. Validación: falta un campo obligatorio → **422**

```bash
curl -s -X POST $URL/predict -H "Content-Type: application/json" -d '{"age": 30}'
```

### 6. La API ignora `duration` (restricción crítica)

El mismo payload del punto 3, pero añadiendo `"duration": 600`. La respuesta es
idéntica y aparece un aviso en el campo `warnings`:

```bash
curl -s -X POST $URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "job": "blue-collar", "marital": "married", "education": "basic.9y",
    "default": "unknown", "housing": "yes", "loan": "no", "contact": "telephone",
    "month": "may", "day_of_week": "mon", "campaign": 3, "pdays": 999, "previous": 0,
    "poutcome": "nonexistent", "emp.var.rate": 1.1, "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0,
    "duration": 600
  }'
```

### 7. Lote de clientes

```bash
curl -s -X POST $URL/predict/batch \
  -H "Content-Type: application/json" \
  -d "{\"clientes\": [$(curl -s $URL/ejemplo), $(curl -s $URL/ejemplo)]}"
```

## Opción C — Script automático

```bash
python client/test_client.py --url $URL
```

Ejecuta las 6 baterías de pruebas anteriores y termina con código de salida 0
si todo pasó.

# Despliegue en Google Cloud Run (con Artifact Registry y GitHub)

Guía completa, de cero a URL pública. Sigue el mismo flujo del manual del taller
(Fase 1: Cloud Run + Artifact Registry + Cloud Build). **No se usa Vertex AI.**

```
GitHub (push a main)
      │
      ▼
Cloud Build  ──build──►  Artifact Registry  ──deploy──►  Cloud Run  ──►  URL pública
```

---

## 0. Prerrequisitos

- Cuenta de Google Cloud con un **proyecto creado y facturación habilitada**.
- Google Cloud CLI instalado:
  - macOS: `brew install --cask google-cloud-sdk`
  - Windows: instalador oficial de Google Cloud CLI (marcar "add gcloud to PATH")
  - Verificar: `gcloud --version`
- Git y una cuenta de GitHub.
- Docker **no es obligatorio**: la imagen se construye en la nube con Cloud Build.

> Si prefieres no instalar nada, puedes hacer todo desde **Cloud Shell**
> (icono `>_` en la consola de GCP): ya viene con gcloud, Docker y Git, y la
> sesión está autenticada.

---

## 1. Subir el proyecto a GitHub

Desde la carpeta del proyecto:

```bash
git init
git add .
git commit -m "Hackathon 6: API de predicción Bank Marketing"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/bank-marketing-api.git
git push -u origin main
```

Comprueba antes de hacer push que `models/model.joblib` **sí** está incluido
(pesa ~6 MB, muy por debajo del límite de 100 MB de GitHub):

```bash
git ls-files models/
# debe mostrar: models/metadata.json  models/metrics.json  models/model.joblib
```

El CSV de datos **no** se sube (está en `.gitignore`); cualquiera puede
regenerarlo con `python src/download_data.py`.

---

## 2. Autenticarse y elegir el proyecto

```bash
gcloud auth login

# Reemplaza por el ID real de tu proyecto (no el nombre)
export PROJECT_ID="tu-proyecto-gcp"
gcloud config set project $PROJECT_ID

export REGION="us-central1"
```

> En Windows/PowerShell, `export X="y"` se escribe `$env:X="y"`.

Verifica:

```bash
gcloud config list
```

---

## 3. Habilitar las APIs necesarias

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

(Tarda ~1 minuto la primera vez.)

---

## 4. Crear el repositorio en Artifact Registry

Solo se hace **una vez por proyecto**. Aquí es donde vivirán las imágenes Docker.

```bash
gcloud artifacts repositories create bank-repo \
  --repository-format=docker \
  --location=$REGION \
  --description="Imagenes de la API de Bank Marketing"
```

Comprobar:

```bash
gcloud artifacts repositories list --location=$REGION
```

---

## 5. Primer despliegue manual (para tener la URL cuanto antes)

Esto construye la imagen en la nube y la despliega. Sirve como verificación de
que el `Dockerfile` está bien antes de automatizar nada.

```bash
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/bank-repo/bank-marketing-api:v1"
echo $IMAGE

# 5.1 Construir y subir la imagen a Artifact Registry
gcloud builds submit --tag $IMAGE

# 5.2 Desplegar en Cloud Run
gcloud run deploy bank-marketing-api \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 5
```

Al terminar, gcloud imprime la **URL pública** del servicio:

```
Service URL: https://bank-marketing-api-XXXXXXXXX-uc.a.run.app
```

Pruébala de inmediato:

```bash
export URL="https://bank-marketing-api-XXXXXXXXX-uc.a.run.app"
curl -s $URL/health
python client/test_client.py --url $URL
```

Y abre `$URL/docs` en el navegador para la documentación interactiva.

---

## 6. Vincular el despliegue con GitHub (despliegue continuo)

A partir de aquí, cada `git push` a `main` reconstruye y redespliega solo.
Hay dos formas; la **A** es la más rápida y es la que muestra el taller.

### Opción A — Desde la consola de Cloud Run (recomendada)

1. Consola de GCP → **Cloud Run** → abre el servicio `bank-marketing-api`.
2. Pestaña **Revisiones** → botón **"Configurar despliegue continuo"**
   (*Set up continuous deployment*).
3. **Repository provider**: GitHub → **Authenticate** → autoriza la app de
   Cloud Build en tu cuenta de GitHub.
4. Selecciona tu repositorio → **Next**.
5. Configuración de la build:
   - **Branch**: `^main$`
   - **Build type**: **Dockerfile**
   - **Source location**: `/Dockerfile`
6. **Save** → **Deploy**.

Cloud Run crea automáticamente un trigger de Cloud Build y un repositorio en
Artifact Registry (`cloud-run-source-deploy`). Desde ese momento, cada push a
`main` genera una nueva revisión.

### Opción B — Con el `cloudbuild.yaml` del repo (por línea de comandos)

El proyecto ya incluye `cloudbuild.yaml` (build → push a Artifact Registry →
deploy a Cloud Run). Para engancharlo a GitHub:

```bash
# 1. Conecta el repositorio de GitHub (abre el navegador para autorizar)
gcloud builds connections create github github-conn --region=$REGION
gcloud builds repositories create bank-marketing-api \
  --remote-uri=https://github.com/TU_USUARIO/bank-marketing-api.git \
  --connection=github-conn --region=$REGION

# 2. Dar permisos a Cloud Build para desplegar en Cloud Run
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# 3. Crear el trigger sobre la rama main
gcloud builds triggers create github \
  --name=deploy-bank-api \
  --region=$REGION \
  --repository=projects/$PROJECT_ID/locations/$REGION/connections/github-conn/repositories/bank-marketing-api \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

Probar el flujo completo:

```bash
git commit --allow-empty -m "trigger deploy"
git push
gcloud builds list --region=$REGION --limit=3
```

---

## 7. Verificación final

```bash
# URL del servicio
gcloud run services describe bank-marketing-api --region $REGION --format="value(status.url)"

# Pruebas automáticas contra producción
python client/test_client.py --url $(gcloud run services describe bank-marketing-api --region $REGION --format="value(status.url)")

# Logs en vivo
gcloud run services logs tail bank-marketing-api --region $REGION
```

Checklist de entrega:

- [ ] `$URL/health` responde `{"status":"ok","model_loaded":true,...}`
- [ ] `$URL/docs` carga Swagger UI (pruebas manuales del docente)
- [ ] `POST $URL/predict` devuelve predicción y probabilidades
- [ ] `python client/test_client.py --url $URL` termina sin fallos

---

## 8. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `PERMISSION_DENIED: Cloud Build API has not been used` | Falta el paso 3: `gcloud services enable ...` |
| `Revision is not ready... failed to start and listen on PORT` | El contenedor debe escuchar en `$PORT` (0.0.0.0). El `Dockerfile` ya lo hace; revisa que no hayas fijado el puerto a mano |
| `denied: Permission artifactregistry.repositories.uploadArtifacts denied` | Falta crear el repo (paso 4) o el rol `artifactregistry.writer` al service account |
| `Memory limit exceeded` en los logs | Sube a `--memory 2Gi` |
| El build falla con `must specify a logs bucket` | Ya está resuelto en `cloudbuild.yaml` con `options.logging: CLOUD_LOGGING_ONLY` |
| La primera petición tarda varios segundos | *Cold start* de Cloud Run. Se puede mitigar con `--min-instances 1` (consume free tier más rápido) |
| `ModuleNotFoundError: app` dentro del contenedor | El `Dockerfile` copia `app/` a `/app/app`. No muevas archivos fuera del paquete `app/` |

---

## 9. Limpieza (para no gastar cuota)

```bash
gcloud run services delete bank-marketing-api --region $REGION
gcloud artifacts repositories delete bank-repo --location $REGION
```

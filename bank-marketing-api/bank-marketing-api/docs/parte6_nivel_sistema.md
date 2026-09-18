# Parte 6 — Nivel de implementación del sistema

## Nivel elegido: **2 — Recomendación**

### Qué hace exactamente el sistema

La API recibe los datos de un cliente y devuelve:

- una clase predicha (`yes` / `no`),
- la probabilidad de ambas clases,
- un nivel de confianza (`alta` / `media` / `baja`).

No llama a nadie, no modifica el CRM, no descarta clientes de la campaña y no
ejecuta ninguna acción comercial. El destinatario de la salida es una persona
(o un sistema de gestión de campañas operado por personas) que decide qué hacer
con esa recomendación: típicamente **ordenar la lista de llamadas del día**
poniendo arriba a los clientes con mayor probabilidad de contratar.

### Por qué no es nivel 1 (Predicción)

Un sistema de nivel 1 se queda en el número: produce una estimación que se
consume en un reporte o en un análisis, sin una acción asociada. Aquí el output
está diseñado para influir en una decisión operativa concreta y recurrente
(a quién se llama primero), y por eso el sistema devuelve además la
probabilidad y un nivel de confianza: son insumos para decidir, no solo para
describir. Eso ya es una recomendación.

### Por qué no es nivel 3 (Acción parcial) ni 4 (Acción automática)

Sería nivel 3 si la API, además de predecir, escribiera directamente en el
marcador telefónico o generara automáticamente la cola de llamadas y un humano
solo la aprobara. Sería nivel 4 si el sistema disparara la llamada, enviara la
oferta o excluyera clientes de la campaña sin intervención humana.

No estamos en ninguno de los dos casos: el endpoint `POST /predict` es
*stateless*, no tiene efectos secundarios y no se integra con ningún sistema
que ejecute acciones. Subir de nivel no es un cambio de modelo, es un cambio de
arquitectura y de gobernanza.

---

## Implicaciones de riesgo

### 1. Riesgos derivados del rendimiento del modelo

Con el umbral calibrado (0.63), en el conjunto de test el modelo obtiene
F1 ≈ 0.52, balanced accuracy ≈ 0.75 y ROC-AUC ≈ 0.81. Traducido al negocio:

| Error | Qué significa | Costo |
|---|---|---|
| Falso positivo (~precisión 0.47) | Se llama a alguien que no contrata | Tiempo del agente; molestia al cliente |
| Falso negativo (~recall 0.59) | Un cliente interesado queda al final de la lista | Ingreso perdido |

Aproximadamente **1 de cada 2 clientes recomendados no contrata**. Esto es
aceptable para priorizar una lista (la tasa base es 11%, así que el modelo
multiplica por ~4 la eficiencia del contacto), pero sería inaceptable como
criterio automático de exclusión. Es el argumento central para quedarse en
nivel 2.

### 2. Riesgo de que la recomendación se convierta en decisión

El riesgo más realista del nivel 2 no es técnico sino organizacional: que el
equipo deje de mirar la lista y trate el orden como una orden. Si nadie llama
nunca a los clientes del final, el sistema se comporta *de facto* como nivel 3
sin haber pasado por ninguna revisión. Mitigación: reservar una fracción
aleatoria de llamadas fuera del ranking (exploración), lo que además genera
datos no sesgados para reentrenar.

### 3. Sesgo y equidad

El modelo usa `age`, `job`, `education` y `marital` como variables predictoras.
Son legales en este contexto, pero implican que la priorización puede
concentrarse sistemáticamente en ciertos grupos (por ejemplo, jubilados y
estudiantes tienen tasas de aceptación muy superiores). En un despliegue real
habría que:

- medir la tasa de contacto por grupo protegido y documentarla,
- verificar que la priorización no degenere en exclusión de segmentos,
- guardar trazabilidad de cada recomendación (el output incluye
  `model_version` y `threshold` justamente para eso).

### 4. Data leakage y validez en producción

`duration` se excluyó del modelo porque solo se conoce al terminar la llamada.
Incluirla habría inflado las métricas (su correlación con el objetivo es
altísima) y habría producido un modelo **inservible en producción**, porque en
el momento de decidir a quién llamar ese dato todavía no existe. Las métricas
reportadas son, por tanto, más bajas pero honestas.

### 5. Drift del contexto macroeconómico

Cinco variables (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`,
`euribor3m`, `nr.employed`) describen la coyuntura de 2008-2010. No son
leakage —se conocen antes de llamar— pero sí atan el modelo a un régimen
económico concreto: con un euríbor actual muy distinto al del entrenamiento,
las predicciones se degradan. Mitigación: monitorear la distribución de
entrada, reentrenar periódicamente y versionar el modelo
(`GET /model-info` expone versión, hiperparámetros y fecha de entrenamiento).

### 6. Riesgos operativos del despliegue

- **Disponibilidad**: Cloud Run escala a cero; la primera petición tras un
  período inactivo paga un *cold start*. El modelo se carga en el `lifespan`
  de FastAPI, no en el primer request, para acotarlo.
- **Seguridad**: el servicio está desplegado con `--allow-unauthenticated`
  porque la evaluación exige una URL pública. En producción se pondría detrás
  de autenticación IAM o de un API Gateway, y se aplicaría rate limiting.
- **Privacidad**: la API no persiste los datos que recibe; los logs no
  registran el payload completo. Los datos de entrada son información personal
  de clientes bancarios y su almacenamiento requeriría base legal.

---

## Qué haría falta para subir de nivel

| Para pasar a | Requisito mínimo |
|---|---|
| Nivel 3 (acción parcial) | Integración con el CRM + aprobación humana explícita + auditoría de cada acción + acuerdo de nivel de servicio |
| Nivel 4 (acción automática) | Todo lo anterior, más: monitoreo de drift en línea, *kill switch*, pruebas A/B contra un grupo de control, umbral mucho más conservador (alta precisión) y revisión de cumplimiento normativo |

Con el rendimiento actual del modelo, **subir a nivel 4 no sería responsable**:
la precisión de ~0.47 implica que la mitad de las acciones automáticas serían
contactos innecesarios.

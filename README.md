# TP1 - Sistema de Reconocimiento Facial

Sistema completo de detección, alineación, extracción de embeddings e identificación/verificación facial. Incluye una API asincrónica (FastAPI), persistencia en PostgreSQL + pgvector y una interfaz web (Gradio).

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

**Integrantes**

| Alumna | Legajo |
|---|---|
| Jauregui, Lucía | J-0627/1 |
| Nebbia, María Sofía | N-1301/3 |

## Resumen del sistema

```text
Imagen → SCRFD (detección + 5 landmarks) → alineación (norm_crop a 112×112)
       → ArcFace w600k_r50 (embedding de 512 dim., normalizado L2)
       → similitud coseno contra la galería (PostgreSQL + pgvector)
       → identidad si similitud ≥ 0,328; en otro caso, "unknown"
```

- **Detector:** SCRFD (`det_10g.onnx`, de InsightFace `buffalo_l`).
- **Modelo de embeddings:** ArcFace ResNet-50 (`w600k_r50.onnx`), embedding de 512 dimensiones.
- **Umbral de similitud:** `SIMILARITY_THRESHOLD=0.328`.

## Modelo (link público de descarga)

Los modelos no se versionan en el repositorio (ver `.gitignore`). Se descargan desde este link público de solo lectura:

**https://drive.google.com/drive/folders/1JVVlcxy7tzFmxelH8kpHnZu-Q_fQNlX9?usp=sharing**

Archivos que hay que copiar a la carpeta `models/` del repositorio:

- `w600k_r50.onnx` (embeddings ArcFace)
- `det_10g.onnx` (detector SCRFD)

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── services/
│       │   ├── face_service.py  # Detección, alineación y embeddings (implementado por el grupo)
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configuración del modelo

1. Descargá `w600k_r50.onnx` y `det_10g.onnx` desde el [link público](https://drive.google.com/drive/folders/1JVVlcxy7tzFmxelH8kpHnZu-Q_fQNlX9?usp=sharing) y copialos a la carpeta `models/`.
2. Los `.env.*.example` ya vienen configurados para estos modelos (`MODEL_NAME=w600k_r50.onnx`, `DETECTOR_MODEL_NAME=det_10g.onnx`, `EMBEDDING_DIM=512`, `FACE_SIZE=112`, `SIMILARITY_THRESHOLD=0.328`).

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Verificá el archivo **.env.docker.example** y, luego, desde la terminal:

```bash
docker compose build
docker compose up -d
```

### 2. Comprobá que funciona

- Estado del backend: `http://localhost:8000/health`
- Interfaz: `http://localhost:8080`

**Notas:**

- **La primera construcción (`docker compose build`) puede tardar entre 30 y 60 minutos** por la instalación de las dependencias. Las siguientes son mucho más rápidas.
- Si el backend falla en el primer arranque porque PostgreSQL todavía se está inicializando, volvé a correr `docker compose up -d`.
- Las credenciales de base de datos de `.env.docker.example` son **de ejemplo, solo para desarrollo local**.

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo *.env.local.example* a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME` (opcional, archivo `.pth`/`.onnx` en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE` y `ALIGN_MARGIN`.
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).


### Implementado por el grupo (`src/lib/services/face_service.py`)

1. `detect_faces`: detección de rostros con SCRFD (`det_10g.onnx`). Devuelve las cajas `(x1, y1, x2, y2)`.
2. `align_face`: recorta la cara con un margen (`ALIGN_MARGIN`), vuelve a detectar sobre el recorte para obtener los 5 landmarks y alinea con `norm_crop` a `FACE_SIZE` (112×112). Devuelve un `AlignedFace` con los keypoints relativos a la caja.
3. `extract_embedding_from_face`: convierte BGR a RGB, normaliza `(x − 127,5) / 127,5`, ejecuta `w600k_r50.onnx` y normaliza el embedding de 512 dimensiones con L2.

Este pipeline es el mismo que se usó en la notebook 1 para alinear el dataset. Se verificó comparando las caras y los embeddings de ambos: la similitud coseno media fue 0,992 (mínima 0,967).

Ver detalles y recomendaciones en [ENUNCIADO.md](ENUNCIADO.md#4-parte-3--integración-en-la-aplicación).

## Modelo y fine-tuning

Se compararon dos modelos sobre los mismos conjuntos de datos (ver `2-train.ipynb`). El modelo elegido se puede descargar desde el [link público de Drive](https://drive.google.com/drive/folders/1JVVlcxy7tzFmxelH8kpHnZu-Q_fQNlX9?usp=sharing).

- **Modelo A — CNN propia** (2,39 M de parámetros, embedding de 128 dimensiones), entrenada desde cero con augmentation, pesos por clase en la loss y AdamW. Experimento 1: softmax + pesos por clase + label smoothing. Experimento 2: pérdida ArcFace (s = 32, m = 0,3).
- **Modelo B — ArcFace preentrenado** (`w600k_r50`, ResNet-50, 512 dimensiones), sin reentrenar.

| Métrica (test) | A — CNN propia | B — ArcFace |
|---|---|---|
| Identificación Top-1 (closed-set, 62 personas) | 80,3% | **100%** |
| Verificación: AUC | 0,889 | **1,000** |
| Verificación: EER | 18,9% | **0,01%** |
| Verificación: TAR@FAR=1e-3 | 11% | **99,99%** |
| Clustering: silhouette / ARI | 0,259 / 0,581 | **0,607 / 1,0** |
| Tamaño / tiempo por imagen | 9,6 MB / 71 ms | 174,4 MB / 407 ms |

**Modelo elegido: ArcFace (B).** Es mucho más preciso y generaliza a personas que no vio en el entrenamiento. El costo es un modelo más pesado y más lento, que es aceptable para esta aplicación. La CNN propia, entrenada con unas 2.000 imágenes de 43 personas, no logra la misma separación entre identidades.

**Umbral de similitud:** `τ = 0,328`, elegido con las parejas de **validación** (open-set de validación + desconocidos de validación) para un FPIR ≤ 1%, y verificado después en test. Con ese umbral, ArcFace identifica correctamente al 100% de las personas registradas y rechaza al 99,5% de los desconocidos.

**Análisis de errores (FP/FN):** los errores de la CNN son confusiones entre personas de aspecto parecido (mismo género, edad y rasgos). Las mujeres son solo el 11,4% de las consultas de test, reflejo del desbalance del dataset; por eso se analizaron los errores por género. El detalle está en `2-train.ipynb`.

## Dataset

- **Fuente y licencia:** LFW (*Labeled Faces in the Wild*), cargado con `sklearn.datasets.fetch_lfw_people(min_faces_per_person=20)`. Se distribuye para investigación y uso académico. Las personas no dieron un consentimiento específico para entrenar sistemas de reconocimiento facial (ver la sección de ética en `1-pre-processing.ipynb`).
- **Cantidad por persona:** 62 identidades y 3.023 imágenes (mínimo 20, mediana 31, máximo 530 por persona). Después del control de calidad quedan **2.977 caras alineadas**.
- **Balance de clases:** muy desbalanceado: George W. Bush aporta 1 de cada 5,7 fotos y la relación máximo/mínimo entre identidades es de aproximadamente 26 a 1. Se compensa con data augmentation solo en train y con pesos por clase en la loss.
- **Variaciones:** pose, expresión, iluminación, anteojos y oclusiones parciales (fotos de prensa tomadas en condiciones reales).
- **Reglas de filtrado y calidad:**
  1. Imágenes sin cara o con más de una: 23 descartadas.
  2. Imágenes borrosas (varianza del Laplaciano, criterio de outliers sobre el logaritmo): 17 descartadas.
  3. Duplicados y casi duplicados (pHash, distancia de Hamming ≤ 2): 5 descartadas.
  4. Una foto con etiqueta dudosa, descartada tras revisarla visualmente.
  5. Dos fotos descartadas por fuga de datos entre train y evaluación.
- **Particiones (70/15/15):** *closed-set* (por foto, estratificado por identidad) y *open-set* (por identidad: 43 / 9 / 10 personas). Además, un conjunto de **desconocidos** formado por identidades de LFW con menos de 20 fotos: 195 imágenes en validación y 201 en test.
- Los datos y las caras alineadas no se versionan (`data/` está en `.gitignore`).


## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Se descargan del [link público](https://drive.google.com/drive/folders/1JVVlcxy7tzFmxelH8kpHnZu-Q_fQNlX9?usp=sharing) y se colocan en `models/`.
- ArcFace se preentrenó con WebFace600K, que puede incluir identidades de LFW, por lo que sus métricas pueden ser optimistas (ver las notebooks).
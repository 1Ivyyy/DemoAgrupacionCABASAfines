# Demo CABAS Afines

Demo en Python para agrupar Comunidades Academicas Basicas (CABAS) por afinidad tematica y describir cada grupo mediante sus terminos representativos. El campo institucional usado como referencia es:

> Ingenieria aplicada, tecnologias convergentes y sostenibilidad de organizaciones y territorios.

## Estructura

- `data/cabas.csv`: dataset inicial con los nombres de las CABAS.
- `src/cabas_afinidad_demo.py`: script principal.
- `outputs/`: resultados generados al ejecutar la demo.
- `INFORME.md`: documentacion metodologica y registro del proceso.
- `requirements.txt`: dependencias para ejecutar con BERTopic + Sentence Transformers.
- `.venv/`: entorno virtual local con las dependencias instaladas.
- `.cache/huggingface/`: cache local para el modelo semantico.

## Instalacion recomendada

Para recrear el entorno desde cero:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

La demo usa el modelo `paraphrase-multilingual-MiniLM-L12-v2` de Sentence Transformers.

## Ejecucion

Con Docker, sin instalar Python ni dependencias:

```bash
docker compose up --build -d
```

La interfaz queda disponible en `http://localhost:8501`. La guia completa esta
en `README.Docker.md`.

Interfaz grafica:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME=.cache/huggingface .venv/bin/streamlit run src/app.py
```

Con BERTopic + Sentence Transformer:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME=.cache/huggingface .venv/bin/python src/cabas_afinidad_demo.py
```

Con porcentaje de afinidad/tolerancia desde consola:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME=.cache/huggingface .venv/bin/python src/cabas_afinidad_demo.py --mode threshold --tolerance 60
```

Para validar la demo sin usar BERTopic, forzando el respaldo local de afinidad tematica por palabras semilla:

```bash
.venv/bin/python src/cabas_afinidad_demo.py --force-fallback
```

## Resultados

La ejecucion produce:

- `outputs/agrupacion_cabas.csv`: cada CABA con su identificador de grupo, definicion, terminos representativos y afinidad promedio interna.
- `outputs/agrupacion_cabas_afinidades.csv`: matriz de afinidad porcentual entre CABAS.
- `outputs/resumen_agrupacion.md`: reporte corto generado automaticamente con los grupos encontrados.

La ultima ejecucion confirmada uso `BERTopic + SentenceTransformer`.

## Modos de agrupacion

- Agrupacion automatica: usa BERTopic + Sentence Transformer para detectar grupos semanticos.
- Agrupacion definiendo tolerancia: usa embeddings semanticos y agrupa CABAS conectadas por una afinidad igual o superior al porcentaje elegido.

En la interfaz grafica se puede cargar un CSV, elegir el modo, ajustar el porcentaje con un slider y revisar la afinidad interna de cada grupo en una matriz interactiva.

## Adaptacion a datos reales

Para usar CABAS reales, reemplace `data/cabas.csv`. La unica columna requerida es:

- `nombre`

Cada nombre debe estar diligenciado y no puede repetirse. No se requiere `id`,
descripcion ni palabras clave porque el agrupamiento utiliza directamente el
nombre de cada CABA.

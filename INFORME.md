# Informe tecnico: Demo de agrupacion de CABAS afines

## 1. Objetivo

El objetivo de la demo es identificar afinidades entre CABAS y agruparlas en conjuntos tematicos que permitan sustentar la conformacion de una Escuela alrededor del campo:

**Ingenieria aplicada, tecnologias convergentes y sostenibilidad de organizaciones y territorios.**

La demo toma una lista de CABAS, construye una representacion textual de cada una, calcula similitud semantica y asigna cada CABA a un grupo. Luego presenta los terminos representativos de cada agrupacion.

## 2. Contexto estatutario usado

El archivo `agents.md` define tres conceptos que orientan el diseno:

- Campo de conocimiento-saber: escenario para abordar problemas y producir conocimientos mediante investigacion-creacion.
- Escuela: unidad academico-administrativa que integra docentes y productividad academica alrededor de un campo.
- CABA: estructura basica, dinamica y flexible organizada por intereses cognitivos especificos dentro del campo.

Por eso la demo no solo agrupa por palabras parecidas. Tambien redacta cada grupo como una capacidad academica integrada que puede aportar a docencia, investigacion-creacion, innovacion, contexto-extension y proyeccion social.

## 3. Insumos creados

Se creo el archivo `data/cabas.csv` con las 15 CABAS listadas en `agents.md`:

1. Ciencias básicas aplicadas a la ingeniería
2. Organizaciones y proyectos
3. Catastro, Gestión Territorial y Avalúos
4. Geodesia y geotecnologías aplicadas al territorio
5. Electrónica, control y automatización
6. Redes inteligentes, telecomunicaciones y seguridad informática
7. Ciencias de sistemas (Cibernética e IA) y computación
8. Ingeniería de software, ciencia de datos y programación de computadores
9. Gestión de TI y transformación digital
10. Sistemas eléctricos de energía y potencia
11. Ciencia de datos para la toma de decisiones en sistemas organizacionales
12. Investigación de operaciones en sistemas organizacionales
13. Gestión de operaciones y cadenas de suministro
14. Ingeniería de productos, materiales y procesos
15. Humanidades en ingeniería

Cada fila contiene unicamente:

- `nombre`: nombre completo y unico de la CABA.

El identificador no es necesario para calcular embeddings, afinidades o grupos.
La posicion de la fila y el nombre permiten relacionar cada resultado con su
CABA de origen.

## 4. Herramientas y dependencias

La demo esta preparada para usar:

- Python.
- BERTopic.
- Sentence Transformers.
- `paraphrase-multilingual-MiniLM-L12-v2` como modelo semantico multilingue.
- UMAP para reduccion dimensional.
- HDBSCAN para agrupamiento basado en densidad.
- Streamlit para la interfaz de usuario.
- Plotly para visualizar matrices de afinidad internas.
- Docker y Docker Compose para empaquetar y ejecutar la aplicacion.

En el entorno del proyecto las dependencias fueron instaladas en el entorno virtual local `.venv`. El modelo `paraphrase-multilingual-MiniLM-L12-v2` tambien fue descargado y validado con una dimension de embedding de 384.

Tambien incluye un modo de respaldo local:

- Palabras semilla por eje tematico.
- TF-IDF para extraer terminos representativos de cada grupo.

Este respaldo existe para que el flujo se pueda ejecutar y verificar aun cuando el equipo no tenga instaladas las dependencias pesadas de BERTopic.

## 5. Proceso implementado

### 5.1 Carga de datos

El script `src/cabas_afinidad_demo.py` lee `data/cabas.csv` con `pandas` y
valida que exista la columna `nombre`. Tambien rechaza nombres vacios o
repetidos, porque el nombre se utiliza como identificador visible en la matriz
de afinidades.

### 5.2 Construccion documental

Para cada CABA se utiliza su nombre completo como documento textual. Sentence
Transformers convierte ese nombre en un embedding semantico que se usa tanto en
BERTopic como en la modalidad por tolerancia.

### 5.3 Embeddings semanticos

Cuando BERTopic esta disponible, el script carga `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")` y calcula embeddings para todos los documentos.

Este modelo es apropiado para una demo en espanol porque es multilingue, liviano y suficiente para capturar cercania semantica entre areas como datos, territorio, energia, operaciones o automatizacion.

Con estos embeddings tambien se calcula una matriz de similitud coseno. Cada valor se expresa como porcentaje de afinidad entre dos CABAS, donde 100% indica maxima cercania semantica bajo el modelo usado.

### 5.4 Agrupamiento con BERTopic

BERTopic combina:

- Embeddings semanticos.
- UMAP para comprimir el espacio vectorial.
- HDBSCAN para detectar grupos.
- Extraccion de terminos representativos por topicos.

La demo configura `min_cluster_size=2` para permitir grupos pequenos, ya que el dataset de prueba es reducido.

### 5.5 Agrupamiento por tolerancia

La version 2 agrega un segundo modo de trabajo: agrupacion definiendo tolerancia. En este modo el usuario elige un porcentaje minimo de afinidad. El sistema calcula la matriz de similitud semantica entre CABAS y conecta aquellas cuyo porcentaje es igual o superior al umbral elegido.

El resultado se construye como grupos de CABAS conectadas por esa regla. Un porcentaje bajo produce grupos mas amplios; un porcentaje alto produce grupos mas estrictos y puede dejar CABAS aisladas.

Este modo esta disponible por consola con:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME=.cache/huggingface .venv/bin/python src/cabas_afinidad_demo.py --mode threshold --tolerance 60
```

### 5.6 Interfaz de usuario

Se agrego una interfaz de usuario en `src/app.py` usando Streamlit. La interfaz permite:

- cargar un archivo CSV de entrada;
- usar el dataset de ejemplo si no se carga archivo;
- elegir entre agrupacion automatica y agrupacion definiendo tolerancia;
- ajustar el porcentaje de afinidad mediante un slider;
- ejecutar la agrupacion con un boton;
- visualizar grupos, terminos representativos y CABAS incluidas;
- ver la afinidad interna de cada grupo mediante una matriz interactiva;
- descargar el CSV de resultados y la matriz de afinidades.

La matriz interactiva permite pasar el cursor sobre una celda para ver la CABA origen, la CABA comparada y el porcentaje de afinidad entre ambas.

### 5.7 Respaldo local

Si BERTopic no esta instalado, o si falla por dependencias del entorno, el script ejecuta automaticamente un respaldo con:

- ejes tematicos definidos a partir del campo de Escuela;
- palabras semilla para asignar afinidad;
- `TfidfVectorizer` para describir los terminos representativos de cada grupo.

Tambien se puede forzar con:

```bash
.venv/bin/python src/cabas_afinidad_demo.py --force-fallback
```

Este respaldo no reemplaza la exigencia metodologica de BERTopic, pero permite validar la estructura de datos, salidas y documentacion en equipos donde aun no se instalaron las librerias finales.

### 5.8 Generacion de salidas

El programa genera:

- `outputs/agrupacion_cabas.csv`: tabla final con identificador de grupo, terminos y afinidad promedio interna.
- `outputs/agrupacion_cabas_afinidades.csv`: matriz porcentual de afinidad entre CABAS.
- `outputs/resumen_agrupacion.md`: reporte automatico de los grupos.

### 5.9 Contenedorizacion

El proyecto incluye una imagen Docker autocontenida basada en Python 3.12. La
imagen instala PyTorch en su variante CPU, las dependencias de
`requirements.txt` y el modelo
`paraphrase-multilingual-MiniLM-L12-v2`.

El modelo se descarga durante la construccion de la imagen y la ejecucion se
configura en modo offline. De esta manera, la maquina destino solo necesita
Docker y no requiere instalar Python, BERTopic, Sentence Transformers o
Streamlit.

La aplicacion se ejecuta con un usuario sin privilegios, expone el puerto 8501
y cuenta con una comprobacion de salud HTTP.

## 6. Verificacion realizada

En el entorno actual se verifico que:

- Python disponible: 3.13.13.
- Entorno virtual creado en `.venv`.
- Dependencias de `requirements.txt` instaladas correctamente.
- `pip check` no reporta dependencias rotas.
- Imports verificados: `bertopic`, `sentence_transformers`, `hdbscan`, `umap`, `torch`, `pandas`, `numpy` y `scikit-learn`.
- Modelo `paraphrase-multilingual-MiniLM-L12-v2` descargado en cache local del proyecto con `HF_HOME=.cache/huggingface`.
- Ejecucion completa de la demo con metodo `BERTopic + SentenceTransformer`.
- Nueva modalidad por tolerancia disponible mediante `--mode threshold --tolerance`.
- Interfaz Streamlit agregada en `src/app.py`.

La ejecucion genero los archivos:

- `outputs/agrupacion_cabas.csv`
- `outputs/agrupacion_cabas_afinidades.csv`
- `outputs/resumen_agrupacion.md`

Tambien se mantiene el modo de respaldo para ejecutar la demo en equipos donde todavia no se hayan instalado BERTopic o el modelo de Sentence Transformers.

## 7. Como ejecutar la demo

Instalacion recomendada:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ejecucion con BERTopic:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME=.cache/huggingface .venv/bin/python src/cabas_afinidad_demo.py
```

Ejecucion con porcentaje de afinidad:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME=.cache/huggingface .venv/bin/python src/cabas_afinidad_demo.py --mode threshold --tolerance 60
```

Ejecucion de la interfaz:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME=.cache/huggingface .venv/bin/streamlit run src/app.py
```

Ejecucion con Docker:

```bash
docker compose up --build -d
```

La interfaz queda disponible en `http://localhost:8501`.

Ejecucion de validacion sin BERTopic:

```bash
.venv/bin/python src/cabas_afinidad_demo.py --force-fallback
```

## 8. Resultado de la ejecucion con BERTopic

La ultima ejecucion se realizo con `BERTopic + SentenceTransformer`. Los grupos se identifican exclusivamente mediante el valor numerico `grupo_id`; el sistema ya no genera nombres tematicos para ellos.

El coeficiente silhouette calculado sobre los textos fue `0.008`. Este valor debe interpretarse con cautela porque el dataset es pequeno y las CABAS son deliberadamente transversales; el objetivo principal de la demo es mostrar el flujo metodologico y producir una primera agrupacion interpretable, no cerrar una clasificacion institucional definitiva.

BERTopic marco algunas CABAS como `-1`, que corresponde a elementos sin grupo estable bajo HDBSCAN. Esto es esperable en datasets pequenos y es util para identificar CABAS que requieren revision experta, un nombre mas especifico o reasignacion manual.

## 9. Interpretacion esperada

Los grupos generados deben leerse como aproximaciones tecnicas. Para una sustentacion formal de Escuela, se recomienda una fase posterior con:

- Validacion de expertos academicos.
- Revision de la composicion de los grupos y posibles fusiones o separaciones.
- Revision de coherencia frente a productividad docente.
- Incorporacion de proyectos, publicaciones, semilleros, laboratorios y lineas de investigacion.
- Comparacion contra las funciones de investigacion-creacion, innovacion, docencia, extension y proyeccion social.

## 10. Archivos entregados

- `README.md`: guia rapida del proyecto.
- `requirements.txt`: dependencias.
- `data/cabas.csv`: datos de ejemplo.
- `src/cabas_afinidad_demo.py`: implementacion.
- `src/app.py`: interfaz de usuario.
- `Dockerfile`: definicion de la imagen autocontenida.
- `compose.yaml`: ejecucion local mediante Docker Compose.
- `.dockerignore`: exclusion del entorno virtual, cache y salidas del contexto.
- `README.Docker.md`: guia de construccion, ejecucion y distribucion.
- `outputs/agrupacion_cabas.csv`: se genera al ejecutar.
- `outputs/agrupacion_cabas_afinidades.csv`: se genera al ejecutar.
- `outputs/resumen_agrupacion.md`: se genera al ejecutar.
- `.venv`: entorno virtual local con las dependencias instaladas.
- `.cache/huggingface`: cache local usada para el modelo semantico.

## 11. Limitaciones

- El dataset es demostrativo, no definitivo.
- Los nombres cortos, ambiguos o compuestos principalmente por siglas pueden reducir la precision semantica del agrupamiento.
- BERTopic y Sentence Transformers requieren descargas grandes, especialmente por PyTorch.
- La ejecucion con BERTopic puede producir grupos `-1` cuando HDBSCAN no encuentra densidad suficiente para asignar todos los documentos.
- El porcentaje de tolerancia debe revisarse con criterio experto: no existe un umbral universal valido para todos los conjuntos de CABAS.

# Informe tecnico: Demo de agrupacion de CABAS afines

## 1. Objetivo

El objetivo de la demo es identificar afinidades entre CABAS y agruparlas en conjuntos tematicos que permitan sustentar la conformacion de una Escuela alrededor del campo:

**Ingenieria aplicada, tecnologias convergentes y sostenibilidad de organizaciones y territorios.**

La demo toma una lista de CABAS, construye una representacion textual de cada una, calcula similitud semantica y asigna cada CABA a un grupo. Luego genera un nombre y una definicion para cada agrupacion.

## 2. Contexto estatutario usado

El archivo `agents.md` define tres conceptos que orientan el diseno:

- Campo de conocimiento-saber: escenario para abordar problemas y producir conocimientos mediante investigacion-creacion.
- Escuela: unidad academico-administrativa que integra docentes y productividad academica alrededor de un campo.
- CABA: estructura basica, dinamica y flexible organizada por intereses cognitivos especificos dentro del campo.

Por eso la demo no solo agrupa por palabras parecidas. Tambien redacta cada grupo como una capacidad academica integrada que puede aportar a docencia, investigacion-creacion, innovacion, contexto-extension y proyeccion social.

## 3. Insumos creados

Se creo el archivo `data/cabas.csv` con 20 CABAS de ejemplo. Incluye las 15 CABAS listadas en `agents.md` y 5 adicionales para dar mas densidad tematica:

- Energias renovables y sostenibilidad energetica.
- Tecnologias ambientales y gestion del recurso hidrico.
- Analitica territorial y ciudades inteligentes.
- Ingenieria de seguridad y resiliencia de infraestructura.
- Bioingenieria y tecnologias para la salud.

Cada fila contiene:

- `id`: identificador.
- `nombre`: nombre de la CABA.
- `descripcion`: descripcion corta de su alcance academico.
- `palabras_clave`: terminos orientadores.

## 4. Herramientas y dependencias

La demo esta preparada para usar:

- Python.
- BERTopic.
- Sentence Transformers.
- `paraphrase-multilingual-MiniLM-L12-v2` como modelo semantico multilingue.
- UMAP para reduccion dimensional.
- HDBSCAN para agrupamiento basado en densidad.

En el entorno del proyecto las dependencias fueron instaladas en el entorno virtual local `.venv`. El modelo `paraphrase-multilingual-MiniLM-L12-v2` tambien fue descargado y validado con una dimension de embedding de 384.

Tambien incluye un modo de respaldo local:

- Palabras semilla por eje tematico.
- TF-IDF para extraer terminos representativos de cada grupo.

Este respaldo existe para que el flujo se pueda ejecutar y verificar aun cuando el equipo no tenga instaladas las dependencias pesadas de BERTopic.

## 5. Proceso implementado

### 5.1 Carga de datos

El script `src/cabas_afinidad_demo.py` lee `data/cabas.csv` con `pandas` y valida que existan las columnas requeridas:

- `nombre`
- `descripcion`
- `palabras_clave`

### 5.2 Construccion documental

Para cada CABA se construye un documento textual uniendo:

- Nombre.
- Descripcion.
- Palabras clave.

Esto mejora el agrupamiento porque el modelo no depende solo del nombre de la CABA, que suele ser corto.

### 5.3 Embeddings semanticos

Cuando BERTopic esta disponible, el script carga `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")` y calcula embeddings para todos los documentos.

Este modelo es apropiado para una demo en espanol porque es multilingue, liviano y suficiente para capturar cercania semantica entre areas como datos, territorio, energia, operaciones o automatizacion.

### 5.4 Agrupamiento con BERTopic

BERTopic combina:

- Embeddings semanticos.
- UMAP para comprimir el espacio vectorial.
- HDBSCAN para detectar grupos.
- Extraccion de terminos representativos por topicos.

La demo configura `min_cluster_size=2` para permitir grupos pequenos, ya que el dataset de prueba es reducido.

### 5.5 Respaldo local

Si BERTopic no esta instalado, o si falla por dependencias del entorno, el script ejecuta automaticamente un respaldo con:

- ejes tematicos definidos a partir del campo de Escuela;
- palabras semilla para asignar afinidad;
- `TfidfVectorizer` para describir los terminos representativos de cada grupo.

Tambien se puede forzar con:

```bash
.venv/bin/python src/cabas_afinidad_demo.py --force-fallback
```

Este respaldo no reemplaza la exigencia metodologica de BERTopic, pero permite validar la estructura de datos, salidas y documentacion en equipos donde aun no se instalaron las librerias finales.

### 5.6 Nombramiento de grupos

Despues del agrupamiento, el script extrae terminos representativos por grupo usando TF-IDF. Luego aplica reglas interpretables para proponer nombres como:

- Territorio, geotecnologias y ciudades inteligentes.
- Computacion, datos e inteligencia artificial aplicada.
- Energia, potencia y transicion sostenible.
- Gestion de organizaciones, operaciones y proyectos.
- Sistemas electronicos, conectividad y automatizacion.
- Sostenibilidad, ambiente y sociedad.
- Productos, materiales y procesos industriales.

Estas etiquetas se pueden ajustar con criterio experto antes de una presentacion institucional.

### 5.7 Generacion de salidas

El programa genera:

- `outputs/agrupacion_cabas.csv`: tabla final con grupo, nombre, definicion y terminos.
- `outputs/resumen_agrupacion.md`: reporte automatico de los grupos.

## 6. Verificacion realizada

En el entorno actual se verifico que:

- Python disponible: 3.13.13.
- Entorno virtual creado en `.venv`.
- Dependencias de `requirements.txt` instaladas correctamente.
- `pip check` no reporta dependencias rotas.
- Imports verificados: `bertopic`, `sentence_transformers`, `hdbscan`, `umap`, `torch`, `pandas`, `numpy` y `scikit-learn`.
- Modelo `paraphrase-multilingual-MiniLM-L12-v2` descargado en cache local del proyecto con `HF_HOME=.cache/huggingface`.
- Ejecucion completa de la demo con metodo `BERTopic + SentenceTransformer`.

La ejecucion genero los archivos:

- `outputs/agrupacion_cabas.csv`
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

Ejecucion de validacion sin BERTopic:

```bash
.venv/bin/python src/cabas_afinidad_demo.py --force-fallback
```

## 8. Resultado de la ejecucion con BERTopic

La ultima ejecucion se realizo con `BERTopic + SentenceTransformer`. El archivo `outputs/resumen_agrupacion.md` reporto los siguientes grupos:

- Grupo -1: CABAS sin grupo estable.
- Grupo 0: Gestion de organizaciones, operaciones y proyectos.
- Grupo 1: Computacion, datos e inteligencia artificial aplicada.
- Grupo 2: Territorio, geotecnologias y ciudades inteligentes.
- Grupo 3: Sistemas electronicos, conectividad y automatizacion.
- Grupo 4: Sistemas electronicos, conectividad y automatizacion.
- Grupo 5: Energia, ambiente y sostenibilidad socio-tecnica.

El coeficiente silhouette calculado sobre los textos fue `0.008`. Este valor debe interpretarse con cautela porque el dataset es pequeno y las CABAS son deliberadamente transversales; el objetivo principal de la demo es mostrar el flujo metodologico y producir una primera agrupacion interpretable, no cerrar una clasificacion institucional definitiva.

BERTopic marco algunas CABAS como `-1`, que corresponde a elementos sin grupo estable bajo HDBSCAN. Esto es esperable en datasets pequenos y es util para identificar CABAS que requieren revision experta, mayor descripcion textual o reasignacion manual.

## 9. Interpretacion esperada

Los grupos generados deben leerse como aproximaciones tecnicas. Para una sustentacion formal de Escuela, se recomienda una fase posterior con:

- Validacion de expertos academicos.
- Ajuste de nombres de grupos y posible fusion de grupos con etiquetas repetidas.
- Revision de coherencia frente a productividad docente.
- Incorporacion de proyectos, publicaciones, semilleros, laboratorios y lineas de investigacion.
- Comparacion contra las funciones de investigacion-creacion, innovacion, docencia, extension y proyeccion social.

## 10. Archivos entregados

- `README.md`: guia rapida del proyecto.
- `requirements.txt`: dependencias.
- `data/cabas.csv`: datos de ejemplo.
- `src/cabas_afinidad_demo.py`: implementacion.
- `outputs/agrupacion_cabas.csv`: se genera al ejecutar.
- `outputs/resumen_agrupacion.md`: se genera al ejecutar.
- `.venv`: entorno virtual local con las dependencias instaladas.
- `.cache/huggingface`: cache local usada para el modelo semantico.

## 11. Limitaciones

- El dataset es demostrativo, no definitivo.
- Los nombres de grupos son propuestas automatizadas y deben revisarse institucionalmente.
- La calidad mejora si cada CABA tiene descripciones mas extensas, productividad academica asociada y palabras clave normalizadas.
- BERTopic y Sentence Transformers requieren descargas grandes, especialmente por PyTorch.
- La ejecucion con BERTopic puede producir grupos `-1` cuando HDBSCAN no encuentra densidad suficiente para asignar todos los documentos.

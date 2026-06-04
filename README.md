# Demo CABAS Afines

Demo en Python para agrupar Comunidades Academicas Basicas (CABAS) por afinidad tematica y proponer un nombre/definicion para cada grupo. El campo institucional usado como referencia es:

> Ingenieria aplicada, tecnologias convergentes y sostenibilidad de organizaciones y territorios.

## Estructura

- `data/cabas.csv`: dataset inicial de CABAS, descripciones y palabras clave.
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

Con BERTopic + Sentence Transformer:

```bash
HF_HUB_DISABLE_XET=1 HF_HOME=.cache/huggingface .venv/bin/python src/cabas_afinidad_demo.py
```

Para validar la demo sin usar BERTopic, forzando el respaldo local de afinidad tematica por palabras semilla:

```bash
.venv/bin/python src/cabas_afinidad_demo.py --force-fallback
```

## Resultados

La ejecucion produce:

- `outputs/agrupacion_cabas.csv`: cada CABA con su grupo, nombre de grupo, definicion y terminos representativos.
- `outputs/resumen_agrupacion.md`: reporte corto generado automaticamente con los grupos encontrados.

La ultima ejecucion confirmada uso `BERTopic + SentenceTransformer`.

## Adaptacion a datos reales

Para usar CABAS reales, reemplace `data/cabas.csv` conservando estas columnas:

- `id`
- `nombre`
- `descripcion`
- `palabras_clave`

Mientras mas descriptiva sea la columna `descripcion`, mejor sera la calidad del agrupamiento.

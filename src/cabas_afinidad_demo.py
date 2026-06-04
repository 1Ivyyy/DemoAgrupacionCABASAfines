from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "cabas.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "agrupacion_cabas.csv"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "resumen_agrupacion.md"

CAMPO_ESCUELA = (
    "Ingenieria aplicada, tecnologias convergentes y sostenibilidad de "
    "organizaciones y territorios"
)


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: int
    nombre: str
    definicion: str
    terminos: list[str]
    cabas: list[str]


def normalize_text(value: str) -> str:
    return " ".join(str(value).replace("_", " ").split())


def build_documents(df: pd.DataFrame) -> list[str]:
    required = {"nombre", "descripcion", "palabras_clave"}
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"El archivo de entrada no tiene columnas requeridas: {missing_cols}")

    return [
        normalize_text(f"{row.nombre}. {row.descripcion}. {row.palabras_clave}")
        for row in df.itertuples(index=False)
    ]


def top_terms_by_cluster(
    documents: list[str], labels: Iterable[int], top_n: int = 6
) -> dict[int, list[str]]:
    labels_array = np.array(list(labels))
    vectorizer = TfidfVectorizer(
        stop_words=spanish_stop_words(),
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.90,
    )
    matrix = vectorizer.fit_transform(documents)
    terms = np.array(vectorizer.get_feature_names_out())

    result: dict[int, list[str]] = {}
    for cluster_id in sorted(set(labels_array)):
        if cluster_id == -1:
            continue
        rows = matrix[labels_array == cluster_id]
        weights = np.asarray(rows.mean(axis=0)).ravel()
        best = weights.argsort()[::-1][:top_n]
        result[int(cluster_id)] = [terms[i] for i in best if weights[i] > 0]
    return result


def spanish_stop_words() -> list[str]:
    return [
        "a",
        "al",
        "con",
        "de",
        "del",
        "e",
        "el",
        "en",
        "entre",
        "la",
        "las",
        "los",
        "para",
        "por",
        "que",
        "se",
        "un",
        "una",
        "y",
    ]


THEMATIC_GROUPS = [
    {
        "label": "Fundamentos, productos y procesos de ingenieria",
        "keywords": [
            "matematica",
            "fisica",
            "estadistica",
            "modelacion",
            "producto",
            "materiales",
            "manufactura",
            "procesos industriales",
            "ciclo de vida",
        ],
    },
    {
        "label": "Gestion de organizaciones, operaciones y proyectos",
        "keywords": [
            "organizaciones",
            "proyectos",
            "estrategia",
            "operaciones",
            "logistica",
            "abastecimiento",
            "suministro",
            "produccion",
            "optimizacion",
            "teoria de colas",
        ],
    },
    {
        "label": "Territorio, geotecnologias y ciudades inteligentes",
        "keywords": [
            "catastro",
            "territorio",
            "avaluos",
            "predial",
            "geodesia",
            "geotecnologias",
            "sig",
            "sensores remotos",
            "ciudades inteligentes",
            "movilidad",
            "datos urbanos",
        ],
    },
    {
        "label": "Computacion, datos e inteligencia artificial aplicada",
        "keywords": [
            "ia",
            "computacion",
            "cibernetica",
            "software",
            "datos",
            "programacion",
            "analitica",
            "machine learning",
            "bi",
            "transformacion digital",
            "ti",
            "arquitectura empresarial",
        ],
    },
    {
        "label": "Sistemas electronicos, conectividad y automatizacion",
        "keywords": [
            "electronica",
            "control",
            "automatizacion",
            "robotica",
            "instrumentacion",
            "redes",
            "telecomunicaciones",
            "ciberseguridad",
            "iot",
            "infraestructura critica",
            "resiliencia",
        ],
    },
    {
        "label": "Energia, ambiente y sostenibilidad socio-tecnica",
        "keywords": [
            "energia",
            "potencia",
            "redes electricas",
            "renovables",
            "eficiencia",
            "transicion",
            "ambiente",
            "agua",
            "residuos",
            "impacto",
            "recursos",
            "humanidades",
            "etica",
            "sociedad",
            "sostenibilidad",
        ],
    },
    {
        "label": "Bioingenieria y tecnologias para la salud",
        "keywords": [
            "bioingenieria",
            "salud",
            "biomedicos",
            "rehabilitacion",
            "dispositivos",
            "biometria",
        ],
    },
]


def label_from_terms(terms: list[str]) -> str:
    joined = " ".join(terms).lower()
    best_label = "Convergencia aplicada de ingenieria y tecnologia"
    best_score = 0
    for theme in THEMATIC_GROUPS:
        score = sum(1 for keyword in theme["keywords"] if keyword in joined)
        if score > best_score:
            best_score = score
            best_label = str(theme["label"])
    return best_label


def definition_from_terms(label: str, terms: list[str]) -> str:
    key_terms = ", ".join(terms[:4]) if terms else "afinidad tematica"
    return (
        f"Agrupa CABAS con afinidad en {key_terms}. Su aporte al campo "
        f"'{CAMPO_ESCUELA}' consiste en integrar capacidades academicas para "
        "investigacion-creacion, docencia, innovacion y proyeccion social."
    )


def run_bertopic(documents: list[str]) -> tuple[str, list[int], object]:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedding_model.encode(documents, show_progress_bar=False)

    n_neighbors = min(10, max(2, len(documents) - 1))
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        language="multilingual",
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(documents, embeddings)
    return "BERTopic + SentenceTransformer", [int(topic) for topic in topics], topic_model


def run_fallback(documents: list[str]) -> tuple[str, list[int], None]:
    labels: list[int] = []
    for document in documents:
        text = document.lower()
        scores = [
            sum(1 for keyword in theme["keywords"] if keyword_matches(text, keyword))
            for theme in THEMATIC_GROUPS
        ]
        labels.append(int(np.argmax(scores)))
    return "Respaldo local de afinidad tematica por palabras semilla", labels, None


def keyword_matches(text: str, keyword: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def build_cluster_summaries(
    df: pd.DataFrame, documents: list[str], labels: list[int]
) -> list[ClusterSummary]:
    terms_by_cluster = top_terms_by_cluster(documents, labels)
    summaries: list[ClusterSummary] = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            terms = ["sin asignacion estable"]
            nombre = "CABAS sin grupo estable"
        else:
            terms = terms_by_cluster.get(int(cluster_id), [])
            nombre = label_from_terms(terms)
        cabas = df.loc[np.array(labels) == cluster_id, "nombre"].tolist()
        summaries.append(
            ClusterSummary(
                cluster_id=int(cluster_id),
                nombre=nombre,
                definicion=definition_from_terms(nombre, terms),
                terminos=terms,
                cabas=cabas,
            )
        )
    return summaries


def attach_cluster_metadata(
    df: pd.DataFrame, summaries: list[ClusterSummary], labels: list[int]
) -> pd.DataFrame:
    by_id = {summary.cluster_id: summary for summary in summaries}
    output = df.copy()
    output["grupo_id"] = labels
    output["grupo_nombre"] = [by_id[label].nombre for label in labels]
    output["grupo_definicion"] = [by_id[label].definicion for label in labels]
    output["terminos_representativos"] = [
        ", ".join(by_id[label].terminos) for label in labels
    ]
    return output


def silhouette(documents: list[str], labels: list[int]) -> float | None:
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None
    vectorizer = TfidfVectorizer(stop_words=spanish_stop_words(), ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents)
    return float(silhouette_score(matrix, labels, metric="cosine"))


def write_markdown_report(
    path: Path,
    method: str,
    input_file: Path,
    output_file: Path,
    summaries: list[ClusterSummary],
    score: float | None,
) -> None:
    lines = [
        "# Resumen de agrupacion de CABAS",
        "",
        f"Campo de conocimiento-saber objetivo: **{CAMPO_ESCUELA}**.",
        "",
        "## Metodo ejecutado",
        "",
        f"- Metodo: {method}",
        f"- Archivo de entrada: `{input_file}`",
        f"- Archivo de salida: `{output_file}`",
        f"- Coeficiente silhouette sobre textos: {score:.3f}" if score is not None else "- Coeficiente silhouette sobre textos: no aplica",
        "",
        "## Grupos encontrados",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"### Grupo {summary.cluster_id}: {summary.nombre}",
                "",
                summary.definicion,
                "",
                f"Terminos representativos: {', '.join(summary.terminos)}",
                "",
                "CABAS incluidas:",
                *[f"- {caba}" for caba in summary.cabas],
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo para agrupar CABAS afines con BERTopic y Sentence Transformers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Usa el agrupamiento local de respaldo aunque BERTopic este instalado.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    documents = build_documents(df)

    if args.force_fallback:
        method, labels, _ = run_fallback(documents)
    else:
        try:
            method, labels, _ = run_bertopic(documents)
        except Exception as exc:
            method, labels, _ = run_fallback(documents)
            method = f"{method} (BERTopic no disponible: {exc.__class__.__name__})"

    summaries = build_cluster_summaries(df, documents, labels)
    result = attach_cluster_metadata(df, summaries, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    score = silhouette(documents, labels)
    write_markdown_report(args.report, method, args.input, args.output, summaries, score)

    payload = {
        "metodo": method,
        "archivo_salida": str(args.output),
        "reporte": str(args.report),
        "grupos": [
            {
                "id": summary.cluster_id,
                "nombre": summary.nombre,
                "cantidad_cabas": len(summary.cabas),
            }
            for summary in summaries
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

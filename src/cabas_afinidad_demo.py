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
from sklearn.metrics.pairwise import cosine_similarity


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
    definicion: str
    terminos: list[str]
    cabas: list[str]


@dataclass(frozen=True)
class AnalysisResult:
    method: str
    labels: list[int]
    summaries: list[ClusterSummary]
    table: pd.DataFrame
    score: float | None
    affinity_matrix: pd.DataFrame


def normalize_text(value: str) -> str:
    return " ".join(str(value).replace("_", " ").split())


def build_documents(df: pd.DataFrame) -> list[str]:
    if "nombre" not in df.columns:
        raise ValueError("El archivo de entrada debe contener la columna 'nombre'.")

    names = df["nombre"].fillna("").map(normalize_text)
    if names.eq("").any():
        rows = ", ".join(str(index + 2) for index in names[names.eq("")].index)
        raise ValueError(f"La columna 'nombre' tiene valores vacios en las filas: {rows}.")

    duplicated = names[names.duplicated(keep=False)]
    if not duplicated.empty:
        repeated = ", ".join(sorted(duplicated.unique()))
        raise ValueError(f"Los nombres de las CABAS deben ser unicos. Repetidos: {repeated}.")

    return names.tolist()


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


THEMATIC_KEYWORD_GROUPS = [
    {
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


def definition_from_terms(terms: list[str]) -> str:
    key_terms = ", ".join(terms[:4]) if terms else "afinidad tematica"
    return (
        f"Agrupa CABAS con afinidad en {key_terms}. Su aporte al campo "
        f"'{CAMPO_ESCUELA}' consiste en integrar capacidades academicas para "
        "investigacion-creacion, docencia, innovacion y proyeccion social."
    )


def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def encode_documents(documents: list[str], embedding_model=None) -> np.ndarray:
    model = embedding_model or load_embedding_model()
    return np.asarray(model.encode(documents, show_progress_bar=False))


def build_affinity_matrix(names: list[str], embeddings: np.ndarray) -> pd.DataFrame:
    similarity = cosine_similarity(embeddings)
    percentages = np.clip(similarity * 100, 0, 100)
    return pd.DataFrame(percentages, index=names, columns=names).round(2)


def run_bertopic(
    documents: list[str], embeddings: np.ndarray | None = None, embedding_model=None
) -> tuple[str, list[int], object, np.ndarray]:
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    embedding_model = embedding_model or load_embedding_model()
    if embeddings is None:
        embeddings = encode_documents(documents, embedding_model)

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
    return "BERTopic + SentenceTransformer", [int(topic) for topic in topics], topic_model, embeddings


def run_fallback(documents: list[str]) -> tuple[str, list[int], None]:
    labels: list[int] = []
    for document in documents:
        text = document.lower()
        scores = [
            sum(1 for keyword in theme["keywords"] if keyword_matches(text, keyword))
            for theme in THEMATIC_KEYWORD_GROUPS
        ]
        labels.append(int(np.argmax(scores)))
    return "Respaldo local de afinidad tematica por palabras semilla", labels, None


def run_threshold_grouping(
    documents: list[str], tolerance_percent: float, embedding_model=None
) -> tuple[str, list[int], np.ndarray]:
    if not 0 <= tolerance_percent <= 100:
        raise ValueError("El porcentaje de afinidad debe estar entre 0 y 100.")

    embeddings = encode_documents(documents, embedding_model)
    similarity = cosine_similarity(embeddings) * 100
    n_docs = len(documents)
    labels = [-1] * n_docs
    current_label = 0

    for start in range(n_docs):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = current_label
        while stack:
            current = stack.pop()
            neighbors = np.where(similarity[current] >= tolerance_percent)[0]
            for neighbor in neighbors:
                if labels[int(neighbor)] == -1:
                    labels[int(neighbor)] = current_label
                    stack.append(int(neighbor))
        current_label += 1

    method = f"Agrupacion por tolerancia semantica >= {tolerance_percent:.0f}%"
    return method, labels, embeddings


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
        else:
            terms = terms_by_cluster.get(int(cluster_id), [])
        cabas = df.loc[np.array(labels) == cluster_id, "nombre"].tolist()
        summaries.append(
            ClusterSummary(
                cluster_id=int(cluster_id),
                definicion=definition_from_terms(terms),
                terminos=terms,
                cabas=cabas,
            )
        )
    return summaries


def attach_cluster_metadata(
    df: pd.DataFrame,
    summaries: list[ClusterSummary],
    labels: list[int],
    affinity_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    by_id = {summary.cluster_id: summary for summary in summaries}
    output = df.copy()
    output["grupo_id"] = labels
    output["grupo_definicion"] = [by_id[label].definicion for label in labels]
    output["terminos_representativos"] = [
        ", ".join(by_id[label].terminos) for label in labels
    ]
    if affinity_matrix is not None:
        output["afinidad_promedio_grupo"] = average_group_affinities(
            output["nombre"].tolist(), labels, affinity_matrix
        )
    return output


def average_group_affinities(
    names: list[str], labels: list[int], affinity_matrix: pd.DataFrame
) -> list[float | None]:
    averages: list[float | None] = []
    for name, label in zip(names, labels):
        group_names = [candidate for candidate, candidate_label in zip(names, labels) if candidate_label == label]
        peers = [candidate for candidate in group_names if candidate != name]
        if not peers:
            averages.append(None)
            continue
        averages.append(round(float(affinity_matrix.loc[name, peers].mean()), 2))
    return averages


def analyze_cabas(
    df: pd.DataFrame,
    mode: str = "automatic",
    tolerance_percent: float = 60,
    force_fallback: bool = False,
) -> AnalysisResult:
    documents = build_documents(df)
    names = df["nombre"].tolist()
    embeddings: np.ndarray | None = None

    if mode == "threshold":
        method, labels, embeddings = run_threshold_grouping(documents, tolerance_percent)
    elif mode == "automatic":
        if force_fallback:
            method, labels, _ = run_fallback(documents)
        else:
            try:
                method, labels, _, embeddings = run_bertopic(documents)
            except Exception as exc:
                method, labels, _ = run_fallback(documents)
                method = f"{method} (BERTopic no disponible: {exc.__class__.__name__})"
    else:
        raise ValueError("Modo no valido. Use 'automatic' o 'threshold'.")

    if embeddings is None:
        embeddings = encode_documents(documents)

    affinity_matrix = build_affinity_matrix(names, embeddings)
    summaries = build_cluster_summaries(df, documents, labels)
    table = attach_cluster_metadata(df, summaries, labels, affinity_matrix)
    score = silhouette(documents, labels)
    return AnalysisResult(method, labels, summaries, table, score, affinity_matrix)


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
    result_table: pd.DataFrame | None = None,
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
                f"### Grupo {summary.cluster_id}",
                "",
                summary.definicion,
                "",
                f"Terminos representativos: {', '.join(summary.terminos)}",
                "",
                "CABAS incluidas:",
                *format_cabas_for_report(summary, result_table),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_cabas_for_report(
    summary: ClusterSummary, result_table: pd.DataFrame | None
) -> list[str]:
    if result_table is None or "afinidad_promedio_grupo" not in result_table.columns:
        return [f"- {caba}" for caba in summary.cabas]

    rows = []
    for caba in summary.cabas:
        match = result_table.loc[result_table["nombre"] == caba, "afinidad_promedio_grupo"]
        value = match.iloc[0] if not match.empty else None
        if pd.isna(value):
            rows.append(f"- {caba} (afinidad interna: no aplica)")
        else:
            rows.append(f"- {caba} (afinidad interna promedio: {float(value):.2f}%)")
    return rows


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
    parser.add_argument(
        "--mode",
        choices=["automatic", "threshold"],
        default="automatic",
        help="automatic usa BERTopic; threshold agrupa por porcentaje minimo de afinidad.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=60,
        help="Porcentaje minimo de afinidad para --mode threshold.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    analysis = analyze_cabas(
        df,
        mode=args.mode,
        tolerance_percent=args.tolerance,
        force_fallback=args.force_fallback,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    analysis.table.to_csv(args.output, index=False)

    affinity_output = args.output.with_name(f"{args.output.stem}_afinidades.csv")
    analysis.affinity_matrix.to_csv(affinity_output)
    write_markdown_report(
        args.report,
        analysis.method,
        args.input,
        args.output,
        analysis.summaries,
        analysis.score,
        analysis.table,
    )

    payload = {
        "metodo": analysis.method,
        "archivo_salida": str(args.output),
        "archivo_afinidades": str(affinity_output),
        "reporte": str(args.report),
        "grupos": [
            {
                "id": summary.cluster_id,
                "cantidad_cabas": len(summary.cabas),
            }
            for summary in analysis.summaries
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

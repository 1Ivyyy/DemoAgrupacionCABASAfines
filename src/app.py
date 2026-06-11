from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from cabas_afinidad_demo import DEFAULT_INPUT, analyze_cabas


st.set_page_config(
    page_title="Agrupacion de CABAS Afines",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_INPUT)


@st.cache_data(show_spinner=False)
def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def csv_download(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def render_group(summary, result_table: pd.DataFrame, affinity_matrix: pd.DataFrame) -> None:
    with st.container(border=True):
        st.subheader(f"Grupo {summary.cluster_id}")
        st.write(summary.definicion)
        st.caption(f"Terminos representativos: {', '.join(summary.terminos)}")

        group_rows = result_table[result_table["grupo_id"] == summary.cluster_id].copy()
        visible_cols = ["nombre", "descripcion", "afinidad_promedio_grupo"]
        st.dataframe(
            group_rows[visible_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "nombre": "CABA",
                "descripcion": "Descripcion",
                "afinidad_promedio_grupo": st.column_config.NumberColumn(
                    "Afinidad promedio interna",
                    format="%.2f%%",
                ),
            },
        )

        names = group_rows["nombre"].tolist()
        if len(names) > 1:
            group_matrix = affinity_matrix.loc[names, names]
            fig = px.imshow(
                group_matrix,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale="Blues",
                zmin=0,
                zmax=100,
                labels={"x": "CABA", "y": "CABA", "color": "Afinidad"},
            )
            fig.update_traces(
                hovertemplate=(
                    "CABA origen: %{y}<br>"
                    "CABA comparada: %{x}<br>"
                    "Afinidad: %{z:.2f}%<extra></extra>"
                )
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Esta CABA no tiene pares dentro del grupo para calcular afinidad interna.")


def main() -> None:
    st.title("Herramienta de Agrupacion de CABAS Afines")
    st.caption("BERTopic + Sentence Transformers")

    uploaded_file = st.file_uploader("Cargar archivo CSV de CABAS", type=["csv"])
    if uploaded_file is None:
        df = load_default_data()
        st.info("Usando el archivo de ejemplo `data/cabas.csv`.")
    else:
        df = load_uploaded_data(uploaded_file)

    mode_label = st.radio(
        "Modo de agrupacion",
        ["Agrupacion automatica", "Agrupacion definiendo tolerancia"],
        horizontal=True,
    )

    tolerance = 60
    if mode_label == "Agrupacion definiendo tolerancia":
        tolerance = st.slider(
            "Porcentaje minimo de afinidad",
            min_value=1,
            max_value=100,
            value=60,
            step=1,
        )

    mode = "threshold" if mode_label == "Agrupacion definiendo tolerancia" else "automatic"

    if st.button("Agrupar", type="primary"):
        try:
            with st.spinner("Calculando agrupaciones y afinidades..."):
                result = analyze_cabas(df, mode=mode, tolerance_percent=tolerance)
        except Exception as exc:
            st.error(f"No fue posible agrupar las CABAS: {exc}")
            return

        st.success(f"Metodo ejecutado: {result.method}")

        metric_cols = st.columns(3)
        metric_cols[0].metric("CABAS", len(result.table))
        metric_cols[1].metric("Grupos", len(result.summaries))
        metric_cols[2].metric(
            "Silhouette",
            "No aplica" if result.score is None else f"{result.score:.3f}",
        )

        st.download_button(
            "Descargar resultados CSV",
            data=csv_download(result.table),
            file_name="agrupacion_cabas.csv",
            mime="text/csv",
        )

        st.download_button(
            "Descargar matriz de afinidades CSV",
            data=csv_download(result.affinity_matrix.reset_index(names="CABA")),
            file_name="afinidades_cabas.csv",
            mime="text/csv",
        )

        st.divider()
        for summary in result.summaries:
            render_group(summary, result.table, result.affinity_matrix)


if __name__ == "__main__":
    main()

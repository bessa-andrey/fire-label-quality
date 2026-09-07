"""
Dashboard de anotacao manual das 604 amostras FIRMS MATOPIBA.

MODO CEGO. Le `validation_manual_gt_blind.csv` (que NAO contem weak label,
gt_painel, dNBR auto, nem qualquer outro rotulo), exibe apenas os tres thumbnails
Sentinel-2 (pre, pos, dNBR) e metadados FIRMS neutros, e grava as decisoes em
`validation_manual_gt_annotations.csv`.

Uso:
    streamlit run src/visualization/annotation_app.py

Caracteristicas:

- Le e grava em CSVs separados que nao contem qualquer outro rotulo (cegamento
  contra weak labels, gt_painel, manual_label, auto_dNBR).
- Ordem das amostras vem do CSV (pre-embaralhado com seed deterministica).
- Tempo por amostra (segundos entre carregamento e clique) e gravado em
  `tempo_segundos`.
- Banner permanente avisando "MODO CEGO".
"""

import datetime
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
VALIDATION_DIR = PROJECT_DIR / "data" / "processed" / "validation"
BLIND_CSV = VALIDATION_DIR / "validation_manual_gt_blind.csv"
RESULTS_CSV = VALIDATION_DIR / "validation_manual_gt_annotations.csv"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Anotacao manual (MODO CEGO)",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# GEE init
# ---------------------------------------------------------------------------

@st.cache_resource
def init_gee():
    ee.Initialize()
    return True


init_gee()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_blind() -> pd.DataFrame:
    if not BLIND_CSV.exists():
        raise FileNotFoundError(
            f"CSV cego nao encontrado em {BLIND_CSV}. "
            "Gere antes via script setup correspondente."
        )
    df = pd.read_csv(BLIND_CSV)
    df["acq_datetime"] = pd.to_datetime(df["acq_datetime"], format="mixed")

    expected = {"sample_id", "latitude", "longitude", "acq_datetime", "confidence", "frp"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV cego sem colunas obrigatorias: {missing}")

    # Sanidade: nao pode haver coluna de rotulo no CSV cego.
    forbidden = {
        "human_label", "is_reliable", "gt_painel",
        "manual_label", "manual_confidence", "auto_mean_dnbr", "auto_pct_burned",
        "reliability_score", "is_burned",
    }
    leaked = forbidden & set(df.columns)
    if leaked:
        raise ValueError(
            f"CSV cego contem colunas proibidas (vazamento): {leaked}. "
            "Reexecute o script setup que gera o CSV cego."
        )

    return df


def load_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
    else:
        df = pd.DataFrame(columns=[
            "sample_id", "human_label", "human_confidence",
            "human_notes", "human_timestamp", "tempo_segundos",
        ])
    return df


def save_results(df_results: pd.DataFrame) -> None:
    df_results.to_csv(RESULTS_CSV, index=False)


# ---------------------------------------------------------------------------
# GEE thumbnails
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def get_thumbnails(lat: float, lon: float, date_str: str, buffer_m: int = 1000):
    point = ee.Geometry.Point([lon, lat])
    geometry = point.buffer(buffer_m)
    region = geometry.bounds()
    date = ee.Date(date_str)

    pre_fire = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(date.advance(-60, "day"), date.advance(-30, "day"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .median())

    post_fire = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(date, date.advance(30, "day"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .median())

    rgb_params = {
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "dimensions": 400,
        "region": region,
        "format": "png",
    }

    pre_nbr = pre_fire.normalizedDifference(["B8", "B12"])
    post_nbr = post_fire.normalizedDifference(["B8", "B12"])
    dnbr = pre_nbr.subtract(post_nbr)

    dnbr_params = {
        "min": -0.2,
        "max": 0.6,
        "palette": ["blue", "green", "yellow", "orange", "red"],
        "dimensions": 400,
        "region": region,
        "format": "png",
    }

    try:
        pre_url = pre_fire.getThumbURL(rgb_params)
        post_url = post_fire.getThumbURL(rgb_params)
        dnbr_url = dnbr.getThumbURL(dnbr_params)
        return pre_url, post_url, dnbr_url
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Anotacao manual das amostras FIRMS MATOPIBA")
    st.warning(
        "MODO CEGO. Nao compare com outras fontes de rotulo. "
        "Decida apenas com base nos tres thumbnails Sentinel-2 e nos metadados FIRMS "
        "exibidos abaixo. Siga o `PROTOCOLO_ANOTACAO.md` literalmente."
    )

    df_blind = load_blind()
    total = len(df_blind)

    if "df_results" not in st.session_state:
        st.session_state.df_results = load_results()
    df_results = st.session_state.df_results

    # Merge para saber quais amostras ja tem rotulo.
    annotated_ids = set(df_results["sample_id"].dropna().astype(str).tolist())
    df_blind["_annotated"] = df_blind["sample_id"].astype(str).isin(annotated_ids)

    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Progresso")
        n_annotated = int(df_blind["_annotated"].sum())
        st.metric("Anotadas", f"{n_annotated} / {total}")
        st.progress(n_annotated / total if total > 0 else 0)

        st.divider()

        nav_mode = st.radio(
            "Ordem",
            ["CSV (embaralhada na Fase 0)", "Apenas nao anotadas"],
            index=0,
        )

        if nav_mode == "Apenas nao anotadas":
            mask = ~df_blind["_annotated"].values
            available_indices = np.where(mask)[0]
        else:
            available_indices = np.arange(total)

        if len(available_indices) == 0:
            st.success("Todas as amostras estao anotadas. Anotacao concluida.")
            st.stop()

        if "pos" not in st.session_state:
            st.session_state.pos = 0
        pos = min(st.session_state.pos, len(available_indices) - 1)

        st.divider()
        st.write(f"Posicao: {pos + 1} / {len(available_indices)}")

        jump = st.number_input(
            "Ir para posicao",
            min_value=1,
            max_value=len(available_indices),
            value=pos + 1,
        ) - 1
        if jump != pos:
            st.session_state.pos = jump
            st.rerun()

        st.divider()

        if n_annotated > 0:
            labeled = df_results.dropna(subset=["human_label"])
            n_real = (labeled["human_label"] == 0).sum()
            n_spur = (labeled["human_label"] == 1).sum()
            n_unc = (labeled["human_label"] == -1).sum()
            st.write(f"Real: {n_real} | Espurio: {n_spur} | Incerto: {n_unc}")

            if "tempo_segundos" in labeled.columns:
                tempos = pd.to_numeric(labeled["tempo_segundos"], errors="coerce").dropna()
                if len(tempos) > 0:
                    st.write(
                        f"Tempo mediano: {tempos.median():.1f}s "
                        f"(min={tempos.min():.0f}, max={tempos.max():.0f})"
                    )
                    if tempos.median() < 40:
                        st.error("Mediana < 40s. Verifique se nao esta chutando.")
                    elif tempos.median() > 150:
                        st.warning("Mediana > 150s. Pause se estiver distraido.")

        st.divider()
        if st.button("Salvar CSV", type="primary"):
            save_results(df_results)
            st.success(f"Salvo em {RESULTS_CSV.name}")

    # ── Amostra atual ──────────────────────────────────────────────────
    current_idx = int(available_indices[pos])
    row = df_blind.iloc[current_idx]

    lat = float(row["latitude"])
    lon = float(row["longitude"])
    date_str = str(row["acq_datetime"])[:10]
    sample_id = str(row["sample_id"])

    # Time tracking: marca instante de carregamento da amostra atual.
    if st.session_state.get("loaded_sample_id") != sample_id:
        st.session_state.loaded_sample_id = sample_id
        st.session_state.loaded_at = time.time()

    # ── Metadata neutros ───────────────────────────────────────────────
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown(f"**sample_id**: `{sample_id}`")
        st.markdown(f"**Lat/Lon**: {lat:.4f}, {lon:.4f}")
        st.markdown(f"**Data**: {date_str}")
        gmaps_url = f"https://www.google.com/maps/@{lat},{lon},15z"
        st.link_button("Abrir no Google Maps", gmaps_url)
    with col_meta2:
        st.markdown(f"**Confidence FIRMS**: {row['confidence']:.0f}")
        st.markdown(f"**FRP**: {row['frp']:.1f} MW")
        if pd.notna(row.get("brightness")):
            st.markdown(f"**Brightness**: {row['brightness']:.1f} K")

    # ── Imagens Sentinel-2 ────────────────────────────────────────────
    with st.spinner("Carregando imagens Sentinel-2..."):
        pre_url, post_url, dnbr_url = get_thumbnails(lat, lon, date_str)

    if pre_url:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Pre-fogo (RGB)**")
            st.image(pre_url, use_container_width=True)
        with col2:
            st.markdown("**Pos-fogo (RGB)**")
            st.image(post_url, use_container_width=True)
        with col3:
            st.markdown("**dNBR (azul=sem queima, vermelho=queimada)**")
            st.image(dnbr_url, use_container_width=True)
    else:
        st.warning("Sem imagens Sentinel-2 disponiveis. Considere marcar como Incerto.")

    st.divider()

    # ── Status da amostra atual ────────────────────────────────────────
    existing_label = df_results[df_results["sample_id"] == sample_id]
    has_label = len(existing_label) > 0
    if has_label:
        lbl_int = existing_label.iloc[0]["human_label"]
        lbl_str = {0: "Real", 1: "Espurio", -1: "Incerto"}.get(int(lbl_int), "?") if pd.notna(lbl_int) else "?"
        conf_val = existing_label.iloc[0].get("human_confidence", "?")
        st.info(f"Ja anotada: **{lbl_str}** (conf={conf_val}). Reanote para alterar.")

    # ── Controles de anotacao ──────────────────────────────────────────
    sample_key = f"s_{sample_id}"

    col_conf, col_notes = st.columns([1, 3])
    with col_conf:
        conf_options = [1, 2, 3, 4, 5]
        conf_labels = ["1 - Muito baixa", "2 - Baixa", "3 - Media", "4 - Alta", "5 - Muito alta"]
        if has_label and pd.notna(existing_label.iloc[0].get("human_confidence")):
            default_idx = int(existing_label.iloc[0]["human_confidence"]) - 1
        else:
            default_idx = 2
        conf_val = st.radio(
            "Confianca",
            conf_options,
            index=default_idx,
            format_func=lambda x: conf_labels[x - 1],
            key=f"ann_confidence_{sample_key}",
        )
        st.session_state["ann_confidence_val"] = conf_val

    with col_notes:
        notes_options = [
            "(sem nota)",
            "Falso positivo do sensor",
            "Queimada agricola controlada (padrao geometrico)",
            "Queimada prescrita / manejo do fogo",
            "Fogo irregular em area agricola",
            "Perimetro urbano",
            "Parque industrial",
            "Reflexo em corpo d'agua",
            "Fogo proximo a zona urbana",
            "Imagem comprometida por nuvens",
            "Sem imagens de satelite disponiveis",
        ]
        saved_note = ""
        if has_label and pd.notna(existing_label.iloc[0].get("human_notes")):
            saved_note = str(existing_label.iloc[0]["human_notes"]).strip()
        default_note_idx = notes_options.index(saved_note) if saved_note in notes_options else 0
        selected_note = st.selectbox(
            "Nota padrao",
            notes_options,
            index=default_note_idx,
            key=f"ann_notes_radio_{sample_key}",
        )
        custom_note = st.text_input(
            "Nota manual (opcional)",
            value="",
            key=f"ann_notes_custom_{sample_key}",
        )
        if custom_note.strip():
            st.session_state["ann_notes_val"] = custom_note.strip()
        elif selected_note == "(sem nota)":
            st.session_state["ann_notes_val"] = ""
        else:
            st.session_state["ann_notes_val"] = selected_note

    def annotate(label_value: int) -> None:
        elapsed = time.time() - st.session_state.get("loaded_at", time.time())
        new_row = {
            "sample_id": sample_id,
            "human_label": label_value,
            "human_confidence": st.session_state.get("ann_confidence_val", 3),
            "human_notes": st.session_state.get("ann_notes_val", ""),
            "human_timestamp": datetime.datetime.now().isoformat(),
            "tempo_segundos": round(elapsed, 2),
        }
        df_r = st.session_state.df_results
        df_r = df_r[df_r["sample_id"] != sample_id]
        df_r = pd.concat([df_r, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.df_results = df_r
        save_results(df_r)
        st.session_state.pos = min(pos + 1, len(available_indices) - 1)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("FOGO REAL (0)", type="primary", use_container_width=True):
            annotate(0)
            st.rerun()
    with col_b:
        if st.button("ESPURIO (1)", type="secondary", use_container_width=True):
            annotate(1)
            st.rerun()
    with col_c:
        if st.button("INCERTO (-1)", use_container_width=True):
            annotate(-1)
            st.rerun()

    st.divider()
    nav1, _, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("< Anterior") and pos > 0:
            st.session_state.pos = pos - 1
            st.rerun()
    with nav3:
        if st.button("Proximo >") and pos < len(available_indices) - 1:
            st.session_state.pos = pos + 1
            st.rerun()


if __name__ == "__main__":
    main()

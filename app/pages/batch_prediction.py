"""
Batch Prediction Page — Liver Cirrhosis Stage Prediction Dashboard.
Allows CSV upload, validation, batch prediction, and result download.
"""

import streamlit as st
import pandas as pd
import io
from app.components.ui_components import (
    render_section_header,
    render_info_banner,
    render_empty_state,
    render_metric_card,
)
from services.prediction_service import PredictionService
from services.preprocessing_service import (
    PreprocessingService, FEATURE_ORDER, CATEGORICAL_MAPS,
)


_CSV_TEMPLATE_COMMENT = (
    "N_Days,Status,Drug,Age,Sex,Ascites,Hepatomegaly,Spiders,Edema,"
    "Bilirubin,Cholesterol,Albumin,Copper,Alk_Phos,SGOT,Tryglicerides,Platelets,Prothrombin\n"
    "1500,C,Placebo,50,F,N,N,N,N,2.0,300.0,3.5,90.0,1000.0,120.0,120.0,250.0,10.5\n"
    "2200,D,D-penicillamine,45,M,Y,Y,Y,S,5.2,400.0,2.8,150.0,1500.0,180.0,95.0,200.0,11.2\n"
)


def _build_csv_template() -> bytes:
    """Build a downloadable CSV template."""
    return _CSV_TEMPLATE_COMMENT.encode("utf-8")


def _colorize_stage(val: str) -> str:
    """Return CSS color string for a stage value."""
    colors = {"Stage 1": "#22c55e", "Stage 2": "#f59e0b", "Stage 3": "#ef4444"}
    color = colors.get(val, "#94a3b8")
    return f"color: {color}; font-weight: 700;"


def render_batch_prediction_page(
    prediction_service: PredictionService,
    preprocessing_service: PreprocessingService,
):
    """Render the full Batch Prediction page."""

    render_section_header(
        "Batch Patient Prediction",
        "Upload a CSV file to predict cirrhosis stages for multiple patients",
        icon="📋",
    )

    # ── Instructions & Template ───────────────────────────────────────────
    with st.expander("📘 How to use Batch Prediction — Column Guide", expanded=False):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            **Required Columns (18 total):**
            | Column | Type | Example |
            |--------|------|---------|
            | N_Days | int | 1500 |
            | Status | str | C / D / CL |
            | Drug | str | Placebo / D-penicillamine |
            | Age | int | 50 *(years — auto-converted)* |
            | Sex | str | F / M |
            | Ascites | str | N / Y |
            | Hepatomegaly | str | N / Y |
            """)

        with col_b:
            st.markdown("""
            **Required Columns (continued):**
            | Column | Type | Example |
            |--------|------|---------|
            | Spiders | str | N / Y |
            | Edema | str | N / S / Y |
            | Bilirubin | float | 2.0 |
            | Cholesterol | float | 300.0 |
            | Albumin | float | 3.5 |
            | Copper | float | 90.0 |
            | Alk_Phos | float | 1000.0 |
            | SGOT | float | 120.0 |
            | Tryglicerides | float | 120.0 |
            | Platelets | float | 250.0 |
            | Prothrombin | float | 10.5 |
            """)

        st.download_button(
            label="⬇️ Download CSV Template",
            data=_build_csv_template(),
            file_name="liver_cirrhosis_batch_template.csv",
            mime="text/csv",
        )

    render_info_banner(
        "If <strong>Age</strong> column contains values <200, it will be treated as years and "
        "automatically converted to days (Age × 365.25) before prediction.",
        color="#f59e0b",
        icon="⚠️",
    )

    # ── File Uploader ─────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload Patient Data CSV",
        type=["csv"],
        help="CSV file with the 18 required feature columns. Max 10,000 rows recommended.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        render_empty_state(
            "No File Uploaded",
            "Upload a CSV file with patient data to run batch predictions. "
            "Download the template above to get started.",
            icon="📤",
        )
        return

    # ── Read & Validate CSV ───────────────────────────────────────────────
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {str(e)}")
        return

    render_section_header("Uploaded Data Preview", icon="👁️")

    # Column validation
    is_valid, missing_cols, extra_cols = preprocessing_service.validate_batch_columns(raw_df)

    # Summary stats
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown(render_metric_card("Rows", f"{len(raw_df):,}", "Patients uploaded",
                                       "#3b82f6", "👥"), unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(render_metric_card("Columns", str(len(raw_df.columns)), "Found in CSV",
                                       "#06b6d4", "📊"), unsafe_allow_html=True)
    with stat_cols[2]:
        missing_str = "✅ 0" if not missing_cols else f"❌ {len(missing_cols)}"
        color = "#22c55e" if not missing_cols else "#ef4444"
        st.markdown(render_metric_card("Missing Cols", missing_str, "Required columns",
                                       color, "🔍"), unsafe_allow_html=True)
    with stat_cols[3]:
        extra_str = str(len(extra_cols)) if extra_cols else "0"
        st.markdown(render_metric_card("Extra Cols", extra_str, "Will be ignored",
                                       "#f59e0b", "➕"), unsafe_allow_html=True)

    if missing_cols:
        st.error(f"❌ **Missing required columns:** `{'`, `'.join(missing_cols)}`")
        st.info("Please add the missing columns and re-upload your file.")
        return

    if extra_cols:
        render_info_banner(
            f"Extra columns found and will be ignored: {', '.join(extra_cols)}",
            color="#f59e0b", icon="ℹ️"
        )

    # Preview
    with st.expander("📄 Preview Uploaded Data (first 10 rows)", expanded=True):
        st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

    # ── Run Batch Predictions ─────────────────────────────────────────────
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(148,163,184,0.12); margin: 1rem 0;'>",
                unsafe_allow_html=True)

    run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
    with run_col2:
        run_btn = st.button(
            f"⚡ Run Batch Prediction ({len(raw_df):,} patients)",
            use_container_width=True,
            key="run_batch_btn",
        )

    if run_btn:
        progress_bar = st.progress(0, text="Preprocessing data...")
        try:
            processed_df, warnings = preprocessing_service.preprocess_batch(raw_df.copy())
            progress_bar.progress(40, text="Running model inference...")

            predictions = prediction_service.predict_batch(processed_df)
            progress_bar.progress(80, text="Assembling results...")

            # Combine original data with predictions
            results_df = pd.concat([raw_df.reset_index(drop=True), predictions], axis=1)
            st.session_state["batch_results"] = results_df
            st.session_state["batch_predictions"] = predictions
            st.session_state["batch_warnings"] = warnings

            progress_bar.progress(100, text="Done! ✅")

            for w in warnings:
                render_info_banner(w, color="#f59e0b", icon="ℹ️")

        except ValueError as ve:
            st.error(f"❌ Validation error: {str(ve)}")
            progress_bar.empty()
            return
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            progress_bar.empty()
            return

    # ── Display Batch Results ─────────────────────────────────────────────
    if "batch_results" not in st.session_state:
        return

    results_df: pd.DataFrame = st.session_state["batch_results"]
    predictions: pd.DataFrame = st.session_state["batch_predictions"]

    render_section_header("Batch Prediction Results", icon="📊")

    # ── Summary Metrics ───────────────────────────────────────────────────
    stage_counts = predictions["Predicted_Stage"].value_counts()
    avg_confidence = predictions["Prediction_Probability"].mean() * 100

    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.markdown(render_metric_card("Total Predicted", f"{len(results_df):,}",
                                       "Patients", "#3b82f6", "👥"), unsafe_allow_html=True)
    with summary_cols[1]:
        s1 = stage_counts.get("Stage 1", 0)
        st.markdown(render_metric_card("Stage 1", str(s1),
                                       f"{s1/len(results_df)*100:.1f}% of patients",
                                       "#22c55e", "🟢"), unsafe_allow_html=True)
    with summary_cols[2]:
        s2 = stage_counts.get("Stage 2", 0)
        st.markdown(render_metric_card("Stage 2", str(s2),
                                       f"{s2/len(results_df)*100:.1f}% of patients",
                                       "#f59e0b", "🟡"), unsafe_allow_html=True)
    with summary_cols[3]:
        s3 = stage_counts.get("Stage 3", 0)
        st.markdown(render_metric_card("Stage 3", str(s3),
                                       f"{s3/len(results_df)*100:.1f}% of patients",
                                       "#ef4444", "🔴"), unsafe_allow_html=True)

    # Average confidence
    st.markdown(
        render_metric_card("Avg Confidence", f"{avg_confidence:.1f}%",
                           "Mean prediction confidence", "#a855f7", "🎯"),
        unsafe_allow_html=True,
    )

    # ── Stage Distribution Chart ──────────────────────────────────────────
    import plotly.graph_objects as go

    stage_dist_fig = go.Figure(go.Pie(
        labels=list(stage_counts.index),
        values=list(stage_counts.values),
        hole=0.45,
        marker_colors=["#22c55e", "#f59e0b", "#ef4444"],
        textfont={"color": "white", "size": 13},
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    stage_dist_fig.update_layout(
        title="Stage Distribution in Batch",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0", "family": "Inter"},
        legend={"bgcolor": "rgba(30,41,59,0.8)"},
        height=350,
    )

    chart_col, _ = st.columns([1.2, 0.8])
    with chart_col:
        st.plotly_chart(stage_dist_fig, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Results Table ─────────────────────────────────────────────────────
    render_section_header("Full Results Table", icon="📋")

    # Display prediction columns highlighted
    display_cols = [
        "Predicted_Stage", "Prediction_Probability",
        "Stage_1_Probability", "Stage_2_Probability", "Stage_3_Probability"
    ] + [c for c in raw_df.columns if c not in predictions.columns]

    available_display = [c for c in display_cols if c in results_df.columns]
    st.dataframe(
        results_df[available_display].style.applymap(
            _colorize_stage, subset=["Predicted_Stage"]
        ),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # ── Download Button ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
    with dl_col2:
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_bytes,
            file_name="liver_cirrhosis_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    render_info_banner(
        "Downloaded CSV includes original patient data plus: "
        "<strong>Predicted_Stage</strong>, <strong>Prediction_Probability</strong>, "
        "<strong>Stage_1_Probability</strong>, <strong>Stage_2_Probability</strong>, "
        "<strong>Stage_3_Probability</strong>.",
        color="#22c55e", icon="✅"
    )

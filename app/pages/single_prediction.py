"""
Single Prediction Page — Liver Cirrhosis Stage Prediction Dashboard.
Clean rewrite: NO raw HTML inside st.form blocks. All HTML self-contained.
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.components.ui_components import (
    render_section_header,
    render_prediction_result_card,
    render_info_banner,
    render_shap_explanation_header,
    render_feature_badge,
    render_empty_state,
    render_metric_card,
)
from services.prediction_service import PredictionService
from services.preprocessing_service import PreprocessingService
from services.analytics_service import AnalyticsService


def render_single_prediction_page(
    prediction_service: PredictionService,
    preprocessing_service: PreprocessingService,
    analytics_service: AnalyticsService,
):
    """Render the full Single Prediction page."""

    render_section_header(
        "Single Patient Prediction",
        "Enter patient clinical data to predict liver cirrhosis stage",
        icon="🔬",
    )

    render_info_banner(
        "Age is collected in <strong>Years</strong> and automatically converted to Days "
        "(Age × 365.25) before model prediction. No scaling is applied.",
        color="#3b82f6",
        icon="💡",
    )

    # ── Section label (self-contained HTML — NOT inside the form) ──────────
    st.html("""
<div style="
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(148,163,184,0.12);
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    margin: 0.5rem 0 0.5rem 0;
    display: flex; align-items: center; gap: 0.6rem;
">
    <span style="font-size:1.1rem;">👤</span>
    <span style="color:#f1f5f9; font-weight:700; font-size:0.95rem;">Patient Numerical Features</span>
</div>
""")

    # ── Patient Data Input Form ────────────────────────────────────────────
    with st.form(key="single_prediction_form", clear_on_submit=False):

        col1, col2, col3 = st.columns(3)

        with col1:
            n_days = st.number_input(
                "N_Days", min_value=0, max_value=10000, value=1500,
                step=1,
                help="Number of days between registration and earlier of death, liver transplant, or study analysis"
            )
            age_years = st.number_input(
                "Age (Years) ⭐",
                min_value=1, max_value=120, value=50, step=1,
                help="Age will automatically be converted into days for model prediction. (Age_Days = Age_Years × 365.25)"
            )
            bilirubin = st.number_input(
                "Bilirubin (mg/dL)", min_value=0.0, max_value=30.0, value=2.0,
                step=0.1, format="%.1f", help="Serum bilirubin in mg/dL"
            )
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)", min_value=0.0, max_value=2000.0, value=300.0,
                step=1.0, format="%.1f", help="Serum cholesterol in mg/dL"
            )

        with col2:
            albumin = st.number_input(
                "Albumin (g/dL)", min_value=0.0, max_value=10.0, value=3.5,
                step=0.01, format="%.2f", help="Serum albumin in g/dL"
            )
            copper = st.number_input(
                "Copper (μg/day)", min_value=0.0, max_value=600.0, value=90.0,
                step=0.1, format="%.1f", help="Urine copper in μg/day"
            )
            alk_phos = st.number_input(
                "Alk_Phos (U/liter)", min_value=0.0, max_value=15000.0, value=1000.0,
                step=1.0, format="%.1f", help="Alkaline phosphotase in U/liter"
            )
            sgot = st.number_input(
                "SGOT (U/mL)", min_value=0.0, max_value=500.0, value=120.0,
                step=0.1, format="%.2f", help="SGOT (AST) in U/mL"
            )

        with col3:
            tryglicerides = st.number_input(
                "Tryglicerides (mg/dL)", min_value=0.0, max_value=1000.0, value=120.0,
                step=0.1, format="%.1f", help="Triglycerides in mg/dL"
            )
            platelets = st.number_input(
                "Platelets (ml/1000)", min_value=0.0, max_value=1000.0, value=250.0,
                step=0.1, format="%.1f", help="Platelets per ml/1000"
            )
            prothrombin = st.number_input(
                "Prothrombin (s)", min_value=0.0, max_value=20.0, value=10.5,
                step=0.1, format="%.1f", help="Prothrombin time in seconds"
            )

        st.divider()

        # ── Categorical Features ──────────────────────────────────────────
        st.markdown("**🏷️ Patient Categorical Features**")

        ccat1, ccat2, ccat3, ccat4 = st.columns(4)

        with ccat1:
            status = st.selectbox(
                "Status",
                options=["C", "D", "CL"],
                index=0,
                help="C=Censored, D=Death, CL=Censored due to liver transplant"
            )
            drug = st.selectbox(
                "Drug",
                options=["Placebo", "D-penicillamine"],
                index=0,
                help="Drug treatment received by the patient"
            )

        with ccat2:
            sex = st.selectbox(
                "Sex", options=["F", "M"], index=0,
                help="Patient biological sex"
            )
            ascites = st.selectbox(
                "Ascites", options=["N", "Y"], index=0,
                help="Presence of ascites (N=No, Y=Yes)"
            )

        with ccat3:
            hepatomegaly = st.selectbox(
                "Hepatomegaly", options=["N", "Y"], index=0,
                help="Presence of hepatomegaly (N=No, Y=Yes)"
            )
            spiders = st.selectbox(
                "Spiders", options=["N", "Y"], index=0,
                help="Presence of spider angiomata (N=No, Y=Yes)"
            )

        with ccat4:
            edema = st.selectbox(
                "Edema", options=["N", "S", "Y"], index=0,
                help="Edema: N=No edema, S=Edema without diuretics/resolved, Y=Edema despite diuretics"
            )

        st.divider()

        # ── Submit Button ─────────────────────────────────────────────────
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            submitted = st.form_submit_button(
                "🔬 Predict Cirrhosis Stage",
                use_container_width=True,
            )

    # ── Prediction Logic ──────────────────────────────────────────────────
    if submitted:
        try:
            input_dict = {
                "N_Days":        n_days,
                "Age_Years":     age_years,
                "Bilirubin":     bilirubin,
                "Cholesterol":   cholesterol,
                "Albumin":       albumin,
                "Copper":        copper,
                "Alk_Phos":      alk_phos,
                "SGOT":          sgot,
                "Tryglicerides": tryglicerides,
                "Platelets":     platelets,
                "Prothrombin":   prothrombin,
                "Status":        status,
                "Drug":          drug,
                "Sex":           sex,
                "Ascites":       ascites,
                "Hepatomegaly":  hepatomegaly,
                "Spiders":       spiders,
                "Edema":         edema,
            }

            with st.spinner("Running prediction..."):
                processed_df = preprocessing_service.preprocess_single(input_dict)
                result = prediction_service.predict_single(processed_df)

            st.session_state["last_prediction"] = result
            st.session_state["last_processed_df"] = processed_df
            st.session_state["last_input"] = input_dict

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            return

    # ── Display Results ───────────────────────────────────────────────────
    if "last_prediction" not in st.session_state:
        render_empty_state(
            "No Prediction Yet",
            "Fill in the patient data above and click 'Predict Cirrhosis Stage' to see results.",
            icon="🔬",
        )
        return

    result = st.session_state["last_prediction"]
    processed_df = st.session_state["last_processed_df"]

    st.divider()
    render_section_header("Prediction Results", icon="🔬")

    # ── Main result + gauge ───────────────────────────────────────────────
    res_col1, res_col2 = st.columns([1.2, 0.8], gap="large")

    with res_col1:
        render_prediction_result_card(
            stage=result["predicted_stage"],
            confidence=result["confidence"],
            description=result["stage_description"],
            color=result["stage_color"],
        )

    with res_col2:
        gauge_fig = analytics_service.build_probability_gauge(
            confidence=result["confidence"],
            stage=result["predicted_stage"],
            color=result["stage_color"],
        )
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Probability Distribution ──────────────────────────────────────────
    render_section_header("Probability Distribution", icon="📊")
    bar_fig = analytics_service.build_probability_bar_chart(result["probabilities"])
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Probability Details ───────────────────────────────────────────────
    prob_cols = st.columns(3)
    stage_colors = {"Stage 1": "#22c55e", "Stage 2": "#f59e0b", "Stage 3": "#ef4444"}
    stage_icons  = {"Stage 1": "🟢", "Stage 2": "🟡", "Stage 3": "🔴"}
    for i, (stage, prob) in enumerate(result["probabilities"].items()):
        with prob_cols[i]:
            st.html(
                render_metric_card(
                    title=stage,
                    value=f"{prob*100:.1f}%",
                    subtitle="Prediction probability",
                    color=stage_colors[stage],
                    icon=stage_icons[stage],
                )
            )

    # ── Age Conversion Note ───────────────────────────────────────────────
    if "last_input" in st.session_state:
        inp = st.session_state["last_input"]
        age_days = int(inp.get("Age_Years", 0) * 365.25)
        render_info_banner(
            f"Age Conversion: <strong>{inp.get('Age_Years')} years</strong> → "
            f"<strong>{age_days:,} days</strong> (used in model prediction)",
            color="#06b6d4",
            icon="🔄",
        )

    # ── SHAP Explainability ───────────────────────────────────────────────
    st.divider()
    render_shap_explanation_header()

    with st.spinner("Computing SHAP explanations..."):
        shap_data = analytics_service.compute_shap_values(processed_df)

    if shap_data is None or "error" in shap_data:
        err = shap_data["error"] if shap_data and "error" in shap_data else "Unknown"
        st.warning(f"⚠️ SHAP computation unavailable: {err}")
    else:
        predicted_class = result["predicted_class"]

        # Waterfall chart
        waterfall_fig = analytics_service.build_shap_waterfall_plotly(
            shap_values=shap_data["shap_values"],
            display_names=shap_data["display_names"],
            predicted_class=predicted_class,
        )
        st.plotly_chart(waterfall_fig, use_container_width=True,
                        config={"displayModeBar": False})

        # SHAP contribution table
        contrib_df = analytics_service.get_shap_contribution_table(
            shap_values=shap_data["shap_values"],
            display_names=shap_data["display_names"],
            feature_values=shap_data["feature_values"],
            predicted_class=predicted_class,
        )

        if "Error" not in contrib_df.columns:
            shap_col1, shap_col2 = st.columns(2, gap="large")

            with shap_col1:
                render_section_header("Top Positive Drivers", "Features increasing risk", icon="⬆️")
                pos_df = contrib_df[contrib_df["SHAP_Value"] > 0].head(5)
                if not pos_df.empty:
                    # Build complete self-contained HTML string
                    badges_html = '<div style="display:grid;grid-template-columns:1fr;gap:0.4rem;">'
                    for _, row in pos_df.iterrows():
                        badges_html += render_feature_badge(row["Feature"], row["Value"], row["SHAP_Value"])
                    badges_html += "</div>"
                    st.html(badges_html)
                else:
                    st.info("No positive SHAP contributors found.")

            with shap_col2:
                render_section_header("Top Negative Drivers", "Features decreasing risk", icon="⬇️")
                neg_df = contrib_df[contrib_df["SHAP_Value"] < 0].head(5)
                if not neg_df.empty:
                    badges_html = '<div style="display:grid;grid-template-columns:1fr;gap:0.4rem;">'
                    for _, row in neg_df.iterrows():
                        badges_html += render_feature_badge(row["Feature"], row["Value"], row["SHAP_Value"])
                    badges_html += "</div>"
                    st.html(badges_html)
                else:
                    st.info("No negative SHAP contributors found.")

            # Full table
            with st.expander("📋 Full SHAP Feature Contribution Table", expanded=False):
                display_df = contrib_df[["Rank", "Feature", "Value", "SHAP_Value", "Direction"]].copy()
                display_df["SHAP_Value"] = display_df["SHAP_Value"].round(4)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "Feature": st.column_config.TextColumn("Feature"),
                        "Value": st.column_config.TextColumn("Value"),
                        "SHAP_Value": st.column_config.NumberColumn("SHAP Value", format="%.4f"),
                        "Direction": st.column_config.TextColumn("Direction"),
                    }
                )

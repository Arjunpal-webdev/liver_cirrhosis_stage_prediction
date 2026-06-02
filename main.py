"""
Liver Cirrhosis Stage Prediction & Analytics Dashboard
======================================================
Main entry point for the Streamlit application.

Run with:
    streamlit run main.py
"""

import sys
import os
from pathlib import Path

# ── Ensure project root is on Python path ─────────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# ── Streamlit Page Config (MUST be first Streamlit call) ──────────────────
st.set_page_config(
    page_title="Liver Cirrhosis Stage Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "# Liver Cirrhosis Stage Prediction Dashboard\n"
            "AI-powered clinical decision support built with XGBoost.\n\n"
            "**Accuracy:** 91% | **ROC AUC:** 0.9798\n\n"
            "_For research use only. Not a substitute for medical advice._"
        ),
    },
)

# ── Load CSS Styles ───────────────────────────────────────────────────────
from utils.helpers import load_css, check_model_files

load_css(ROOT_DIR / "app" / "styles" / "main.css")

# ── Import Services ───────────────────────────────────────────────────────
from services.preprocessing_service import PreprocessingService
from services.prediction_service import PredictionService
from services.analytics_service import AnalyticsService

# ── Import Pages ──────────────────────────────────────────────────────────
from app.pages.single_prediction import render_single_prediction_page
from app.pages.batch_prediction import render_batch_prediction_page
from app.pages.model_analytics import render_model_analytics_page

# ── Import Components ─────────────────────────────────────────────────────
from app.components.ui_components import (
    render_dashboard_header,
    render_sidebar_nav,
    render_info_banner,
)


@st.cache_resource(show_spinner=False)
def initialize_services() -> tuple[PredictionService, PreprocessingService, AnalyticsService]:
    """Initialize and cache all services."""
    preprocessing_service = PreprocessingService()
    prediction_service = PredictionService()
    analytics_service = AnalyticsService(
        model=prediction_service.model,
        preprocessing_service=preprocessing_service,
    )
    return prediction_service, preprocessing_service, analytics_service


def check_dependencies() -> bool:
    """Check that required model files exist before starting."""
    file_status = check_model_files(ROOT_DIR)
    missing = [k for k, v in file_status.items() if not v["exists"] and k in ("model", "encoder")]
    if missing:
        st.error(
            f"❌ **Missing required files:** `{'`, `'.join(missing)}`\n\n"
            f"Please ensure the following files are present:\n"
            f"- `models/best_xgb_model.pkl`\n"
            f"- `models/label_encoder.pkl`"
        )
        return False
    return True


def main():
    """Main application function."""

    # ── Render Sidebar Navigation ─────────────────────────────────────────
    selected_page = render_sidebar_nav()

    # ── Check Dependencies ────────────────────────────────────────────────
    if not check_dependencies():
        st.stop()

    # ── Initialize Services ───────────────────────────────────────────────
    try:
        with st.spinner("Initializing AI model..."):
            pred_svc, prep_svc, analytics_svc = initialize_services()
    except FileNotFoundError as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info(
            "Please place `best_xgb_model.pkl` and `label_encoder.pkl` "
            "in the `models/` directory and restart the application."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Service initialization error: {str(e)}")
        st.stop()

    # ── Render Dashboard Header ───────────────────────────────────────────
    render_dashboard_header()

    # ── Route to Selected Page ────────────────────────────────────────────
    if "Single Prediction" in selected_page:
        render_single_prediction_page(pred_svc, prep_svc, analytics_svc)

    elif "Batch Prediction" in selected_page:
        render_batch_prediction_page(pred_svc, prep_svc)

    elif "Model Analytics" in selected_page:
        render_model_analytics_page(pred_svc, prep_svc, analytics_svc)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid rgba(148,163,184,0.1);
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
    ">
        🫀 Liver Cirrhosis Stage Prediction Dashboard &nbsp;|&nbsp;
        Built with Streamlit &amp; XGBoost &nbsp;|&nbsp;
        <span style="color: #64748b;">For research use only — not medical advice</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

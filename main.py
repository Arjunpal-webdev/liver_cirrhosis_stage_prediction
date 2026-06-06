"""
Liver Cirrhosis Stage Prediction & Analytics Dashboard
======================================================
Main entry point for the Streamlit application.

Run with:
    streamlit run main.py
"""

import sys
from pathlib import Path

# ── Ensure project root is on Python path ─────────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# ── Streamlit Page Config (MUST be first Streamlit call) ──────────────────
st.set_page_config(
    page_title="Liver Cirrhosis Stage Prediction",
    page_icon="🩺",
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

# ── Load CSS ──────────────────────────────────────────────────────────────
from utils.helpers import load_css, check_model_files

load_css(ROOT_DIR / "app" / "styles" / "main.css")

# ── Services ──────────────────────────────────────────────────────────────
from services.preprocessing_service import PreprocessingService
from services.prediction_service import PredictionService
from services.analytics_service import AnalyticsService

# ── Pages ─────────────────────────────────────────────────────────────────
from app.pages.single_prediction import render_single_prediction_page
from app.pages.batch_prediction import render_batch_prediction_page
from app.pages.model_analytics import render_model_analytics_page

# ── Components ────────────────────────────────────────────────────────────
from app.components.ui_components import (
    render_dashboard_header,
    render_sidebar_nav,
)


# ── Cached Service Initializer ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_services():
    """Initialize and cache all backend services (runs once per session)."""
    prep_svc = PreprocessingService()
    pred_svc = PredictionService()
    ana_svc  = AnalyticsService(model=pred_svc.model,
                                preprocessing_service=prep_svc)
    return pred_svc, prep_svc, ana_svc


# ── Dependency Check ─────────────────────────────────────────────────────
def check_dependencies() -> bool:
    file_status = check_model_files(ROOT_DIR)
    missing = [k for k, v in file_status.items()
               if not v["exists"] and k in ("model", "encoder")]
    if missing:
        st.error(
            f"Missing required model files: {', '.join(missing)}\n\n"
            "Please place `best_xgb_model.pkl` and `label_encoder.pkl` "
            "inside the `models/` directory."
        )
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRY — runs on every Streamlit page load
# ══════════════════════════════════════════════════════════════════════════

# 1. Sidebar (renders first so nav is always visible)
selected_page = render_sidebar_nav()

# 2. Dependency gate
if not check_dependencies():
    st.stop()

# 3. Load services
try:
    with st.spinner("Initializing AI model..."):
        pred_svc, prep_svc, analytics_svc = initialize_services()
except FileNotFoundError as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Service initialization error: {exc}")
    st.stop()

# 4. Dashboard header
render_dashboard_header()

# 5. Route to selected page
if "Single Prediction" in selected_page:
    render_single_prediction_page(pred_svc, prep_svc, analytics_svc)

elif "Batch Prediction" in selected_page:
    render_batch_prediction_page(pred_svc, prep_svc)

elif "Model Analytics" in selected_page:
    render_model_analytics_page(pred_svc, prep_svc, analytics_svc)

# 6. Footer
st.html("""
<div style="
    margin-top:3rem; padding:1rem 0;
    border-top:1px solid rgba(148,163,184,0.1);
    text-align:center; color:#475569; font-size:0.78rem;
">
    🩺 Liver Cirrhosis Stage Prediction Dashboard &nbsp;|&nbsp;
    Built with Streamlit &amp; XGBoost &nbsp;|&nbsp;
    <span style="color:#64748b;">For research use only — not medical advice</span>
</div>
""")

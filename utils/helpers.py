"""
Utility functions for the Liver Cirrhosis Prediction Dashboard.
"""

import streamlit as st
import pandas as pd
from pathlib import Path


def load_css(css_path: Path | str) -> None:
    """Load and inject a CSS file into the Streamlit app."""
    css_path = Path(css_path)
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_path}")


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with appropriate decimal places."""
    if isinstance(value, int) or value == int(value):
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def validate_numeric_range(value: float, min_val: float, max_val: float,
                            field_name: str) -> list[str]:
    """Validate that a numeric value is within expected range."""
    errors = []
    if value < min_val or value > max_val:
        errors.append(
            f"'{field_name}' value {value} is outside expected range [{min_val}, {max_val}]"
        )
    return errors


def check_model_files(base_path: Path) -> dict:
    """Check if required model files exist."""
    files = {
        "model": base_path / "models" / "best_xgb_model.pkl",
        "encoder": base_path / "models" / "label_encoder.pkl",
        "data": base_path / "data" / "liver_cirrhosis.csv",
    }
    status = {}
    for key, path in files.items():
        status[key] = {"path": str(path), "exists": path.exists(),
                       "size": path.stat().st_size if path.exists() else 0}
    return status


def get_stage_color(stage: str) -> str:
    """Return hex color for a stage string."""
    colors = {"Stage 1": "#22c55e", "Stage 2": "#f59e0b", "Stage 3": "#ef4444"}
    return colors.get(stage, "#94a3b8")


def stage_int_to_str(stage_int: int) -> str:
    """Convert integer stage (1/2/3) to display string."""
    return f"Stage {stage_int}"


def confidence_to_level(confidence: float) -> tuple[str, str]:
    """Convert confidence percentage to level label and color."""
    if confidence >= 80:
        return "High", "#22c55e"
    elif confidence >= 60:
        return "Medium", "#f59e0b"
    else:
        return "Low", "#ef4444"

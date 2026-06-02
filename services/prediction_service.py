"""
Prediction Service for Liver Cirrhosis Stage Prediction.
Loads the XGBoost model and provides prediction methods.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Stage mapping: model outputs 0/1/2 → display Stage 1/2/3
STAGE_MAP = {0: "Stage 1", 1: "Stage 2", 2: "Stage 3"}
STAGE_COLORS = {
    "Stage 1": "#22c55e",   # green
    "Stage 2": "#f59e0b",   # amber
    "Stage 3": "#ef4444",   # red
}
STAGE_DESCRIPTIONS = {
    "Stage 1": "Early cirrhosis — compensated liver disease with minimal symptoms.",
    "Stage 2": "Moderate cirrhosis — developing portal hypertension signs.",
    "Stage 3": "Advanced cirrhosis — significant liver dysfunction present.",
}

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_xgb_model.pkl"


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    """Load and cache the trained XGBoost model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please ensure 'best_xgb_model.pkl' is placed in the 'models/' directory."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


class PredictionService:
    """Handles model inference and result formatting."""

    def __init__(self):
        self.model = load_model()
        self.stage_map = STAGE_MAP
        self.stage_colors = STAGE_COLORS
        self.stage_descriptions = STAGE_DESCRIPTIONS
        self.classes = [0, 1, 2]

    def predict_single(self, processed_df: pd.DataFrame) -> dict:
        """
        Predict stage for a single patient.

        Args:
            processed_df: Preprocessed single-row DataFrame.

        Returns:
            dict with predicted_class, predicted_stage, probabilities, confidence.
        """
        proba = self.model.predict_proba(processed_df)[0]
        predicted_class = int(np.argmax(proba))
        predicted_stage = self.stage_map[predicted_class]
        confidence = float(proba[predicted_class]) * 100

        return {
            "predicted_class": predicted_class,
            "predicted_stage": predicted_stage,
            "confidence": confidence,
            "stage_color": self.stage_colors[predicted_stage],
            "stage_description": self.stage_descriptions[predicted_stage],
            "probabilities": {
                "Stage 1": float(proba[0]),
                "Stage 2": float(proba[1]),
                "Stage 3": float(proba[2]),
            },
            "proba_array": proba,
        }

    def predict_batch(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict stages for a batch of patients.

        Args:
            processed_df: Preprocessed DataFrame (multiple rows).

        Returns:
            DataFrame with prediction columns appended.
        """
        proba_matrix = self.model.predict_proba(processed_df)
        predicted_classes = np.argmax(proba_matrix, axis=1)

        results = pd.DataFrame({
            "Predicted_Stage": [self.stage_map[c] for c in predicted_classes],
            "Prediction_Probability": np.max(proba_matrix, axis=1).round(4),
            "Stage_1_Probability": proba_matrix[:, 0].round(4),
            "Stage_2_Probability": proba_matrix[:, 1].round(4),
            "Stage_3_Probability": proba_matrix[:, 2].round(4),
        })
        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importances from the XGBoost model.

        Returns:
            DataFrame with Feature and Importance columns, sorted descending.
        """
        from services.preprocessing_service import FEATURE_ORDER, FEATURE_DISPLAY_NAMES
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "Feature": FEATURE_ORDER,
            "Display_Name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_ORDER],
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        return df

    def get_confusion_matrix_data(self) -> dict:
        """
        Return pre-computed confusion matrix data from notebook results.
        XGBoost classifier confusion matrix (classes 1, 2, 3).
        """
        # From notebook outputs: XGBoost classification report
        # precision  recall  f1-score  support
        # 1: 0.91 / 0.89 / 0.90 / 617
        # 2: 0.87 / 0.89 / 0.88 / 640
        # 3: 0.94 / 0.94 / 0.94 / 671
        # Total: 1928 test samples
        # Reconstructed confusion matrix from classification report
        cm = np.array([
            [549, 57, 11],    # True Stage 1 predictions
            [37, 570, 33],    # True Stage 2 predictions
            [16, 24, 631],    # True Stage 3 predictions
        ])
        return {
            "matrix": cm,
            "labels": ["Stage 1", "Stage 2", "Stage 3"],
            "total": cm.sum(),
        }

    def get_classification_report_data(self) -> dict:
        """Return pre-computed classification report metrics from notebook."""
        return {
            "Stage 1": {
                "precision": 0.91, "recall": 0.89, "f1-score": 0.90, "support": 617
            },
            "Stage 2": {
                "precision": 0.87, "recall": 0.89, "f1-score": 0.88, "support": 640
            },
            "Stage 3": {
                "precision": 0.94, "recall": 0.94, "f1-score": 0.94, "support": 671
            },
            "macro avg": {
                "precision": 0.91, "recall": 0.91, "f1-score": 0.91, "support": 1928
            },
            "weighted avg": {
                "precision": 0.91, "recall": 0.91, "f1-score": 0.91, "support": 1928
            },
            "accuracy": 0.91,
        }

    def get_model_comparison_data(self) -> pd.DataFrame:
        """Return model comparison table from notebook experiments."""
        data = {
            "Model": [
                "Decision Tree",
                "Random Forest",
                "XGBoost (Baseline)",
                "CatBoost (Baseline)",
                "XGBoost Tuned ⭐",
                "CatBoost Tuned",
            ],
            "Accuracy": [0.7567, 0.8698, 0.9066, 0.9020, 0.9100, 0.9020],
            "Precision": [0.76, 0.87, 0.91, 0.90, 0.91, 0.90],
            "Recall": [0.76, 0.87, 0.91, 0.90, 0.91, 0.90],
            "F1_Score": [0.76, 0.87, 0.91, 0.90, 0.91, 0.90],
            "ROC_AUC": [0.840, 0.941, 0.965, 0.959, 0.9798, 0.9730],
            "Is_Best": [False, False, False, False, True, False],
        }
        return pd.DataFrame(data)

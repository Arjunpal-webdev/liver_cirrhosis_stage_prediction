"""
Preprocessing Service for Liver Cirrhosis Stage Prediction.
Handles all feature engineering and encoding transformations.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL ENCODING MAPS
# These mappings exactly replicate the LabelEncoder fitting order used in
# training: LabelEncoder was fit on each categorical column alphabetically.
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL_MAPS = {
    "Status":       {"C": 0, "CL": 1, "D": 2},
    "Drug":         {"D-penicillamine": 0, "Placebo": 1},
    "Sex":          {"F": 0, "M": 1},
    "Ascites":      {"N": 0, "Y": 1},
    "Hepatomegaly": {"N": 0, "Y": 1},
    "Spiders":      {"N": 0, "Y": 1},
    "Edema":        {"N": 0, "S": 1, "Y": 2},
}

# Exact feature order the model was trained on
FEATURE_ORDER = [
    "N_Days", "Status", "Drug", "Age", "Sex", "Ascites",
    "Hepatomegaly", "Spiders", "Edema", "Bilirubin", "Cholesterol",
    "Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides",
    "Platelets", "Prothrombin",
]

CATEGORICAL_COLS = list(CATEGORICAL_MAPS.keys())

NUMERICAL_COLS = [
    "N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin",
    "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
]

# Display labels (Age shown as Years for readability)
FEATURE_DISPLAY_NAMES = {
    "N_Days":       "N_Days",
    "Status":       "Status",
    "Drug":         "Drug",
    "Age":          "Age (Years)",
    "Sex":          "Sex",
    "Ascites":      "Ascites",
    "Hepatomegaly": "Hepatomegaly",
    "Spiders":      "Spiders",
    "Edema":        "Edema",
    "Bilirubin":    "Bilirubin",
    "Cholesterol":  "Cholesterol",
    "Albumin":      "Albumin",
    "Copper":       "Copper",
    "Alk_Phos":     "Alk_Phos",
    "SGOT":         "SGOT",
    "Tryglicerides":"Tryglicerides",
    "Platelets":    "Platelets",
    "Prothrombin":  "Prothrombin",
}


class PreprocessingService:
    """Handles all data preprocessing for the liver cirrhosis prediction model."""

    def __init__(self):
        self.categorical_maps = CATEGORICAL_MAPS
        self.feature_order = FEATURE_ORDER
        self.categorical_cols = CATEGORICAL_COLS
        self.numerical_cols = NUMERICAL_COLS

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fixed label encoding to categorical columns.
        Uses the same mappings as training — no re-fitting.
        """
        df = df.copy()
        for col, mapping in self.categorical_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                if df[col].isnull().any():
                    invalid = df[df[col].isnull()].index.tolist()
                    raise ValueError(
                        f"Column '{col}' contains unseen/invalid values at rows: {invalid}. "
                        f"Expected one of: {list(mapping.keys())}"
                    )
        return df

    def preprocess_single(self, input_dict: dict) -> pd.DataFrame:
        """
        Preprocess a single patient record from raw UI inputs.
        Converts Age from years to days and encodes categoricals.

        Args:
            input_dict: dict with raw feature values (Age in years).

        Returns:
            pd.DataFrame with a single row ready for model prediction.
        """
        data = input_dict.copy()

        # Convert Age from years to days
        if "Age_Years" in data:
            data["Age"] = int(data.pop("Age_Years") * 365.25)

        # Build dataframe in exact feature order
        df = pd.DataFrame([data], columns=self.feature_order)

        # Apply categorical encoding
        df = self.encode_categoricals(df)

        # Ensure correct dtypes
        for col in self.numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def preprocess_batch(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Preprocess a batch CSV upload.

        Args:
            df: Raw DataFrame from CSV upload.

        Returns:
            Tuple of (processed_df, list_of_warnings).
        """
        warnings = []
        df = df.copy()

        # Check for Age in years (if 'Age' column looks like years, convert)
        # If Age < 200, treat as years; otherwise treat as days
        if "Age" in df.columns:
            if df["Age"].median() < 200:
                df["Age"] = (df["Age"] * 365.25).astype(int)
                warnings.append(
                    "ℹ️ Age column detected as years — automatically converted to days for prediction."
                )

        # Validate required columns
        missing = [c for c in self.feature_order if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Keep only feature columns in correct order
        df = df[self.feature_order].copy()

        # Apply categorical encoding
        df = self.encode_categoricals(df)

        # Ensure numeric types
        for col in self.numerical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df, warnings

    def validate_batch_columns(self, df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
        """
        Validate that a batch DataFrame has the required columns.

        Returns:
            Tuple of (is_valid, missing_cols, extra_cols)
        """
        required = set(self.feature_order)
        present = set(df.columns)
        missing = list(required - present)
        extra = list(present - required - {"Stage"})  # Stage is OK if present
        is_valid = len(missing) == 0
        return is_valid, missing, extra

    def get_feature_display_name(self, feature: str) -> str:
        """Return display-friendly feature name."""
        return FEATURE_DISPLAY_NAMES.get(feature, feature)

    def get_all_display_names(self) -> dict:
        """Return all display name mappings."""
        return FEATURE_DISPLAY_NAMES.copy()

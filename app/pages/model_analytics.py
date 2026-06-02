"""
Model Analytics Page — Liver Cirrhosis Stage Prediction Dashboard.
Clean rewrite: All HTML is fully self-contained in single st.markdown calls.
7 tabs: Overview, Comparison, ROC AUC, Confusion Matrix,
        Classification Report, Feature Importance, Decision Boundary.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.components.ui_components import (
    render_section_header,
    render_metric_card,
    render_info_banner,
    render_stage_legend,
)
from services.prediction_service import PredictionService
from services.preprocessing_service import PreprocessingService
from services.analytics_service import AnalyticsService


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a key-value info card as a single HTML string
# ─────────────────────────────────────────────────────────────────────────────
def _kv_card(title: str, items: dict, value_color: str = "#e2e8f0",
             mono: bool = False) -> str:
    """Return a complete self-contained HTML card."""
    rows = ""
    for k, v in items.items():
        key_style = (
            "color:#94a3b8;font-size:0.85rem;"
            + ("font-family:monospace;" if mono else "")
        )
        val_style = (
            f"color:{value_color};font-size:0.85rem;font-weight:600;"
            + ("font-family:monospace;" if mono else "")
        )
        rows += f"""
        <div style="display:flex;justify-content:space-between;padding:0.4rem 0;
                    border-bottom:1px solid rgba(148,163,184,0.07);">
            <span style="{key_style}">{k}</span>
            <span style="{val_style}">{v}</span>
        </div>"""
    return f"""
    <div style="background:rgba(30,41,59,0.5);border:1px solid rgba(148,163,184,0.12);
                border-radius:14px;padding:1.25rem;">
        <div style="font-weight:700;color:#f1f5f9;margin-bottom:0.75rem;">{title}</div>
        {rows}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: ranking row card
# ─────────────────────────────────────────────────────────────────────────────
def _rank_card(medal: str, model_name: str, roc_auc: float, accuracy: float,
               is_best: bool) -> str:
    bg     = "rgba(59,130,246,0.08)" if is_best else "rgba(30,41,59,0.4)"
    border = "rgba(59,130,246,0.3)"  if is_best else "rgba(148,163,184,0.1)"
    return f"""
    <div style="background:{bg};border:1px solid {border};border-radius:10px;
                padding:0.75rem 1rem;display:flex;align-items:center;gap:1rem;
                margin-bottom:0.4rem;">
        <span style="font-size:1.2rem;min-width:28px;">{medal}</span>
        <span style="color:#f1f5f9;font-weight:600;flex:1;">{model_name}</span>
        <span style="color:#a855f7;font-weight:700;font-size:0.9rem;">
            ROC AUC: {roc_auc:.4f}
        </span>
        <span style="color:#22c55e;font-size:0.85rem;">
            Acc: {accuracy:.4f}
        </span>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: per-class classification report card
# ─────────────────────────────────────────────────────────────────────────────
def _class_report_card(cls: str, color: str, m: dict) -> str:
    return f"""
    <div style="background:rgba(30,41,59,0.5);border:1px solid rgba(148,163,184,0.1);
                border-left:4px solid {color};border-radius:14px;
                padding:1rem 1.25rem;margin-bottom:0.6rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;
                    flex-wrap:wrap;gap:0.75rem;">
            <span style="color:{color};font-weight:700;font-size:1.05rem;">{cls}</span>
            <div style="display:flex;gap:2rem;flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;
                                letter-spacing:0.06em;font-weight:600;">Precision</div>
                    <div style="color:#3b82f6;font-size:1.3rem;font-weight:800;">{m['precision']:.2f}</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;
                                letter-spacing:0.06em;font-weight:600;">Recall</div>
                    <div style="color:#22c55e;font-size:1.3rem;font-weight:800;">{m['recall']:.2f}</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;
                                letter-spacing:0.06em;font-weight:600;">F1 Score</div>
                    <div style="color:#a855f7;font-size:1.3rem;font-weight:800;">{m['f1-score']:.2f}</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;
                                letter-spacing:0.06em;font-weight:600;">Support</div>
                    <div style="color:#f59e0b;font-size:1.3rem;font-weight:800;">{m['support']:,}</div>
                </div>
            </div>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: averages card
# ─────────────────────────────────────────────────────────────────────────────
def _avg_card(label: str, color: str, m: dict) -> str:
    return f"""
    <div style="background:rgba(30,41,59,0.5);border:1px solid rgba(148,163,184,0.1);
                border-top:3px solid {color};border-radius:14px;padding:1.25rem;">
        <div style="color:{color};font-weight:700;font-size:0.9rem;
                    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.75rem;">
            {label}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
            <div>
                <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;">Precision</div>
                <div style="color:#3b82f6;font-weight:700;font-size:1.1rem;">{m['precision']:.4f}</div>
            </div>
            <div>
                <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;">Recall</div>
                <div style="color:#22c55e;font-weight:700;font-size:1.1rem;">{m['recall']:.4f}</div>
            </div>
            <div>
                <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;">F1 Score</div>
                <div style="color:#a855f7;font-weight:700;font-size:1.1rem;">{m['f1-score']:.4f}</div>
            </div>
            <div>
                <div style="color:#64748b;font-size:0.72rem;text-transform:uppercase;">Support</div>
                <div style="color:#f59e0b;font-weight:700;font-size:1.1rem;">{m['support']:,}</div>
            </div>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def render_model_analytics_page(
    prediction_service: PredictionService,
    preprocessing_service: PreprocessingService,
    analytics_service: AnalyticsService,
):
    """Render the full Model Analytics page with 7 tabs."""

    render_section_header(
        "Model Analytics & Performance",
        "Comprehensive analysis of the XGBoost model's performance metrics",
        icon="📊",
    )

    tabs = st.tabs([
        "🏆 Overview",
        "⚖️ Comparison",
        "📈 ROC AUC",
        "🔢 Confusion Matrix",
        "📋 Classification Report",
        "🌟 Feature Importance",
        "🗺️ Decision Boundary",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — MODEL OVERVIEW
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        render_section_header("Best Model Overview", icon="🏆")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        metrics_data = [
            ("Best Model",    "XGBoost",  "Tuned with RandomizedSearchCV", "#3b82f6", "🤖", m_col1),
            ("Accuracy",      "91.0%",    "On hold-out test set",          "#22c55e", "✅", m_col2),
            ("Macro ROC AUC", "0.9798",   "Multi-class one-vs-rest",       "#a855f7", "📈", m_col3),
            ("Classes",       "3 Stages", "Stage 1, 2, 3",                 "#f59e0b", "🎯", m_col4),
        ]
        for title, value, subtitle, color, icon, col in metrics_data:
            with col:
                st.markdown(render_metric_card(title, value, subtitle, color, icon),
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m2c1, m2c2, m2c3, m2c4 = st.columns(4)
        secondary = [
            ("N Estimators",  "200",  "XGBoost trees",    "#06b6d4", "🌲", m2c1),
            ("Max Depth",     "7",    "Tree depth",        "#f59e0b", "📏", m2c2),
            ("Learning Rate", "0.10", "eta parameter",     "#22c55e", "⚡", m2c3),
            ("Features",      "18",   "Input features",    "#ef4444", "📊", m2c4),
        ]
        for title, value, subtitle, color, icon, col in secondary:
            with col:
                st.markdown(render_metric_card(title, value, subtitle, color, icon),
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dataset info + hyperparams — both fully self-contained HTML blocks
        render_section_header("Dataset Information", icon="🗃️")
        d_col1, d_col2 = st.columns(2, gap="large")

        with d_col1:
            st.markdown(_kv_card("📁 Dataset Details", {
                "Dataset":        "Liver Cirrhosis Dataset",
                "Source":         "Clinical trial data (PBC study)",
                "Total Samples":  "~9,641",
                "Test Set Size":  "1,928 samples (20%)",
                "Target Variable":"Stage (1, 2, 3)",
                "Features":       "18 clinical features",
                "Missing Values": "Imputed with median",
            }), unsafe_allow_html=True)

        with d_col2:
            st.markdown(_kv_card("⚙️ Best Model Hyperparameters", {
                "subsample":       "0.8",
                "n_estimators":    "200",
                "max_depth":       "7",
                "learning_rate":   "0.1",
                "gamma":           "0.2",
                "colsample_bytree":"0.6",
                "objective":       "multi:softprob",
                "eval_metric":     "mlogloss",
                "Tuning":          "RandomizedSearchCV (50 iter, 3-fold)",
            }, value_color="#60a5fa", mono=True), unsafe_allow_html=True)

        render_stage_legend()

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — MODEL COMPARISON
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        render_section_header("Model Comparison", icon="⚖️")
        comparison_df = prediction_service.get_model_comparison_data()

        render_info_banner(
            "⭐ <strong>Best Model: Tuned XGBoost</strong> — Highest Accuracy (91%) and "
            "Macro ROC AUC (0.9798) on the test set.",
            color="#f59e0b", icon="🏆"
        )

        render_section_header("Metrics Comparison Table", icon="📋")
        display_df = comparison_df.drop(columns=["Is_Best"]).rename(columns={
            "F1_Score": "F1 Score", "ROC_AUC": "ROC AUC",
        })

        def highlight_best(row):
            if "⭐" in str(row["Model"]):
                return ["background-color: rgba(59,130,246,0.12); font-weight: 700;"] * len(row)
            return [""] * len(row)

        styled_df = display_df.style.apply(highlight_best, axis=1).format({
            "Accuracy":  "{:.4f}",
            "Precision": "{:.4f}",
            "Recall":    "{:.4f}",
            "F1 Score":  "{:.4f}",
            "ROC AUC":   "{:.4f}",
        })
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        render_section_header("Performance Comparison — Bar Chart", icon="📊")
        bar_fig = analytics_service.build_model_comparison_bar(comparison_df)
        st.plotly_chart(bar_fig, use_container_width=True)

        render_section_header("Performance Comparison — Radar Chart", icon="🕸️")
        radar_fig = analytics_service.build_radar_chart(comparison_df)
        st.plotly_chart(radar_fig, use_container_width=True)

        render_section_header("Model Rankings by ROC AUC", icon="🏅")
        ranked_df = comparison_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]

        # Build all ranking cards as ONE HTML block
        rankings_html = ""
        for i, row in ranked_df.iterrows():
            rankings_html += _rank_card(
                medals[i], row["Model"], row["ROC_AUC"], row["Accuracy"],
                bool(row.get("Is_Best", False))
            )
        st.markdown(rankings_html, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — ROC AUC ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        render_section_header("ROC AUC Analysis", icon="📈")

        render_info_banner(
            "ROC curves computed using One-vs-Rest (OvR) strategy for multi-class classification. "
            "Macro-averaged AUC reported as the unweighted mean of class-specific AUCs.",
            color="#3b82f6", icon="📊"
        )

        auc_cols = st.columns(4)
        auc_data = [
            ("XGBoost Macro AUC", "0.9798", "#3b82f6", "🏆"),
            ("CatBoost Macro AUC","0.9730", "#a855f7", "📊"),
            ("Stage 1 AUC",       "~0.98",  "#22c55e", "🟢"),
            ("Stage 3 AUC",       "~0.99",  "#ef4444", "🔴"),
        ]
        for i, (title, val, color, icon) in enumerate(auc_data):
            with auc_cols[i]:
                st.markdown(
                    render_metric_card(title, val, "XGBoost best model", color, icon),
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        render_section_header("Multi-Class ROC Curves — Best XGBoost", icon="📈")
        roc_fig = analytics_service.build_roc_curve_plotly()
        st.plotly_chart(roc_fig, use_container_width=True)

        render_section_header("XGBoost vs CatBoost — ROC Comparison", icon="⚖️")
        comp_roc_fig = analytics_service.build_comparison_roc_plotly()
        st.plotly_chart(comp_roc_fig, use_container_width=True)

        render_info_banner(
            "Note: ROC curves shown are representative approximations based on published AUC values "
            "from the training notebook. Actual curves are generated from held-out test set predictions.",
            color="#64748b", icon="ℹ️"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 — CONFUSION MATRIX
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        render_section_header("Confusion Matrix", icon="🔢")

        cm_data = prediction_service.get_confusion_matrix_data()
        cm     = cm_data["matrix"]
        labels = cm_data["labels"]
        correct = int(np.trace(cm))
        total   = int(cm.sum())
        accuracy = correct / total * 100

        cm_cols = st.columns(4)
        with cm_cols[0]:
            st.markdown(render_metric_card("Accuracy",     f"{accuracy:.1f}%",
                                           "Overall correct", "#22c55e", "✅"),
                        unsafe_allow_html=True)
        with cm_cols[1]:
            st.markdown(render_metric_card("Correct",      f"{correct:,}",
                                           f"of {total:,}", "#3b82f6", "🎯"),
                        unsafe_allow_html=True)
        with cm_cols[2]:
            errors = total - correct
            st.markdown(render_metric_card("Errors",       f"{errors:,}",
                                           f"{errors/total*100:.1f}% error rate", "#ef4444", "❌"),
                        unsafe_allow_html=True)
        with cm_cols[3]:
            st.markdown(render_metric_card("Test Samples", f"{total:,}",
                                           "Hold-out test set", "#a855f7", "📊"),
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        cm_fig = analytics_service.build_confusion_matrix_heatmap(cm_data)
        st.plotly_chart(cm_fig, use_container_width=True)

        render_section_header("Per-Class Accuracy Breakdown", icon="📊")
        class_cols   = st.columns(3)
        class_colors = ["#22c55e", "#f59e0b", "#ef4444"]
        for i, (label, color) in enumerate(zip(labels, class_colors)):
            with class_cols[i]:
                class_acc = cm[i, i] / cm[i, :].sum() * 100
                st.markdown(render_metric_card(
                    label, f"{class_acc:.1f}%",
                    f"Recall: {cm[i,i]}/{cm[i,:].sum()} correct",
                    color, "🎯"
                ), unsafe_allow_html=True)

        with st.expander("📋 Raw Confusion Matrix Data", expanded=False):
            cm_display = pd.DataFrame(
                cm,
                index=[f"Actual {l}"   for l in labels],
                columns=[f"Pred {l}" for l in labels]
            )
            st.dataframe(cm_display, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 5 — CLASSIFICATION REPORT
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        render_section_header("Classification Report", icon="📋")

        report_data = prediction_service.get_classification_report_data()
        overall_acc = report_data.get("accuracy", 0.91)

        oa_col1, oa_col2, oa_col3 = st.columns([1, 2, 1])
        with oa_col2:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(34,197,94,0.12),rgba(59,130,246,0.08));
                        border:1px solid rgba(34,197,94,0.25);border-radius:16px;
                        padding:1.5rem;text-align:center;margin-bottom:1.5rem;">
                <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase;
                            letter-spacing:0.08em;font-weight:600;margin-bottom:0.3rem;">
                    Overall Accuracy
                </div>
                <div style="font-size:3rem;font-weight:900;color:#22c55e;
                            letter-spacing:-0.04em;">{overall_acc*100:.1f}%</div>
                <div style="font-size:0.8rem;color:#64748b;margin-top:0.25rem;">
                    on 1,928 test samples
                </div>
            </div>
            """, unsafe_allow_html=True)

        classes       = ["Stage 1", "Stage 2", "Stage 3"]
        class_colors  = ["#22c55e", "#f59e0b", "#ef4444"]

        # Build all per-class cards as ONE HTML string
        class_cards_html = ""
        for cls, color in zip(classes, class_colors):
            if cls in report_data:
                class_cards_html += _class_report_card(cls, color, report_data[cls])
        st.markdown(class_cards_html, unsafe_allow_html=True)

        render_section_header("Macro & Weighted Averages", icon="📐")
        avg_cols = st.columns(2)
        avg_entries = [("macro avg", "#06b6d4"), ("weighted avg", "#f59e0b")]
        for i, (avg_key, color) in enumerate(avg_entries):
            if avg_key in report_data:
                with avg_cols[i]:
                    st.markdown(
                        _avg_card(avg_key.title(), color, report_data[avg_key]),
                        unsafe_allow_html=True
                    )

        render_section_header("Metrics Comparison — Per Class", icon="📊")
        cr_fig = go.Figure()
        metrics      = ["precision", "recall", "f1-score"]
        metric_labels= ["Precision", "Recall", "F1 Score"]
        colors_bar   = ["#3b82f6", "#22c55e", "#a855f7"]

        for metric, label, color in zip(metrics, metric_labels, colors_bar):
            cr_fig.add_trace(go.Bar(
                name=label,
                x=classes,
                y=[report_data[cls][metric] for cls in classes if cls in report_data],
                marker_color=color,
                text=[f"{report_data[cls][metric]:.2f}" for cls in classes if cls in report_data],
                textposition="outside",
                textfont={"color": "#e2e8f0"},
            ))

        cr_fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.1], gridcolor="rgba(255,255,255,0.1)", color="#94a3b8"),
            xaxis={"color": "#94a3b8"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0", "family": "Inter"},
            legend={"bgcolor": "rgba(30,41,59,0.8)"},
            height=380,
        )
        st.plotly_chart(cr_fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 6 — FEATURE IMPORTANCE
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        render_section_header("Feature Importance — XGBoost", icon="🌟")

        importance_df = prediction_service.get_feature_importance()
        top_feature   = importance_df.iloc[0]

        render_info_banner(
            f"🏆 Most important feature: <strong>{top_feature['Display_Name']}</strong> "
            f"(importance score: <strong>{top_feature['Importance']:.4f}</strong>)",
            color="#f59e0b", icon="⭐"
        )

        fi_col1, fi_col2, fi_col3 = st.columns([1, 2, 1])
        with fi_col2:
            top_n = st.select_slider(
                "Number of features to display",
                options=[5, 10, 15, 18],
                value=15,
                key="feature_importance_n",
            )

        fi_fig = analytics_service.build_feature_importance_chart(importance_df, top_n=top_n)
        st.plotly_chart(fi_fig, use_container_width=True)

        render_section_header("Complete Feature Ranking", icon="📋")
        rank_df = importance_df[["Rank", "Display_Name", "Importance"]].rename(
            columns={"Display_Name": "Feature", "Importance": "Importance Score"}
        )
        rank_df["Importance Score"] = rank_df["Importance Score"].round(6)
        max_imp = rank_df["Importance Score"].max()
        rank_df["Visual"] = rank_df["Importance Score"].apply(
            lambda x: "█" * int(x / max_imp * 20)
        )
        st.dataframe(
            rank_df, use_container_width=True, hide_index=True,
            column_config={
                "Rank":             st.column_config.NumberColumn("Rank", width="small"),
                "Feature":          st.column_config.TextColumn("Feature"),
                "Importance Score": st.column_config.NumberColumn("Importance", format="%.6f"),
                "Visual":           st.column_config.TextColumn("Relative Importance"),
            }
        )

        render_section_header("Feature Importance Treemap", icon="🗺️")
        treemap_fig = go.Figure(go.Treemap(
            labels=importance_df["Display_Name"].tolist(),
            parents=[""] * len(importance_df),
            values=importance_df["Importance"].tolist(),
            marker=dict(
                colors=importance_df["Importance"].tolist(),
                colorscale="Blues",
                showscale=True,
            ),
            textfont={"size": 12, "color": "white"},
            hovertemplate="<b>%{label}</b><br>Importance: %{value:.4f}<extra></extra>",
        ))
        treemap_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0", "family": "Inter"},
            height=450,
            margin={"t": 30, "b": 10, "l": 10, "r": 10},
        )
        st.plotly_chart(treemap_fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 7 — DECISION BOUNDARY
    # ═════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        render_section_header("Decision Boundary Visualization", icon="🗺️")

        render_info_banner(
            "PCA (Principal Component Analysis) reduces the 18-dimensional feature space to 2D "
            "for visualization. Colors represent actual cirrhosis stages.",
            color="#3b82f6", icon="📐"
        )

        with st.spinner("Generating visualization..."):
            pca_fig = analytics_service.build_pca_visualization()

        render_stage_legend()
        st.plotly_chart(pca_fig, use_container_width=True)

        render_info_banner(
            "Each point represents a patient from the dataset. "
            "Clear clusters indicate the XGBoost model can linearly separate the stages well "
            "even in a reduced 2D projection.",
            color="#22c55e", icon="💡"
        )

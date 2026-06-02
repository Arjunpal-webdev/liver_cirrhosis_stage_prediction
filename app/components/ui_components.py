"""
Reusable UI Components for the Liver Cirrhosis Dashboard.
All components return HTML strings or render directly via Streamlit.
"""

import streamlit as st


def render_dashboard_header():
    """Render the main dashboard header with logo, title, and badges."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0f172a 100%);
        border: 1px solid rgba(59,130,246,0.25);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 40px rgba(0,0,0,0.5);
    ">
        <!-- Glow effect -->
        <div style="
            position: absolute; top: -60px; right: -60px;
            width: 200px; height: 200px;
            background: radial-gradient(circle, rgba(59,130,246,0.2), transparent 70%);
            pointer-events: none;
        "></div>
        <div style="
            position: absolute; bottom: -60px; left: 30%;
            width: 300px; height: 200px;
            background: radial-gradient(circle, rgba(6,182,212,0.1), transparent 70%);
            pointer-events: none;
        "></div>

        <div style="display: flex; align-items: center; gap: 1.2rem; margin-bottom: 0.8rem;">
            <div style="
                width: 56px; height: 56px;
                background: linear-gradient(135deg, #1d4ed8, #3b82f6);
                border-radius: 14px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.8rem;
                box-shadow: 0 4px 20px rgba(59,130,246,0.4);
                flex-shrink: 0;
            ">🫀</div>
            <div>
                <h1 style="
                    margin: 0;
                    font-size: 1.85rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, #f1f5f9, #94a3b8);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing: -0.03em;
                    line-height: 1.2;
                ">Liver Cirrhosis Stage</h1>
                <h2 style="
                    margin: 0;
                    font-size: 1.85rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, #3b82f6, #06b6d4);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    letter-spacing: -0.03em;
                    line-height: 1.2;
                ">Prediction & Analytics</h2>
            </div>
        </div>

        <p style="
            color: #94a3b8;
            margin: 0.5rem 0 1rem 0;
            font-size: 0.95rem;
            font-weight: 400;
            max-width: 600px;
        ">AI-powered clinical decision support using XGBoost with 91% accuracy & 0.9798 ROC AUC</p>

        <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
            <span style="
                background: rgba(59,130,246,0.15); color: #60a5fa;
                border: 1px solid rgba(59,130,246,0.3);
                border-radius: 20px; padding: 0.25rem 0.75rem;
                font-size: 0.78rem; font-weight: 600;
            ">🤖 XGBoost Tuned</span>
            <span style="
                background: rgba(34,197,94,0.15); color: #4ade80;
                border: 1px solid rgba(34,197,94,0.3);
                border-radius: 20px; padding: 0.25rem 0.75rem;
                font-size: 0.78rem; font-weight: 600;
            ">✅ Accuracy: 91%</span>
            <span style="
                background: rgba(168,85,247,0.15); color: #c084fc;
                border: 1px solid rgba(168,85,247,0.3);
                border-radius: 20px; padding: 0.25rem 0.75rem;
                font-size: 0.78rem; font-weight: 600;
            ">📊 ROC AUC: 0.9798</span>
            <span style="
                background: rgba(245,158,11,0.15); color: #fbbf24;
                border: 1px solid rgba(245,158,11,0.3);
                border-radius: 20px; padding: 0.25rem 0.75rem;
                font-size: 0.78rem; font-weight: 600;
            ">🔍 SHAP Explainability</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "",
                       color: str = "#3b82f6", icon: str = "📊") -> str:
    """Return HTML for a metric card."""
    return f"""
    <div style="
        background: linear-gradient(145deg, #1e293b, #253147);
        border: 1px solid rgba(148,163,184,0.12);
        border-left: 4px solid {color};
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        height: 100%;
    ">
        <div style="display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.6rem;">
            <span style="font-size: 1.4rem;">{icon}</span>
            <span style="color: #94a3b8; font-size: 0.78rem; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 0.06em;">{title}</span>
        </div>
        <div style="font-size: 2rem; font-weight: 800; color: {color};
                    letter-spacing: -0.03em; line-height: 1.1;">{value}</div>
        {f'<div style="color: #64748b; font-size: 0.8rem; margin-top: 0.4rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """


def render_prediction_result_card(stage: str, confidence: float,
                                  description: str, color: str):
    """Render a large prediction result card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(145deg, #1e293b, #253147);
        border: 2px solid {color}40;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 60px {color}15;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -40px; left: 50%;
            transform: translateX(-50%);
            width: 200px; height: 120px;
            background: radial-gradient(ellipse, {color}20, transparent 70%);
            pointer-events: none;
        "></div>

        <div style="
            display: inline-flex; align-items: center; justify-content: center;
            width: 72px; height: 72px;
            background: {color}20;
            border: 2px solid {color}50;
            border-radius: 50%;
            font-size: 2rem;
            margin-bottom: 1rem;
        ">🎯</div>

        <p style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">
            Predicted Diagnosis
        </p>

        <h2 style="
            font-size: 2.8rem;
            font-weight: 900;
            color: {color};
            margin: 0 0 0.25rem 0;
            letter-spacing: -0.04em;
        ">{stage}</h2>

        <div style="
            background: {color}15;
            border: 1px solid {color}30;
            border-radius: 30px;
            padding: 0.4rem 1.2rem;
            display: inline-block;
            margin: 0.5rem 0 1rem 0;
        ">
            <span style="color: {color}; font-weight: 700; font-size: 1.1rem;">
                {confidence:.1f}% Confidence
            </span>
        </div>

        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0; max-width: 400px;
                  margin-left: auto; margin-right: auto;">
            {description}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = "", icon: str = "📌"):
    """Render a section header with icon and subtitle."""
    st.markdown(f"""
    <div style="margin: 1.5rem 0 1rem 0;">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <h3 style="
                margin: 0;
                font-size: 1.3rem;
                font-weight: 700;
                color: #f1f5f9;
                letter-spacing: -0.02em;
            ">{title}</h3>
        </div>
        {f'<p style="color: #64748b; margin: 0.3rem 0 0 2.25rem; font-size: 0.88rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_info_banner(message: str, color: str = "#3b82f6", icon: str = "ℹ️"):
    """Render a colored info banner."""
    st.markdown(f"""
    <div style="
        background: {color}10;
        border: 1px solid {color}30;
        border-left: 4px solid {color};
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.75rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    ">
        <span style="font-size: 1rem; flex-shrink: 0; margin-top: 1px;">{icon}</span>
        <span style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_nav():
    """Render the sidebar navigation content."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
            <div style="
                width: 60px; height: 60px;
                background: linear-gradient(135deg, #1d4ed8, #3b82f6);
                border-radius: 16px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.8rem;
                margin: 0 auto 0.75rem auto;
                box-shadow: 0 4px 20px rgba(59,130,246,0.4);
            ">🫀</div>
            <div style="
                font-size: 1rem; font-weight: 700; color: #f1f5f9;
                letter-spacing: -0.01em;
            ">Liver Cirrhosis AI</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.2rem;">
                Clinical Decision Support
            </div>
        </div>
        <hr style="border: none; border-top: 1px solid rgba(148,163,184,0.12); margin: 0 0 1rem 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=["🔬 Single Prediction", "📋 Batch Prediction", "📊 Model Analytics"],
            label_visibility="collapsed",
        )

        model_cards = [
            ("🤖", "Model", "XGBoost Tuned", "#3b82f6"),
            ("✅", "Accuracy", "91.0%", "#22c55e"),
            ("📈", "ROC AUC", "0.9798", "#a855f7"),
            ("🎯", "Classes", "3 Stages", "#f59e0b"),
        ]

        cards_html = "".join([
            f'''<div style="
                background: rgba(30,41,59,0.6);
                border: 1px solid rgba(148,163,184,0.1);
                border-left: 3px solid {color};
                border-radius: 8px;
                padding: 0.5rem 0.75rem;
                margin-bottom: 0.4rem;
                display: flex; align-items: center; gap: 0.5rem;
            ">
                <span style="font-size: 0.9rem;">{icon}</span>
                <div>
                    <div style="font-size: 0.68rem; color: #64748b; font-weight: 600;
                                text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
                    <div style="font-size: 0.85rem; color: {color}; font-weight: 700;">{val}</div>
                </div>
            </div>'''
            for icon, label, val, color in model_cards
        ])

        st.markdown(f"""
        <hr style="border: none; border-top: 1px solid rgba(148,163,184,0.12); margin: 1rem 0;">
        <div style="padding: 0.5rem 0;">
            <div style="font-size: 0.72rem; color: #475569; text-transform: uppercase;
                        letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.75rem;">
                Model Info
            </div>
            {cards_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: auto; padding-top: 1.5rem;">
            <div style="
                background: rgba(59,130,246,0.08);
                border: 1px solid rgba(59,130,246,0.2);
                border-radius: 10px;
                padding: 0.75rem;
                font-size: 0.75rem;
                color: #64748b;
                text-align: center;
            ">
                ⚠️ For research use only.<br>
                Not a substitute for medical advice.
            </div>
        </div>
        """, unsafe_allow_html=True)

    return page


def render_shap_explanation_header():
    """Render SHAP section header with explanation text."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(168,85,247,0.08), rgba(59,130,246,0.08));
        border: 1px solid rgba(168,85,247,0.2);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    ">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.4rem;">🔍</span>
            <h4 style="margin: 0; color: #c084fc; font-weight: 700; font-size: 1.1rem;">
                SHAP Explainability — Why This Prediction?
            </h4>
        </div>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 0; line-height: 1.6;">
            These features contributed most to the predicted stage. 
            <span style="color: #ef4444; font-weight: 600;">Red bars ↑</span> push the prediction 
            towards this stage, while 
            <span style="color: #22c55e; font-weight: 600;">green bars ↓</span> push away from it.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_feature_badge(feature: str, value: str, shap_val: float):
    """Render a feature badge showing value and SHAP direction."""
    is_positive = shap_val > 0
    color = "#ef4444" if is_positive else "#22c55e"
    icon = "↑" if is_positive else "↓"
    bg = "rgba(239,68,68,0.1)" if is_positive else "rgba(34,197,94,0.1)"
    border = "rgba(239,68,68,0.3)" if is_positive else "rgba(34,197,94,0.3)"

    return f"""
    <div style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    ">
        <span style="color: #94a3b8; font-size: 0.72rem; font-weight: 600;
                     text-transform: uppercase; letter-spacing: 0.05em;">{feature}</span>
        <div style="display: flex; align-items: baseline; gap: 0.3rem;">
            <span style="color: {color}; font-weight: 700; font-size: 1rem;">{icon} {value}</span>
            <span style="color: {color}; font-size: 0.75rem; opacity: 0.8;">
                ({'+' if is_positive else ''}{shap_val:.3f})
            </span>
        </div>
    </div>
    """


def render_stage_legend():
    """Render a color-coded stage legend."""
    st.markdown("""
    <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.5rem 0;">
        <div style="display: flex; align-items: center; gap: 0.4rem;">
            <div style="width: 12px; height: 12px; border-radius: 50%;
                        background: #22c55e; flex-shrink: 0;"></div>
            <span style="color: #94a3b8; font-size: 0.8rem;">Stage 1 — Early</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.4rem;">
            <div style="width: 12px; height: 12px; border-radius: 50%;
                        background: #f59e0b; flex-shrink: 0;"></div>
            <span style="color: #94a3b8; font-size: 0.8rem;">Stage 2 — Moderate</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.4rem;">
            <div style="width: 12px; height: 12px; border-radius: 50%;
                        background: #ef4444; flex-shrink: 0;"></div>
            <span style="color: #94a3b8; font-size: 0.8rem;">Stage 3 — Advanced</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_loading_spinner(message: str = "Processing..."):
    """Show a styled loading indicator."""
    st.markdown(f"""
    <div style="
        text-align: center; padding: 2rem;
        color: #94a3b8; font-size: 0.9rem;
    ">
        <div style="
            display: inline-block;
            width: 40px; height: 40px;
            border: 3px solid rgba(59,130,246,0.2);
            border-top-color: #3b82f6;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-bottom: 0.75rem;
        "></div>
        <br>{message}
        <style>
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        </style>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(title: str, message: str, icon: str = "📭"):
    """Render an empty state placeholder."""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(30,41,59,0.4);
        border: 1px dashed rgba(148,163,184,0.2);
        border-radius: 16px;
        margin: 1rem 0;
    ">
        <div style="font-size: 3rem; margin-bottom: 0.75rem;">{icon}</div>
        <h4 style="color: #e2e8f0; font-weight: 600; margin: 0 0 0.4rem 0;">{title}</h4>
        <p style="color: #64748b; font-size: 0.88rem; margin: 0; max-width: 400px;
                  margin-left: auto; margin-right: auto;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

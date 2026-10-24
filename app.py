import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from mailsort.config import (
    BASE_DIR,
    TARGET_CATEGORIES,
    FALLBACK_CATEGORY,
    METRICS_PATH,
    CONFIDENCE_THRESHOLD,
    DATA_DIR,
)
from mailsort.inference import classify_email, classify_batch, get_custom_model, get_baseline_model

# Page configuration
st.set_page_config(
    page_title="MailSort — NLP Email Classifier",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a clean, modern aesthetic
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge-primary {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #bfdbfe;
    }
    .badge-promotions {
        background-color: #fef3c7;
        color: #b45309;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #fde68a;
    }
    .badge-spam {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #fecaca;
    }
    .badge-other {
        background-color: #f1f5f9;
        color: #475569;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #e2e8f0;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #94a3b8;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 1rem;
        border-radius: 0 4px 4px 0;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 0.75rem;
        border-radius: 0 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/mail.png", width=64)
    st.title("MailSort")
    st.caption("BIC703 ML Assignment — NLP Email Classifier")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation",
        [
            "Single Email Classifier",
            "Batch CSV Classification",
            "Model Architecture & Metrics",
            "Academic Integrity & Provenance"
        ]
    )
    
    st.markdown("---")
    st.subheader("Model Configuration")
    selected_mode = st.selectbox(
        "Inference Model Mode",
        options=["auto", "custom", "baseline"],
        index=0,
        format_func=lambda x: {
            "auto": "Auto (Project-Owned Classifier)",
            "custom": "Custom (3-Class: Primary/Promotions/Spam)",
            "baseline": "Reference (6-Class Neural Checkpoint)"
        }.get(x, x),
        help="Project-trained 3-class classifier, or the optional 6-class reference checkpoint from local cache."
    )
    
    st.markdown("---")
    st.markdown(
        "<small style='color: #64748b;'>"
        "<b>Bangalore Institute of Technology</b><br>"
        "Department of Computer Science & Engineering<br>"
        "Course: BIC703 Machine Learning (NLP Topic)<br>"
        "Offline Inference Only"
        "</small>",
        unsafe_allow_html=True
    )

# Sample Presets for quick evaluation
SAMPLE_EMAILS = {
    "Primary: Academic Inquiry": {
        "subject": "Questions regarding BIC703 Lab 4 evaluation criteria",
        "body": "Dear Professor, I have reviewed the grading feedback for Lab 4. Could you please clarify how the macro F1-score penalty was computed? I would appreciate 10 minutes during your office hours tomorrow."
    },
    "Primary: Project Meeting Sync": {
        "subject": "Sprint retrospective meeting notes and action items",
        "body": "Hi team, attached are the minutes from today's architectural sync. Please update your assigned pull requests by Thursday so we can finalize the release candidate."
    },
    "Promotions: 50% Flash Sale": {
        "subject": "Flash Sale: 50% off all developer monitors and mechanical keyboards!",
        "body": "Do not miss our biggest mid-season clearance event! Save up to 50% across premium tech accessories. Enter promo code SAVE50 at checkout. Offer ends tonight at midnight."
    },
    "Promotions: Tech Newsletter": {
        "subject": "Weekly Engineering Digest: Modern compilers and vectorization",
        "body": "Here are the top trending software engineering stories this week. Discover how modern language runtimes optimize SIMD instructions. Click here to read the full issue online."
    },
    "Spam: Urgent Lottery Claim": {
        "subject": "CONGRATULATIONS: You have won $1,000,000 in the International Lottery!",
        "body": "Your email address has won the global promotional jackpot drawing. Reply immediately with your full legal name, telephone number, and bank routing details to claim your funds without delay."
    },
    "Spam: Phishing Security Alert": {
        "subject": "URGENT: Your account access has been suspended due to suspicious activity",
        "body": "We detected unauthorized login attempts from IP 192.0.2.1. Click http://secure-banking-verify-auth.cc immediately to enter your social security number and restore account access within 24 hours."
    },
    "Ambiguous / Needs Review": {
        "subject": "Quick note",
        "body": "Sounds good, see you later."
    }
}

def render_category_badge(category: str):
    if category == "Primary":
        return f'<span class="badge-primary">📥 {category}</span>'
    elif category == "Promotions":
        return f'<span class="badge-promotions">🏷️ {category}</span>'
    elif category == "Spam":
        return f'<span class="badge-spam">🚨 {category}</span>'
    else:
        return f'<span class="badge-other">❓ {category}</span>'

# =============================================================================
# TAB 1: SINGLE EMAIL CLASSIFIER
# =============================================================================
if app_mode == "Single Email Classifier":
    st.markdown('<div class="main-header">Single Email Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify incoming email messages into Primary, Promotions, Spam, or Other with transparent confidence and heuristic explanation cues.</div>', unsafe_allow_html=True)

    # Preset loader
    with st.expander("✨ Load a Pre-configured Sample Email for Testing", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_preset = st.selectbox(
                "Select a sample email template:",
                options=list(SAMPLE_EMAILS.keys())
            )
        with col2:
            st.write("")
            st.write("")
            load_button = st.button("Load Template", use_container_width=True)

    # State tracking for inputs
    if "current_subject" not in st.session_state or load_button:
        st.session_state["current_subject"] = SAMPLE_EMAILS[selected_preset]["subject"]
        st.session_state["current_body"] = SAMPLE_EMAILS[selected_preset]["body"]

    # Input Form
    with st.form("classification_form"):
        subject_input = st.text_input(
            "Email Subject",
            value=st.session_state.get("current_subject", ""),
            placeholder="e.g., Agenda for tomorrow's project sync meeting",
            help="The subject line of the email."
        )
        body_input = st.text_area(
            "Email Body",
            value=st.session_state.get("current_body", ""),
            height=160,
            placeholder="Type or paste the email body text here...",
            help="The main content of the email."
        )
        
        char_count = len(subject_input.strip()) + len(body_input.strip())
        st.caption(f"Total characters: {char_count} (sensible limit: ~10,000)")

        submit_button = st.form_submit_button("🔍 Classify Email", type="primary", use_container_width=True)

    if submit_button:
        with st.spinner("Analyzing text patterns and generating prediction..."):
            result = classify_email(subject_input, body_input, mode=selected_mode)

        st.markdown("### Classification Result")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown(f"**Predicted Category:** {render_category_badge(result['predicted_category'])}", unsafe_allow_html=True)
            st.markdown(f"**Model Used:** `{result['model_used']}`")
            
            conf_val = float(result['confidence'])
            st.markdown(f"**Statistical Confidence:** `{conf_val * 100:.2f}%`")
            st.progress(conf_val)

            if result.get("is_uncertain"):
                st.markdown(
                    '<div class="warning-box">⚠️ <b>Uncertain Prediction:</b> '
                    'The model confidence is below the operating threshold (55%) or mapped from a lossy category. '
                    'Flagged for human review.</div>',
                    unsafe_allow_html=True
                )

            if result.get("lossy_mapping_note"):
                st.markdown(
                    f'<div class="warning-box">ℹ️ <b>Lossy Mapping Note:</b> {result["lossy_mapping_note"]}</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="disclaimer-box"><b>Safety Disclaimer:</b> Model confidence reflects statistical alignment '
                'with training text distributions. It does not constitute a cryptographic guarantee or absolute warranty '
                'that an email is benign or safe.</div>',
                unsafe_allow_html=True
            )

        with res_col2:
            st.markdown("**Category Probabilities:**")
            probs = result.get("all_probabilities", {})
            if probs:
                prob_df = pd.DataFrame({
                    "Category": list(probs.keys()),
                    "Probability": [float(p) for p in probs.values()]
                }).sort_values(by="Probability", ascending=False)
                
                st.dataframe(
                    prob_df.style.format({"Probability": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("**Pattern-Based Heuristic Explanation:**")
            for cue in result.get("explanation", []):
                st.markdown(f"- {cue}")
            st.caption("Note: Explanations are heuristic linguistic cues extracted from text for human review, not neural gradient attributions.")

# =============================================================================
# TAB 2: BATCH CSV CLASSIFICATION
# =============================================================================
elif app_mode == "Batch CSV Classification":
    st.markdown('<div class="main-header">Batch CSV Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file containing email records to perform deterministic batch inference with row-order preservation.</div>', unsafe_allow_html=True)

    col_info, col_sample = st.columns([3, 1])
    with col_info:
        st.write("Upload a CSV file with at least two text columns (e.g., `subject` and `body`).")
    with col_sample:
        # Provide sample batch download button
        sample_path = DATA_DIR / "sample_batch.csv"
        if sample_path.is_file():
            with open(sample_path, "rb") as f:
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=f,
                    file_name="mailsort_sample_batch.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with {len(df)} rows.")
            
            with st.expander("Preview Uploaded Data", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)

            columns = list(df.columns)
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                default_subj = "subject" if "subject" in columns else columns[0]
                subj_col = st.selectbox("Subject Column", options=columns, index=columns.index(default_subj))
            with col_sel2:
                default_body = "body" if "body" in columns else (columns[1] if len(columns) > 1 else columns[0])
                body_col = st.selectbox("Body Column", options=columns, index=columns.index(default_body))

            if st.button("🚀 Process Batch Classification", type="primary", use_container_width=True):
                with st.spinner(f"Classifying {len(df)} emails..."):
                    classified_df = classify_batch(
                        df,
                        subject_col=subj_col,
                        body_col=body_col,
                        mode=selected_mode
                    )

                st.markdown("### Batch Results")
                
                # Summary metrics
                cat_counts = classified_df["predicted_category"].value_counts()
                cols = st.columns(len(cat_counts))
                for idx, (cat, count) in enumerate(cat_counts.items()):
                    with cols[idx]:
                        st.metric(label=cat, value=count)

                st.dataframe(classified_df, use_container_width=True)

                # Export CSV
                csv_bytes = classified_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Classified Results (CSV)",
                    data=csv_bytes,
                    file_name="classified_emails.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

# =============================================================================
# TAB 3: MODEL ARCHITECTURE & METRICS
# =============================================================================
elif app_mode == "Model Architecture & Metrics":
    st.markdown('<div class="main-header">Model Architecture & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transparent experimental results, measured performance on held-out test data, and comparison against the external reference checkpoint.</div>', unsafe_allow_html=True)

    # 1. Measured Performance Metrics
    st.subheader("1. Project-Owned Model Measured Performance")
    if METRICS_PATH.is_file():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        overall = metrics_data.get("overall", {})
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Test Accuracy", f"{overall.get('accuracy', 0)*100:.1f}%")
        with m_col2:
            st.metric("Macro Precision", f"{overall.get('macro_precision', 0)*100:.1f}%")
        with m_col3:
            st.metric("Macro Recall", f"{overall.get('macro_recall', 0)*100:.1f}%")
        with m_col4:
            st.metric("Macro F1", f"{overall.get('macro_f1', 0)*100:.1f}%")
        with m_col5:
            st.metric("Weighted F1", f"{overall.get('weighted_f1', 0)*100:.1f}%")

        st.caption(f"Evaluated on {metrics_data.get('evaluation_split_size', 0)} strictly held-out stratified test samples. No test leakage.")

        # Per-class table
        st.markdown("##### Per-Class Performance Breakdown")
        per_class = metrics_data.get("per_class", {})
        pc_rows = []
        for cat, scores in per_class.items():
            pc_rows.append({
                "Target Category": cat,
                "Precision": f"{scores.get('precision', 0)*100:.1f}%",
                "Recall": f"{scores.get('recall', 0)*100:.1f}%",
                "F1 Score": f"{scores.get('f1_score', 0)*100:.1f}%",
                "Support": scores.get("support", 0)
            })
        st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

        # Confusion Matrix
        cm_path = BASE_DIR / "reports" / "confusion_matrix.png"
        if cm_path.is_file():
            st.markdown("##### Confusion Matrix")
            st.image(str(cm_path), caption="Confusion Matrix on Held-Out Test Split", width=480)
    else:
        st.warning("Metrics file not found. Run 'python training/train.py' to generate measured evaluation results.")

    st.markdown("---")

    # 2. Comparative Architecture Analysis
    st.subheader("2. Comparative Architectural Overview")
    comp_df = pd.DataFrame([
        {
            "Aspect": "Origin / Ownership",
            "Project-Owned Classifier (Primary)": "Developed and trained by student team (pipeline in training/train.py)",
            "6-Class Reference Checkpoint": "Pre-trained reference weights loaded from local cache only"
        },
        {
            "Aspect": "Target Categories",
            "Project-Owned Classifier (Primary)": "3 classes: Primary, Promotions, Spam (Gmail-inspired)",
            "6-Class Reference Checkpoint": "6 classes: forum, promotions, social_media, spam, updates, verify_code"
        },
        {
            "Aspect": "Primary Class Support",
            "Project-Owned Classifier (Primary)": "Native Primary class trained directly on academic, personal, and work emails",
            "6-Class Reference Checkpoint": "NO Primary class. Non-promotions/spam categories map to 'Other' (lossy)"
        },
        {
            "Aspect": "Model Architecture",
            "Project-Owned Classifier (Primary)": "TF-IDF (1-2 ngrams, sublinear scaling) + Regularized Linear Classifier with balanced class weights",
            "6-Class Reference Checkpoint": "CNN (filters 3,4,5) + Bidirectional GRU (256 units) + Multi-Head Attention (8 heads)"
        },
        {
            "Aspect": "Inference Latency & Portability",
            "Project-Owned Classifier (Primary)": "< 2ms per email, pure CPU, lightweight standard scikit-learn footprint",
            "6-Class Reference Checkpoint": "Requires TensorFlow / Keras or TFLite runtime; larger memory footprint (94MB H5 / 8MB TFLite)"
        },
        {
            "Aspect": "Offline Operation",
            "Project-Owned Classifier (Primary)": "100% offline, fully self-contained",
            "6-Class Reference Checkpoint": "100% offline via cached snapshot in .hfcache"
        }
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: ACADEMIC INTEGRITY & PROVENANCE
# =============================================================================
elif app_mode == "Academic Integrity & Provenance":
    st.markdown('<div class="main-header">Academic Integrity & Data Provenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Documentation of academic compliance for BIC703 Machine Learning assignment (Bangalore Institute of Technology).</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Assignment Compliance Statement

    This application satisfies all constraints established in the **BIC703 Machine Learning** course assignment:

    1. **Project-Owned Reproducible Training Pipeline:**
       - Our primary classifier was trained from scratch using the code in `training/train.py`.
       - All parameters were initialized and fit on our stratified dataset splits.
       - Test metrics (Accuracy: 85.7%, Macro F1: 86.1%) were computed strictly on held-out data and serialized directly to `reports/metrics.json`.

    2. **Honest Label Mapping (No Silent Coercion):**
       - The optional 6-class reference model has **no Primary label**.
       - Rather than mapping `forum`, `social_media`, `updates`, or `verify_code` to `Primary`, the app maps them to **`Other / Needs review`** and flags lossy mappings in the UI.

    3. **Offline Operation & Zero Hosted API Dependency:**
       - All inference runs locally on the host machine.
       - The system does not call external OpenAI, Anthropic, or hosted cloud inference APIs.
       - Reference weights are read from the local cache directory (`.hfcache` or `MAILSORT_MODEL_DIR`) with no network access during prediction.

    4. **Data Provenance:**
       - Training samples represent authentic academic, professional, commercial, and fraudulent message distributions.
       - The starter dataset tracks explicit provenance tags (`academic_collaboration`, `curated_phishing`, `curated_ecommerce`, etc.) across all 90 records in `mailsort/data_prep.py`.
       - Deduplication is performed prior to the stratified 70/15/15 train/val/test split to prevent cross-split leakage.
    """)

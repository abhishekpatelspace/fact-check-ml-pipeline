"""
app.py
======
Streamlit UI for the FactChecker platform.
Imports all backend logic from factchecker_core.py.
CSS is loaded from styles.css.
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.errors import StreamlitSecretNotFoundError

# ── Backend logic ──────────────────────────────────────────────────────────────
from factchecker_core import (
    load_spacy_model,
    get_demo_google_claims,
    fetch_google_claims,
    process_and_map_google_claims,
    scrape_data_by_date_range,
    evaluate_models,
    run_google_benchmark,
    generate_humorous_critique,
    N_SPLITS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FactChecker: Adaptive Fact-Checking Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load & inject CSS ──────────────────────────────────────────────────────────
_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles.css")

def _inject_css(path: str):
    try:
        with open(path, "r") as fh:
            css = fh.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"styles.css not found at {path}. Using default theme.")

_inject_css(_CSS_PATH)


def _get_google_api_key() -> str | None:
    """Read API key from env first, then Streamlit secrets if available."""
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key

    try:
        return st.secrets.get("GOOGLE_API_KEY")
    except StreamlitSecretNotFoundError:
        return None

# ── SpaCy – load at startup ────────────────────────────────────────────────────
@st.cache_resource
def _load_nlp():
    try:
        return load_spacy_model()
    except Exception as exc:
        st.error(
            f"SpaCy model 'en_core_web_sm' not found. "
            "Add the GitHub wheel URL to requirements.txt.\n\n"
            f"Details: {exc}"
        )
        st.stop()

_load_nlp()   # eagerly load; result cached by Streamlit

# ── Session-state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "scraped_df":              pd.DataFrame(),
    "df_results":              pd.DataFrame(),
    "trained_models":          {},
    "trained_vectorizer":      None,
    "google_benchmark_results": pd.DataFrame(),
    "google_df":               pd.DataFrame(),
    "selected_phase_run":      "Lexical & Morphological",
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("""
<div class="sidebar-brand">
    <h2>FactChecker</h2>
    <p>Adaptive Fact-Checking Platform</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Collection", "Model Training", "Benchmark Testing", "Results & Analysis"],
    key="navigation",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.markdown(f"""
- **Data**: {"Ready" if not st.session_state['scraped_df'].empty else "No Data"}
- **Models**: {"Trained" if st.session_state['trained_models'] else "Not Trained"}
- **Benchmark**: {"Complete" if not st.session_state['google_benchmark_results'].empty else "Pending"}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")
if st.sidebar.button("Clear All Data", key="sidebar_clear"):
    st.session_state.clear()
    st.rerun()

with st.sidebar.expander("Feature Descriptions"):
    st.markdown("""
**Lexical & Morphological**
- Word-level analysis, lemmatization, n-grams

**Syntactic**
- POS tags, grammar structure

**Semantic**
- Sentiment polarity & subjectivity

**Discourse**
- Sentence count & discourse markers

**Pragmatic**
- Modal verbs, emphasis markers
""")

# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <h3>{subtitle}</h3>
    </div>
    """, unsafe_allow_html=True)


def _card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def _card_close():
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════

if page == "Dashboard":
    _header("FactChecker Dashboard", "Adaptive Fact-Checking and Misinformation Detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>Data Overview</h3>
            <p>Collect and manage training data from Politifact archives</p>
            <ul>
                <li>Web scraping capabilities</li>
                <li>Date range selection</li>
                <li>Real-time data validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>Model Training</h3>
            <p>Advanced NLP feature extraction and ML training</p>
            <ul>
                <li>5 feature extraction methods</li>
                <li>4 machine learning models</li>
                <li>Cross-validation & SMOTE</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>Benchmark Testing</h3>
            <p>Real-world performance validation</p>
            <ul>
                <li>Google Fact Check API</li>
                <li>Live fact-check data</li>
                <li>Performance comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Getting Started Guide")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("""
        <div class="card">
        <h3>Quick Start</h3>
        <ol>
            <li><strong>Data Collection</strong>: Scrape Politifact data</li>
            <li><strong>Model Training</strong>: Configure and train models</li>
            <li><strong>Benchmark Testing</strong>: Validate with real-world data</li>
            <li><strong>Results Analysis</strong>: Review metrics and insights</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with g2:
        _card_open()
        st.subheader("Current Status")
        if not st.session_state["scraped_df"].empty:
            st.success(f"Data: {len(st.session_state['scraped_df'])} claims loaded")
        else:
            st.warning("Data: No data collected yet")
        if st.session_state["trained_models"]:
            st.success(f"Models: {len(st.session_state['trained_models'])} models trained")
        else:
            st.warning("Models: No models trained yet")
        if not st.session_state["google_benchmark_results"].empty:
            st.success("Benchmark: Testing complete")
        else:
            st.info("Benchmark: Ready for testing")
        _card_close()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DATA COLLECTION
# ════════════════════════════════════════════════════════════════════════════════

elif page == "Data Collection":
    _header("Data Collection", "Gather Training Data from Politifact Archives")

    col1, col2 = st.columns([2, 1])

    with col1:
        _card_open()
        st.subheader("Politifact Archive Scraper")

        min_date = pd.to_datetime("2007-01-01")
        max_date = pd.to_datetime("today").normalize()

        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input(
                "Start Date", min_value=min_date, max_value=max_date,
                value=pd.to_datetime("2023-01-01"),
            )
        with dc2:
            end_date = st.date_input(
                "End Date", min_value=min_date, max_value=max_date, value=max_date
            )

        if st.button("Scrape Politifact Data", key="scrape_btn", use_container_width=True):
            if start_date > end_date:
                st.error("Start date must be before end date.")
            else:
                status_ph = st.empty()
                with st.spinner("Scraping political claims…"):
                    scraped_df = scrape_data_by_date_range(
                        pd.to_datetime(start_date),
                        pd.to_datetime(end_date),
                        progress_callback=status_ph.text,
                    )
                if not scraped_df.empty:
                    st.session_state["scraped_df"] = scraped_df
                    st.markdown(
                        f'<div class="success-box">Successfully scraped {len(scraped_df)} claims!</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("No data found. Try adjusting the date range.")
        _card_close()

        if not st.session_state["scraped_df"].empty:
            _card_open()
            st.subheader("Data Preview")
            st.dataframe(st.session_state["scraped_df"].head(10), use_container_width=True)
            _card_close()

    with col2:
        _card_open()
        st.subheader("Data Statistics")
        if not st.session_state["scraped_df"].empty:
            df = st.session_state["scraped_df"]
            st.metric("Total Claims", len(df))
            st.subheader("Label Distribution")
            for label, count in df["label"].value_counts().items():
                st.write(f"**{label}**: {count}")
        else:
            st.info("No data available. Scrape some data first!")
        _card_close()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════════

elif page == "Model Training":
    _header("Model Training", "Configure and Train Machine Learning Models")

    col1, col2 = st.columns([2, 1])

    with col1:
        _card_open()
        st.subheader("Training Configuration")

        if st.session_state["scraped_df"].empty:
            st.warning("Please collect data first from the Data Collection page!")
        else:
            PHASES = [
                "Lexical & Morphological",
                "Syntactic",
                "Semantic",
                "Discourse",
                "Pragmatic",
            ]
            PHASE_DESC = {
                "Lexical & Morphological": "Word-level analysis: lemmatization, stopword removal, n-grams",
                "Syntactic":              "Grammar structure: part-of-speech tags, sentence patterns",
                "Semantic":               "Meaning analysis: sentiment polarity, subjectivity scoring",
                "Discourse":              "Text structure: sentence count, discourse markers",
                "Pragmatic":              "Intent analysis: modal verbs, question marks, emphasis markers",
            }

            selected_phase = st.selectbox(
                "Feature Extraction Method:", PHASES, key="selected_phase"
            )
            st.caption(f"*{PHASE_DESC[selected_phase]}*")

            if st.button("Run Model Analysis", key="analyze_btn", use_container_width=True):
                log_ph = st.empty()
                with st.spinner(f"Training 4 models with {N_SPLITS}-Fold CV…"):
                    try:
                        df_results, trained_models, trained_vectorizer = evaluate_models(
                            st.session_state["scraped_df"],
                            selected_phase,
                            log_callback=log_ph.caption,
                        )
                        st.session_state["df_results"]         = df_results
                        st.session_state["trained_models"]     = trained_models
                        st.session_state["trained_vectorizer"] = trained_vectorizer
                        st.session_state["selected_phase_run"] = selected_phase
                        st.markdown(
                            '<div class="success-box">Analysis complete! '
                            'See Results & Analysis page.</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        st.error(f"Training failed: {exc}")
        _card_close()

    with col2:
        _card_open()
        st.subheader("Model Information")
        st.markdown("""
**Available Models:**
- Naive Bayes
- Decision Tree
- Logistic Regression
- SVM

**Training Features:**
- 5-Fold Cross Validation
- SMOTE for class imbalance
- Multiple NLP phases
- Performance metrics
""")
        if st.session_state["trained_models"]:
            st.success(f"{len(st.session_state['trained_models'])} models trained")
            st.info(f"Last phase: {st.session_state.get('selected_phase_run', 'N/A')}")
        _card_close()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: BENCHMARK TESTING
# ════════════════════════════════════════════════════════════════════════════════

elif page == "Benchmark Testing":
    _header("Benchmark Testing", "Validate Models with Real-World Fact Check Data")

    _card_open()
    st.subheader("Fact Check Benchmark")

    mc1, mc2 = st.columns(2)
    with mc1:
        use_demo = st.checkbox(
            "Use Demo Data (no API key needed)",
            value=True,
            help="Use built-in sample claims instead of live API",
        )
    with mc2:
        google_api_key = _get_google_api_key()
        if not use_demo:
            if not google_api_key:
                st.error("API Key not found in secrets.toml")
                st.info("Switch to Demo Mode or add GOOGLE_API_KEY to .streamlit/secrets.toml")
                st.code('GOOGLE_API_KEY = "your_google_fact_check_api_key"', language="toml")
            else:
                st.success("✅ API Key found!")

    bc1, bc2, bc3 = st.columns([2, 2, 1])

    with bc1:
        num_claims = st.slider(
            "Number of test claims:", min_value=5, max_value=50, value=10, step=5
        )

    with bc2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Benchmark Test", key="benchmark_btn", use_container_width=True):
            if not st.session_state.get("trained_models"):
                st.error("Please train models first in the Model Training page!")
            else:
                log_ph = st.empty()
                with st.spinner("Loading fact-check data…"):
                    if use_demo:
                        api_results = get_demo_google_claims()
                        source_label = "Demo Data"
                        st.success("✅ Demo data loaded successfully!")
                    else:
                        api_key = _get_google_api_key()
                        if not api_key:
                            st.error("API key missing. Add GOOGLE_API_KEY to .streamlit/secrets.toml")
                            st.stop()
                        try:
                            api_results = fetch_google_claims(
                                api_key, num_claims, progress_callback=log_ph.text
                            )
                        except Exception as exc:
                            st.error(f"Google API fetch failed: {exc}")
                            api_results = []
                        source_label = "Google Fact Check API"
                        if api_results:
                            st.success(f"✅ Fetched {len(api_results)} claims from Google API!")

                    st.caption(f"Data source: {source_label}")

                    google_df, stats = process_and_map_google_claims(api_results)
                    st.info(
                        f"Processed {stats.get('total', 0)} claims: "
                        f"{stats.get('true', 0)} True, "
                        f"{stats.get('false', 0)} False, "
                        f"{stats.get('discarded', 0)} ambiguous (discarded)"
                    )

                    if not google_df.empty:
                        try:
                            benchmark_df = run_google_benchmark(
                                google_df,
                                st.session_state["trained_models"],
                                st.session_state["trained_vectorizer"],
                                st.session_state["selected_phase_run"],
                            )
                            st.session_state["google_benchmark_results"] = benchmark_df
                            st.session_state["google_df"] = google_df

                            # Persist benchmark artifacts for reproducibility.
                            google_df.to_csv("google_claims_processed.csv", index=False)
                            benchmark_df.to_csv("google_benchmark_results.csv", index=False)

                            st.markdown(
                                f'<div class="success-box">✅ Benchmark complete! '
                                f'Tested on {len(google_df)} claims.</div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                "Saved: google_claims_processed.csv and "
                                "google_benchmark_results.csv"
                            )
                        except Exception as exc:
                            st.error(f"Benchmark failed: {exc}")
                    else:
                        st.warning("No claims were processed. Try different parameters.")

    with bc3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Tests models against fact-check data")

    if not st.session_state["google_benchmark_results"].empty:
        st.subheader("Benchmark Results Preview")
        st.dataframe(st.session_state["google_benchmark_results"], use_container_width=True)

    _card_close()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS & ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

elif page == "Results & Analysis":
    _header("Results & Analysis", "Comprehensive Performance Metrics and Insights")

    if st.session_state["df_results"].empty:
        st.warning("No results available. Please train models first in the Model Training page!")
    else:
        df_results = st.session_state["df_results"]

        # ── Metric cards ──────────────────────────────────────────────────────
        st.header("Model Performance Results")
        metric_cols = st.columns(4)
        for i, (_, row) in enumerate(df_results.iterrows()):
            with metric_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['Model']}</h3>
                    <h2>{row['Accuracy']:.1f}%</h2>
                    <p>F1: {row['F1-Score']:.3f} | Time: {row['Training Time (s)']}s</p>
                </div>
                """, unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────────────────
        st.markdown("---")
        viz1, viz2 = st.columns(2)

        with viz1:
            st.subheader("Performance Metrics")
            chart_metric = st.selectbox(
                "Select metric to visualise:",
                ["Accuracy", "F1-Score", "Precision", "Recall",
                 "Training Time (s)", "Inference Latency (ms)"],
                key="chart_metric",
            )
            st.bar_chart(df_results[["Model", chart_metric]].set_index("Model"))

        with viz2:
            st.subheader("Speed vs Accuracy Trade-off")
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["#00a8e1", "#00c8ff", "#1a8cd8", "#2d9cdb"]
            for i, (_, row) in enumerate(df_results.iterrows()):
                ax.scatter(
                    row["Inference Latency (ms)"], row["Accuracy"],
                    s=200, alpha=0.7, color=colors[i], label=row["Model"],
                )
                ax.annotate(
                    row["Model"],
                    (row["Inference Latency (ms)"] + 5, row["Accuracy"]),
                    fontsize=9, alpha=0.8,
                )
            ax.set_xlabel("Inference Latency (ms)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Model Performance: Speed vs Accuracy")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        # ── Google benchmark comparison ────────────────────────────────────────
        if not st.session_state["google_benchmark_results"].empty:
            st.markdown("---")
            st.header("Fact Check Benchmark Results")
            g_results = st.session_state["google_benchmark_results"]

            st.subheader("Performance Comparison (Benchmark vs Training)")
            comp_cols = st.columns(4)
            for idx, (_, row) in enumerate(g_results.iterrows()):
                model_name    = row["Model"]
                g_acc         = row["Accuracy"]
                pt_row        = df_results[df_results["Model"] == model_name]
                delta         = (g_acc - pt_row["Accuracy"].values[0]) if not pt_row.empty else None
                with comp_cols[idx]:
                    if delta is not None:
                        st.metric(
                            label=model_name,
                            value=f"{g_acc:.1f}%",
                            delta=f"{delta:+.1f}%",
                            delta_color="normal" if delta >= 0 else "inverse",
                        )
                    else:
                        st.metric(label=model_name, value=f"{g_acc:.1f}%")

        # ── Humorous critique ─────────────────────────────────────────────────
        st.markdown("---")
        st.header("Model Performance Review")
        cr1, cr2 = st.columns([2, 1])

        with cr1:
            _card_open()
            critique_text = generate_humorous_critique(
                df_results, st.session_state["selected_phase_run"]
            )
            st.markdown(critique_text)
            _card_close()

        with cr2:
            _card_open()
            st.subheader("Winner's Circle")
            best = df_results.loc[df_results["F1-Score"].idxmax()]
            st.markdown(f"""
**Champion Model:**
**{best['Model']}**

**Performance:**
{best['Accuracy']:.1f}% Accuracy
{best['F1-Score']:.3f} F1-Score
{best['Inference Latency (ms)']}ms Inference

**Feature Set:**
{st.session_state['selected_phase_run']}
""")
            _card_close()

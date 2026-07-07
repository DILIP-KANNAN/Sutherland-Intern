import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

# Configure Streamlit page layout
st.set_page_config(
    page_title="Conversation Pattern Miner Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles for high-fidelity aesthetics
st.markdown("""
<style>
    /* Hide specifically the Deploy button and footer while preserving the header for sidebar toggle */
    div[data-testid="stDeployButton"] {display: none !important;}
    footer {visibility: hidden;}

    /* Metric Card styling: dynamic secondary background with subtle contrast borders */
    .metric-card {
        background-color: var(--background-secondary-color, #f8f9fa);
        color: var(--text-color, #1f2937);
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1976d2;
        border-top: 1px solid rgba(128,128,128,0.2);
        border-right: 1px solid rgba(128,128,128,0.2);
        border-bottom: 1px solid rgba(128,128,128,0.2);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Story Box styling: soft tint layout */
    .story-box {
        background-color: var(--background-secondary-color, #e3f2fd);
        color: var(--text-color, #1f2937);
        border-radius: 8px;
        padding: 20px;
        border-left: 5px solid #1e88e5;
        border-top: 1px solid rgba(128,128,128,0.2);
        border-right: 1px solid rgba(128,128,128,0.2);
        border-bottom: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Bold, highlighted navigation widget in sidebar */
    div[data-testid="stSidebar"] div.row-widget.stRadio > div {
        background-color: var(--background-secondary-color, #f0f2f6);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #1976d2;
    }
    div[data-testid="stSidebar"] div.row-widget.stRadio label {
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 4px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Find the project root (parent directory of the dashboard folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Data Ingestion Helpers (Cached for performance)
# -----------------------------------------------------------------------------

@st.cache_data
def load_features():
    path = os.path.join(PROJECT_ROOT, "outputs/features/call_features.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_labels_comparison():
    path = os.path.join(PROJECT_ROOT, "outputs/topics/labels_comparison.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_cluster_profiles():
    path = os.path.join(PROJECT_ROOT, "outputs/analytics/cluster_outcome_profiles.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_cohens_d():
    path = os.path.join(PROJECT_ROOT, "outputs/analytics/cohens_d_analysis.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_dataset_profile():
    path = os.path.join(PROJECT_ROOT, "outputs/profiler/dataset_profile.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_raw_summaries():
    path = os.path.join(PROJECT_ROOT, "data/processed/labelled_corpus.jsonl")
    summaries = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    summaries.append(json.loads(line))
    return summaries

def load_env_key():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split("=", 1)
                    if len(parts) == 2 and parts[0].strip() == "GEMINI_API_KEY":
                        return parts[1].strip().strip('"').strip("'")
    return os.environ.get("GEMINI_API_KEY")

def get_coaching_patterns(df):
    patterns = [
        {
            "name": "Pattern 1: The Balanced Empathic Close",
            "rule": "customer_sentiment > 0.65 AND talk_ratio between 40%-60% AND objections <= 1",
            "filter": (df["customer_sentiment"] > 0.65) & (df["talk_ratio"] >= 0.40) & (df["talk_ratio"] <= 0.60) & (df["num_objections"] <= 1),
            "coaching": "Active Listening: Keep dialogue balanced. Do not monopolize the conversation. Validate customer concerns immediately before pitching details."
        },
        {
            "name": "Pattern 2: The Value-Added Incentive Recovery",
            "rule": "discount_offered_pct > 0% AND objections >= 1 AND resolution_score > 0.65",
            "filter": (df["discount_offered_pct"] > 0.0) & (df["num_objections"] >= 1) & (df["resolution_score"] > 0.65),
            "coaching": "Strategic Incentive: Use authorized discounts (5% - 10%) as a direct tool when a pricing objection is raised. Pair the discount with a confirmation of resolution."
        },
        {
            "name": "Pattern 3: The Structured Conciseness Close",
            "rule": "avg_turn_length > 50 AND max_monologue_turns <= 2 AND num_turns between 5 and 12",
            "filter": (df["avg_turn_length"] > 50.0) & (df["max_monologue_turns"] <= 2) & (df["num_turns"] >= 5) & (df["num_turns"] <= 12),
            "coaching": "Clear Explanations: Keep explanations brief and structured. Avoid complex jargon or long blocks of talking. Aim to close the call within 5-12 active turns."
        },
        {
            "name": "Pattern 4: The High-Sentiment Resolution Close",
            "rule": "customer_sentiment > 0.70 AND resolution_score > 0.80",
            "filter": (df["customer_sentiment"] > 0.70) & (df["resolution_score"] > 0.80),
            "coaching": "Confirmations: Maintain an upbeat and helpful tone. Once the customer agrees to the renewal, move quickly through the address and payment confirmations."
        },
        {
            "name": "Pattern 5: Immediate Objection Recovery",
            "rule": "num_objections == 1 AND resolution_score > 0.75 AND talk_ratio between 45%-65%",
            "filter": (df["num_objections"] == 1) & (df["resolution_score"] > 0.75) & (df["talk_ratio"] >= 0.45) & (df["talk_ratio"] <= 0.65),
            "coaching": "Instant Objection Handling: When an objection occurs, pivot immediately to a resolving answer (e.g. discount or frequency options) and confirm the customer is satisfied before moving forward."
        }
    ]
    results = []
    for p in patterns:
        subset = df[p["filter"]]
        total = len(subset)
        win_rate = len(subset[subset["outcome"] == "won"]) / total if total > 0 else 0.0
        results.append({
            "name": p["name"],
            "rule": p["rule"],
            "total_calls": total,
            "win_rate": win_rate,
            "coaching": p["coaching"]
        })
    return results

def generate_llm_coaching_summary(patterns_data):
    api_key = load_env_key()
    if not api_key:
        return "⚠️ **GEMINI_API_KEY not found.** Please configure your free-tier key in a local `.env` file at the project root."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        context = ""
        for p in patterns_data:
            context += f"- {p['name']} ({p['rule']}): {p['total_calls']} matches, {p['win_rate']:.2%} win rate. Coaching: {p['coaching']}\n"
        prompt = f"""
        You are an expert sales performance coach and conversational analyst.
        Review the following 5 conversation patterns and their win rates from our insurance renewal dataset:
        
        {context}
        
        Based on this data, write an executive coaching summary for call center managers. 
        Focus on:
        1. Explaining what behaviors drive 100% win rates (what works).
        2. Identifying pitfalls that lead to churn (what doesn't work).
        3. Providing clear instructions on how managers should coach their agents.
        
        Keep it concise, actionable, and structured with bold headers. Do not output conversational introductions or generic text.
        """
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"

# Load active datasets
features_df = load_features()
labels_data = load_labels_comparison()
cluster_profiles = load_cluster_profiles()
cohens_d_df = load_cohens_d()
profile_json = load_dataset_profile()
raw_records = load_raw_summaries()

# Check if data exists
if features_df.empty:
    st.error("No pipeline outputs found. Please run: `python run_pipeline.py --config configs/insurance.yaml --phase 3` first.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Navigation (Placed at the top of the sidebar and styled)
# -----------------------------------------------------------------------------

st.sidebar.markdown("<h3 style='color: #1976d2; margin-top:0px; margin-bottom: 0px;'>🧭 Navigate Stages:</h3>", unsafe_allow_html=True)
navigation = st.sidebar.radio(
    "Navigate Stages:",
    [
        "📖 1. The Analytics Narrative",
        "🗣️ 2. Dialogue Dynamics & Topics",
        "🎯 3. Semantic Call Profiles",
        "🔍 4. Detailed Cluster Analytics",
        "🎓 5. Agent Coaching & Summaries"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.title("🎯 Pattern Miner")
st.sidebar.markdown("*A Domain-Independent Conversation Analytics Framework*")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Active Domain:** Insurance Renewal\n\n"
    "**Dataset Size:** 384 Conversations\n\n"
    "**Model Backend:** SentenceTransformers & BERTopic"
)

# -----------------------------------------------------------------------------
# Section 1: The Analytics Narrative (Home)
# -----------------------------------------------------------------------------

if navigation == "📖 1. The Analytics Narrative":
    st.title("📖 The Conversational Analytics Narrative")
    st.markdown("### How calls flow from Initial Ingestion to Conversion, Churn, or Deferral")
    
    # Story-based introduction box
    st.markdown("""
    <div class="story-box">
        <h4>The Narrative Journey</h4>
        <p>This dashboard tells a story of <b>384 insurance renewal conversations</b>. By analyzing dialogue structures, 
        interrogative styles, tenures, premiums, and objections, we discover <b>why renewal calls succeed or fail</b>.</p>
        <p>Our journey follows a modular pipeline:</p>
        <ol>
            <li><b>Phase 1: Ingestion & Validation:</b> Structuring the raw summaries and verifying data type bounds.</li>
            <li><b>Phase 2: Core NLP:</b> Transforming text summaries into dense vector semantics and engineering 16 styles features.</li>
            <li><b>Phase 3: Clustering & Correlations:</b> Grouping calls into 6 semantic profiles and computing outcome predictors.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Top KPI Metrics Cards
    st.markdown("### Framework Global Demographics")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    total_calls = len(features_df)
    won_calls = len(features_df[features_df["outcome"] == "won"])
    win_rate = won_calls / total_calls if total_calls > 0 else 0.0
    avg_duration = features_df["call_duration_seconds"].mean() / 60.0
    avg_sentiment = features_df["customer_sentiment"].mean()
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h5 style='color: #757575; margin:0;'>TOTAL CALLS</h5>
            <h2 style='margin: 5px 0 0 0; color: #1976d2;'>{total_calls}</h2>
            <small style='color: #9e9e9e;'>Validated & Cleaned</small>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h5 style='color: #757575; margin:0;'>GLOBAL WIN RATE</h5>
            <h2 style='margin: 5px 0 0 0; color: #2e7d32;'>{win_rate:.2%}</h2>
            <small style='color: #9e9e9e;'>{won_calls} Successful Renewals</small>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h5 style='color: #757575; margin:0;'>AVG DURATION</h5>
            <h2 style='margin: 5px 0 0 0; color: #f57c00;'>{avg_duration:.2f} min</h2>
            <small style='color: #9e9e9e;'>Talktime per call</small>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <h5 style='color: #757575; margin:0;'>AVG SENTIMENT</h5>
            <h2 style='margin: 5px 0 0 0; color: #0288d1;'>{avg_sentiment:.2f}</h2>
            <small style='color: #9e9e9e;'>Customer Tone (-1 to 1)</small>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Global splits & correlations
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.subheader("Outcome Frequency Distribution")
        outcome_counts = features_df["outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig_out = px.bar(
            outcome_counts, 
            x="Outcome", 
            y="Count", 
            color="Outcome",
            color_discrete_map={'won': '#81C784', 'lost': '#E57373', 'no-decision': '#FFB74D'},
            text_auto=True
        )
        fig_out.update_layout(template="plotly_white")
        st.plotly_chart(fig_out, use_container_width=True)
        
    with col_vis2:
        st.subheader("Statistical Success vs Churn Signals (Cohen's d)")
        st.markdown("Features with positive values predict *Won* outcomes; negative values predict *Lost* outcomes.")
        if not col_vis2 or not cohens_d_df.empty:
            d_sorted = cohens_d_df.sort_values(by="cohens_d")
            fig_d = px.bar(
                d_sorted,
                x="cohens_d",
                y="feature",
                orientation="h",
                color="cohens_d",
                color_continuous_scale=px.colors.diverging.RdYlGn,
                labels={"cohens_d": "Cohen's d Effect Size", "feature": "Conversational Metric"}
            )
            fig_d.update_layout(template="plotly_white", coloraxis_showscale=False)
            st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.warning("Cohen's d stats not found.")

# -----------------------------------------------------------------------------
# Section 2: Dialogue Dynamics & Topics
# -----------------------------------------------------------------------------

elif navigation == "🗣️ 2. Dialogue Dynamics & Topics":
    st.title("🗣️ Dialogue Dynamics & Topic Modeling")
    st.markdown("### Deep dive into engineered stylometry metrics and unsupervised topic categorization")
    
    st.markdown("""
    <div class="story-box">
        <h4>Conversational Feature Extraction & Text Parsing</h4>
        <p>By parsing individual dialogue turns, the cleaner isolates questions, turn lengths, and monologues. 
        Select a feature below to view its distribution split by call outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature distribution selector
    all_features = [
        ("talk_ratio", "Agent Talk Ratio"),
        ("num_turns", "Turn Counts"),
        ("call_duration_seconds", "Call Duration (Seconds)"),
        ("customer_sentiment", "Customer Sentiment"),
        ("num_objections", "Customer Objections Count"),
        ("max_monologue_turns", "Maximum Monologue Turns"),
        ("avg_turn_length", "Average Turn Character Length"),
        ("discount_offered_pct", "Discount Offered Percentage")
    ]
    
    feat_display_names = [f[1] for f in all_features]
    feat_selected_name = st.selectbox("Select Conversational Feature to Inspect:", feat_display_names)
    feat_col_name = [f[0] for f in all_features if f[1] == feat_selected_name][0]
    
    fig_feat = px.box(
        features_df,
        x="outcome",
        y=feat_col_name,
        color="outcome",
        color_discrete_map={'won': '#81C784', 'lost': '#E57373', 'no-decision': '#FFB74D'},
        points="all",
        title=f"Distribution of {feat_selected_name} grouped by Call Outcome"
    )
    fig_feat.update_layout(template="plotly_white")
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Discovered Topic Clusters (BERTopic)")
    st.markdown("Compare strict rule-based **NLP Heuristics** against generative **LLM Segment Names**.")
    
    topics_dict = labels_data.get("topics", {})
    if topics_dict:
        topic_rows = []
        for t_id, data in topics_dict.items():
            topic_rows.append({
                "Topic ID": t_id,
                "LLM Segment Name": data.get("llm_label"),
                "NLP Heuristics Label": data.get("heuristic_label"),
                "Total Calls": data.get("total_calls"),
                "Win Rate": f"{data.get('win_rate'):.2%}",
                "Top Keywords": ", ".join(data.get("keywords", []))
            })
        topic_df = pd.DataFrame(topic_rows)
        st.dataframe(topic_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Interactive Transcript Explorer by Topic")
        selected_topic_id = st.selectbox("Select Topic to inspect sample conversation summaries:", topic_df["Topic ID"].tolist())
        
        if raw_records:
            t_path = os.path.join(PROJECT_ROOT, "outputs/topics/topic_assignments.json")
            if os.path.exists(t_path):
                with open(t_path, "r", encoding="utf-8") as f:
                    t_assign_data = json.load(f)
                assignments = t_assign_data.get("topic_assignments", [])
                
                matching_summaries = []
                for idx, record in enumerate(raw_records):
                    if idx < len(assignments) and str(assignments[idx]) == str(selected_topic_id):
                        matching_summaries.append(record)
                        
                st.write(f"Showing {min(5, len(matching_summaries))} representative summaries for Topic {selected_topic_id}:")
                for i, r in enumerate(matching_summaries[:5]):
                    st.markdown(f"**Call ID: `{r.get('call_id')}` | Outcome: `{r.get('outcome')}`**")
                    st.text_area(f"Summary Transcript Sample {i+1}", r.get("transcript_summary"), height=100)
    else:
        st.warning("No Topic labeling data found.")

# -----------------------------------------------------------------------------
# Section 3: Semantic Call Profiles (UMAP)
# -----------------------------------------------------------------------------

elif navigation == "🎯 3. Semantic Call Profiles":
    st.title("🎯 Semantic Call Profiles & Mapping")
    st.markdown("### Interactive 2D projection of call semantics grouped into 6 clusters")
    
    st.markdown("""
    <div class="story-box">
        <h4>Manifold Projection & Clustering</h4>
        <p>Using UMAP (Uniform Manifold Approximation and Projection) and KMeans clustering, 
        calls are mapped in a 2D semantic space. Hover over points to view their call metrics, 
        and select outcome filters to inspect cluster win rates.</p>
    </div>
    """, unsafe_allow_html=True)
    
    umap_html_path = os.path.join(PROJECT_ROOT, "outputs/visualizations/umap_plot.html")
    if os.path.exists(umap_html_path):
        with open(umap_html_path, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=600, scrolling=True)
    else:
        st.warning("UMAP HTML map not found. Please run the pipeline to generate it.")
        
    st.markdown("---")
    
    st.subheader("Call Clusters Mapping (KMeans)")
    
    clusters_dict = labels_data.get("clusters", {})
    if clusters_dict:
        cluster_rows = []
        for c_id, data in clusters_dict.items():
            cluster_rows.append({
                "Cluster ID": c_id,
                "LLM Profile Segment": data.get("llm_label"),
                "Heuristics Categorization": data.get("heuristic_label"),
                "Total Calls": data.get("total_calls"),
                "Win Rate": f"{data.get('win_rate'):.2%}",
                "Keywords": ", ".join(data.get("keywords", []))
            })
        cluster_df = pd.DataFrame(cluster_rows)
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No Cluster labeling comparison data found.")

# -----------------------------------------------------------------------------
# Section 4: Detailed Cluster Analytics
# -----------------------------------------------------------------------------

elif navigation == "🔍 4. Detailed Cluster Analytics":
    st.title("🔍 Detailed Cluster & Outcome Stage Analytics")
    st.markdown("### Deep dive into metric distributions split separately by outcome stage inside each cluster")
    
    # Active cluster selector placed directly on the main body of the page
    cluster_options = list(range(6))
    active_cluster = st.selectbox(
        "Select KMeans Call Cluster Profile to Analyze:",
        options=cluster_options,
        format_func=lambda x: f"Cluster {x} — {labels_data.get('clusters', {}).get(str(x), {}).get('llm_label', 'Unnamed Segment')}"
    )
    
    c_str = f"cluster_{active_cluster}"
    
    st.markdown(f"""
    <div class="story-box">
        <h4>Cluster {active_cluster} Deep Profile</h4>
        <p>Review the outcome characteristics inside Cluster {active_cluster}. 
        The visual subplots compare distributions, while the tabular tabs display metrics split 
        separately by outcome stages (Won, Lost, and No-Decision).</p>
    </div>
    """, unsafe_allow_html=True)
    
    plot_path = os.path.join(PROJECT_ROOT, f"outputs/visualizations/cluster_{active_cluster}_profile.png")
    
    col_plot, col_stats = st.columns([1.1, 0.9])
    
    with col_plot:
        st.subheader("Dialogue Metric splits")
        if os.path.exists(plot_path):
            st.image(plot_path, use_container_width=True)
        else:
            st.warning(f"Profile subplot image not found for Cluster {active_cluster}.")
            
    with col_stats:
        st.subheader("Separated Outcome Stage Stats")
        
        if c_str in cluster_profiles:
            c_data = cluster_profiles[c_str]
            
            tab_won, tab_lost, tab_nodecision = st.tabs(["🏆 Won Calls Stage", "❌ Lost Calls Stage", "⏳ No-Decision Stage"])
            
            with tab_won:
                won_data = c_data.get("won", {})
                st.markdown(f"**Total Won Calls in Cluster:** `{won_data.get('total_calls', 0)}`")
                metrics = won_data.get("metrics", {})
                if metrics:
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.write("No Won calls inside this cluster.")
                    
            with tab_lost:
                lost_data = c_data.get("lost", {})
                st.markdown(f"**Total Lost Calls in Cluster:** `{lost_data.get('total_calls', 0)}`")
                metrics = lost_data.get("metrics", {})
                if metrics:
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.write("No Lost calls inside this cluster.")
                    
            with tab_nodecision:
                nd_data = c_data.get("no-decision", {})
                st.markdown(f"**Total No-Decision Calls in Cluster:** `{nd_data.get('total_calls', 0)}`")
                metrics = nd_data.get("metrics", {})
                if metrics:
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.write("No No-Decision calls inside this cluster.")
        else:
            st.warning("No detailed profiling stats found for this cluster.")
            
    st.markdown("---")
    
    st.subheader("Global Outcome Stage Summaries (Across All Clusters)")
    st.markdown("Select a global outcome profile to review mean and range metrics:")
    
    summary_tab1, summary_tab2, summary_tab3 = st.tabs(["🏆 Global Won Stats", "❌ Global Lost Stats", "⏳ Global No-Decision Stats"])
    
    with summary_tab1:
        csv_path = os.path.join(PROJECT_ROOT, "outputs/analytics/won_outcome_stats.csv")
        if os.path.exists(csv_path):
            st.dataframe(pd.read_csv(csv_path), use_container_width=True, hide_index=True)
            
    with summary_tab2:
        csv_path = os.path.join(PROJECT_ROOT, "outputs/analytics/lost_outcome_stats.csv")
        if os.path.exists(csv_path):
            st.dataframe(pd.read_csv(csv_path), use_container_width=True, hide_index=True)
            
    with summary_tab3:
        csv_path = os.path.join(PROJECT_ROOT, "outputs/analytics/no_decision_outcome_stats.csv")
        if os.path.exists(csv_path):
            st.dataframe(pd.read_csv(csv_path), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Section 5: Agent Coaching & Summaries
# -----------------------------------------------------------------------------

elif navigation == "🎓 5. Agent Coaching & Summaries":
    st.title("🎓 Conversation Coaching & Patterns Summary")
    st.markdown("### Deterministic dialogue formulas for agent coaching and performance optimization")
    
    st.markdown("""
    <div class="story-box">
        <h4>Designing High-Performance Call Guidelines</h4>
        <p>By defining target thresholds for dialogue duration, monologues, customer sentiments, and objections, 
        we establish rules that predict renewal success without needing a direct LLM evaluation of every call. 
        Review the 5 conversion patterns below.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate patterns
    patterns_data = get_coaching_patterns(features_df)
    
    # Display the 5 patterns
    for p in patterns_data:
        with st.container():
            col_rule, col_stat = st.columns([0.7, 0.3])
            with col_rule:
                st.markdown(f"#### {p['name']}")
                st.markdown(f"**Mathematical Formula:** `{p['rule']}`")
                st.markdown(f"**Agent Coaching Cue:** *{p['coaching']}*")
            with col_stat:
                st.metric("Conversion Win Rate", f"{p['win_rate']:.2%}")
                st.caption(f"Dataset Matches: {p['total_calls']} calls")
            st.markdown("---")
            
    # LLM summary generator section
    st.subheader("⚡ Executive Coaching Summary (AI Generated)")
    st.markdown("Generate a custom coaching summary from the Gemini LLM analyzing these patterns.")
    
    if st.button("Generate AI Coaching Summary"):
        with st.spinner("Analyzing patterns via Gemini API..."):
            summary_text = generate_llm_coaching_summary(patterns_data)
            st.markdown(summary_text)

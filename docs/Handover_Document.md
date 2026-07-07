# Handover Document — Phase 3: Clustering & Outcome Analysis

This handover document summarizes the architecture, completed components, dataset insights, and next steps for the **Conversation Pattern Miner** framework as of Phase 3 completion.

---

## 1. Architecture Overview & Engineering Principles
The framework is designed to be **domain-independent**, **configuration-driven**, and **modular**. Adapting the pipeline to a new conversational domain (e.g., banking, support, collections) requires only updating configuration files and pointing to a new dataset; the source code remains identical.

- **Reusability**: No domain-specific keywords or logic are hardcoded. Everything is driven by parameters defined in domain-specific configuration files.
- **Single Responsibility**: Each module owns exactly one stage of the pipeline: loading, validation, profiling, embeddings, topics, clustering, analytics, and visualizations.
- **Pipeline Architecture**: Every stage consumes the output of the previous stage and saves intermediate artifacts to the `outputs/` directory.

---

## 2. Implemented Components
The following modules were developed, tested, and integrated by the end of Phase 3:

- **`src/utils/config_loader.py`**: Merges default configuration settings (`configs/default.yaml`) with domain specific files (`configs/insurance.yaml`).
- **`src/utils/logger.py`**: A structured JSON logger writing logs to `logs/pipeline.log` for machine ingestion, while maintaining human-readable console outputs.
- **`src/features/schemas.py`**: Pydantic schemas enforcing data validation rules on loaded call records.
- **`src/ingestion/loader.py`**: A robust file parser handling standard JSON lists and JSONL formats.
- **`src/ingestion/validator.py`**: Validates parsed records against the Pydantic schema and validates domain configuration (e.g., expected outcomes).
- **`src/preprocessing/cleaner.py`**: Turn-segmentation regex logic parsing conversation summaries into speaker utterance objects.
- **`src/embeddings/embedder.py`**: Call embeddings module using `all-MiniLM-L6-v2` SentenceTransformers.
- **`src/features/extractor.py`**: Features extractor engineering 16 different tabular features (talk ratio, monologue counts, question count etc).
- **`src/topics/topic_model.py`**: Topic model segmenting transcripts into 8 topics using BERTopic.
- **`src/clustering/clusterer.py`**: Unsupervised KMeans clustering module grouping calls into 6 clusters.
- **`src/analytics/outcome_analysis.py`**: Outcomes analyzer calculating Pearson correlations, Cohen's d effect sizes, and topic/cluster win splits.
- **`src/visualization/visualizer.py`**: Generates Plotly interactive UMAP maps, static UMAP projections, and feature boxplots.

---

## 3. Key Dataset & Statistical Insights
Our execution of the pipeline on `synthetic_calls.json` (384 records) yielded the following statistical findings:

### Predictive Feature Signals (Cohen's d Effect Size)
- **`resolution_score` ($d = 6.7144$)**: The strongest predictor. Won calls average a 0.86 resolution rate compared to 0.25 on Lost calls.
- **`customer_sentiment` ($d = 4.0876$)**: Highly predictive. Customer sentiment scores are significantly higher in Won calls (0.74) compared to Lost calls (0.28).
- **`avg_turn_length` ($d = 3.6452$)**: Successful calls have longer turn exchanges (63.14 characters vs. 43.66 on Lost calls), suggesting more detailed and productive interactions.
- **`num_objections` ($d = -2.1530$)**: A strong negative correlation. Lost calls average 1.88 objections compared to only 0.53 on Won calls.

### Cluster Profiles & Outcomes
- **Cluster 0 (95.2% Win Rate)**: Empathy-led discount closers.
- **Cluster 1 (81.4% No-Decision Rate)**: Deferred callbacks/follow-up requests.
- **Cluster 2 (88.5% Win Rate)**: Direct renewals with minor objections.
- **Cluster 3 (100.0% Win Rate)**: Immediate sign-ups/instant conversions.
- **Cluster 4 (96.3% Loss Rate)**: Unresolved price objections and rigid scripting failures.
- **Cluster 5 (75.0% No-Decision Rate)**: Customer indecisiveness regarding policy parameters.

All generated plots and statistical metrics are exported to [outputs/](file:///c:/Users/dilip/Desktop/Sutherland_Intern/outputs/).

---

## 4. Next Steps & Phase 4 Transition
The next phase focuses on **Phase 4 — Insights, Dashboard & Hand-over**:

1. **Interactive Streamlit Explorer (`dashboard/app.py`)**: Create the Streamlit dashboard layout displaying:
   - An interactive UMAP topic map (using `umap_plot.html` Plotly iframe or Plotly figure rendering).
   - Conversational metrics filters (outcome filter, cluster selection).
   - Side-by-side cluster profiling tables.
   - Inline transcript turn displays.
2. **Generalization Guide (`docs/Generalization_Guide.md`)**: Complete the step-by-step guide explaining how to point this framework to a new domain (e.g. banking or support).
3. **Unit Tests Coverage**: Add dashboard loading checks.

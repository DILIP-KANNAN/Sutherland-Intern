# Handover Document — Project Completion

This handover document summarizes the architecture, completed components, final dataset insights, and operational scripts for the **Conversation Pattern Miner** framework.

---

## 1. Architecture Overview & Engineering Principles
The framework is designed to be **domain-independent**, **configuration-driven**, and **modular**. Adapting the pipeline to a new conversational domain (e.g., banking, support, collections) requires only updating configuration files and pointing to a new dataset; the source code remains identical.

- **Reusability**: No domain-specific keywords or logic are hardcoded. Everything is driven by parameters defined in domain-specific configuration files.
- **Single Responsibility**: Each module owns exactly one stage of the pipeline: loading, validation, profiling, embeddings, topics, clustering, analytics, labeling, and dashboard visualization.
- **Pipeline Architecture**: Every stage consumes the output of the previous stage and saves intermediate artifacts to the `outputs/` directory.

---

## 2. Implemented Components
The following modules have been fully developed, tested, and integrated:

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
- **`src/analytics/outcome_analysis.py`**: Outcomes analyzer calculating Pearson correlations, Cohen's d effect sizes, detailed cluster outcome stats, and global stage summaries.
- **`src/visualization/visualizer.py`**: Generates Plotly interactive UMAP maps, static UMAP projections, feature boxplots, and cluster 2x2 comparison subplots.
- **`src/topics/labeler.py`**: Dual-method segment labeler containing regex PII masking, Heuristic keywords rules, and Gemini API batch prompt requests.
- **`dashboard/app.py`**: Multi-page interactive Streamlit dashboard incorporating the conversation analytics story, dialogue feature explorer, interactive 2D scatter plots, 2x2 cluster outcome grids, and dynamic agent coaching cue metrics with an on-the-fly LLM performance analysis helper.

---

## 3. Key Dataset & Statistical Insights
Execution of the pipeline on `synthetic_calls.json` (384 records) yielded the following statistical findings:

### Predictive Feature Signals (Cohen's d Effect Size)
- **`resolution_score` ($d = 6.71$)**: The strongest predictor. Won calls average a 0.86 resolution rate compared to 0.25 on Lost calls.
- **`customer_sentiment` ($d = 4.08$)**: Highly predictive. Customer sentiment scores are significantly higher in Won calls (0.74) compared to Lost calls (0.28).
- **`avg_turn_length` ($d = 3.64$)**: Successful calls have longer turn exchanges (63.14 characters vs. 43.66 on Lost calls).
- **`num_objections` ($d = -2.15$)**: A strong negative correlation. Lost calls average 1.88 objections compared to only 0.53 on Won calls.

### Cluster Profiles & Outcomes
- **Cluster 0 (95.2% Win Rate)**: Empathy-led discount closers.
- **Cluster 1 (81.4% No-Decision Rate)**: Deferred callbacks/follow-up requests.
- **Cluster 2 (88.5% Win Rate)**: Direct renewals with minor objections.
- **Cluster 3 (100.0% Win Rate)**: Immediate sign-ups/instant conversions.
- **Cluster 4 (96.3% Loss Rate)**: Unresolved price objections and rigid scripting failures.
- **Cluster 5 (75.0% No-Decision Rate)**: Customer indecisiveness regarding policy parameters.

---

## 4. Run Instructions

### To run the pipeline orchestrator:
```bash
python run_pipeline.py --config configs/insurance.yaml --phase 3
```

### To run the 18 tests:
```bash
pytest tests/
```

### To run the analytical dashboard:
```bash
streamlit run dashboard/app.py
```
*(If you are already inside the `dashboard/` directory, run `streamlit run app.py` instead)*

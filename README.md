# Conversation Pattern Miner

A domain-independent, configuration-driven **Conversation Analytics Framework** designed to mine call transcripts, extract conversational features, discover topics/clusters, label segments, and identify outcome-correlating patterns.

## Current Project Status
- **Phase 4: Insights, Dashboard & Hand-over** is complete.
- The framework successfully ingests and validates call transcripts, extracts semantic embeddings and 16 conversational features, clusters calls into 6 distinct profiles using KMeans, performs statistical outcome correlation analysis (Cohen's d and Pearson correlation), labels segments using both strict NLP heuristics and batched Gemini 2.5 LLM calls, and exposes a fully interactive, story-based Streamlit dashboard with automated coaching generators.

## Project Structure
```
conversation_pattern_miner/
├── configs/               # Domain-specific YAML configurations (default, insurance)
├── data/
│   └── processed/         # Processed datasets (synthetic_calls, validated labelled_corpus)
├── dashboard/
│   └── app.py             # Story-based Streamlit dashboard app
├── docs/                  # Project documentation (Architecture, Setup Guide, Handover, Generalization)
├── src/
│   ├── ingestion/         # Dataset loading and Pydantic schema validation logic
│   ├── features/          # Validation schemas and metrics extraction functions
│   ├── preprocessing/     # Transcript clean turn parsing and segmentation
│   ├── embeddings/        # Semantic call embeddings (SentenceTransformers)
│   ├── topics/            # Unsupervised topic discovery (BERTopic) and segment labeling (Heuristic & LLM)
│   ├── clustering/        # Unsupervised KMeans call grouping
│   ├── analytics/         # Statistical correlations, splits, and Cohen's d effect sizes
│   ├── visualization/     # UMAP scatter plotting and feature boxplots
│   └── utils/             # Merging YAML configs and custom JSON logging setup
├── outputs/
│   ├── profiler/          # Dataset profile report and basic EDA distributions
│   ├── embeddings/        # Call semantic embeddings matrix (npy format)
│   ├── features/          # Tabular conversational feature matrix (csv format)
│   ├── topics/            # Topic assignment arrays, definitions, and label comparisons (json format)
│   ├── clusters/          # KMeans cluster assignments (json format)
│   ├── analytics/         # Statistical reports (Pearson correlations, Cohen's d, outcome stage stats)
│   └── visualizations/    # Static PNG boxplots, cluster profiles, and interactive HTML UMAP plots
├── tests/                 # Unit tests (18 tests covering all pipeline stages and dashboard)
├── run_pipeline.py        # Framework main orchestrator runner
└── requirements.txt       # Python package dependencies
```

## Running the Pipeline Phases

The framework execution is split into modular, sequential phases. You can execute any phase directly using the `--phase` argument:

### Phase 1 — Ingestion, Validation & Profiling
* **Description**: Ingests raw conversational data, validates it against strict schemas, checks values against domain configuration labels, and performs exploratory data profiling (calculating demographics, objection splits, and plotting initial distributions).
* **Command**:
  ```bash
  python run_pipeline.py --config configs/insurance.yaml --phase 1
  ```
* **Key Outputs**:
  - Validated corpus: `data/processed/labelled_corpus.jsonl`
  - Profile statistics: `outputs/profiler/dataset_profile.json`
  - Plots: `outputs/profiler/outcome_distribution.png`, `talk_ratio_distribution.png`

### Phase 2 — Core NLP (Embeddings, Features & Topics)
* **Description**: Segment and parse clean dialogue turns. Generates 384-dimensional dense semantic call embeddings using the transformer backend and extracts 16 conversational and style features (like monologues, sentiment, question counts). Performs unsupervised topic modeling using BERTopic to map the calls into 8 core topic representations.
* **Command**:
  ```bash
  python run_pipeline.py --config configs/insurance.yaml --phase 2
  ```
* **Key Outputs**:
  - Embeddings matrix: `outputs/embeddings/call_embeddings.npy` (shape 384x384)
  - Features DataFrame: `outputs/features/call_features.csv` (shape 384x18)
  - Topic definitions & assignments: `outputs/topics/topic_assignments.json`

### Phase 3 — Clustering & Outcome Analysis
* **Description**: Groups call vector representations into 6 clusters using KMeans. Runs outcome correlation metrics (Pearson coefficients) and Cohen's d effect size calculations between conversion states (Won vs. Lost). Generates static outcome boxplots and an interactive 2D UMAP scatter map. Generates strict NLP Heuristic labels and calls `gemini-2.5-flash` in batch mode (exactly 2 calls with regex-anonymized transcripts) to output segment name comparisons.
* **Command**:
  ```bash
  python run_pipeline.py --config configs/insurance.yaml --phase 3
  ```
* **Key Outputs**:
  - Cluster IDs: `outputs/clusters/cluster_assignments.json`
  - Analytics CSVs: `outputs/analytics/cohens_d_analysis.csv`, `feature_correlations.csv`, and decoupled outcome stage stats
  - UMAP Scatter Map: `outputs/visualizations/umap_plot.html` (interactive HTML)
  - Segment Label comparisons: `outputs/topics/labels_comparison.json`

### Phase 4 — Analytical Streamlit Dashboard
* **Description**: Launces the story-based multi-page dashboard displaying the narrative, dialogue distributions, interactive UMAP scatter plot, 2x2 cluster-outcome profile boxplots, and agent coaching cues with an on-the-fly AI coaching summary generator.
* **Command**:
  ```bash
  streamlit run dashboard/app.py
  ```

## Running Tests
To verify the implementation using the test suite of 18 tests:
```bash
pytest tests/
```

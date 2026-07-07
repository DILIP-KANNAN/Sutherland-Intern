# Conversation Pattern Miner

A domain-independent, configuration-driven **Conversation Analytics Framework** designed to mine call transcripts, extract conversational features, discover topics/clusters, and identify outcome-correlating patterns.

## Current Project Status
- **Phase 3: Clustering & Outcome Analysis** is complete. 
- The framework successfully ingests and validates call transcripts, extracts semantic embeddings and 16 conversational features, clusters calls into 6 distinct profiles using KMeans, performs statistical outcome correlation analysis (Cohen's d and Pearson correlation), and generates static boxplots and interactive 2D UMAP scatter maps.

## Project Structure
```
conversation_pattern_miner/
├── configs/               # Domain-specific YAML configurations (default, insurance)
├── data/
│   └── processed/         # Processed datasets (synthetic_calls, validated labelled_corpus)
├── docs/                  # Project documentation (Architecture, Setup Guide, Handover Document)
├── src/
│   ├── ingestion/         # Dataset loading and Pydantic schema validation logic
│   ├── features/          # Validation schemas and metrics extraction functions
│   ├── preprocessing/     # Transcript clean turn parsing and segmentation
│   ├── embeddings/        # Semantic call embeddings (SentenceTransformers)
│   ├── topics/            # Unsupervised topic discovery (BERTopic)
│   ├── clustering/        # Unsupervised KMeans call grouping
│   ├── analytics/         # Statistical correlations, splits, and Cohen's d effect sizes
│   ├── visualization/     # UMAP scatter plotting and feature boxplots
│   └── utils/             # Merging YAML configs and custom JSON logging setup
├── outputs/
│   ├── profiler/          # Dataset profile report and basic EDA distributions
│   ├── embeddings/        # Call semantic embeddings matrix (npy format)
│   ├── features/          # Tabular conversational feature matrix (csv format)
│   ├── topics/            # Topic assignment arrays and definitions (json format)
│   ├── clusters/          # KMeans cluster assignments (json format)
│   ├── analytics/         # Statistical reports (Pearson correlations, Cohen's d)
│   └── visualizations/    # Static PNG boxplots and interactive HTML UMAP plots
├── tests/                 # Unit tests (11 tests covering all pipeline stages)
├── run_pipeline.py        # Framework main orchestrator runner
└── requirements.txt       # Python package dependencies
```

## Running the Pipeline Phases

The framework execution is split into three modular, sequential phases. You can execute any phase directly using the `--phase` argument:

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
* **Description**: Groups call vector representations into 6 clusters using KMeans. Runs outcome correlation metrics (Pearson coefficients) and Cohen's d effect size calculations between conversion states (Won vs. Lost). Generates static outcome boxplots and an interactive 2D UMAP scatter map.
* **Command**:
  ```bash
  python run_pipeline.py --config configs/insurance.yaml --phase 3
  ```
* **Key Outputs**:
  - Cluster IDs: `outputs/clusters/cluster_assignments.json`
  - Analytics CSVs: `outputs/analytics/cohens_d_analysis.csv`, `feature_correlations.csv`
  - UMAP Scatter Map: `outputs/visualizations/umap_plot.html` (interactive HTML)


## Running Tests
To verify the implementation using the test suite:
```bash
pytest tests/
```

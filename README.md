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

## Running the Pipeline
To run the pipeline up to Phase 3 (Clustering, Analytics, and Visualizations):
```bash
python run_pipeline.py --config configs/insurance.yaml --phase 3
```

## Running Tests
To verify the implementation using the test suite:
```bash
pytest tests/
```

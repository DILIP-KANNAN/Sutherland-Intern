# Conversation Pattern Miner

A domain-independent, configuration-driven **Conversation Analytics Framework** designed to mine call transcripts, extract conversational features, discover topics/clusters, and identify outcome-correlating patterns.

## Current Project Status
- **Phase 1: Ingestion, Validation, and Dataset Profiling** is complete. 
- The framework successfully ingests and validates the corpus, outputs clean JSONL files, and generates statistical profiles along with key EDA plots.

## Project Structure
```
conversation_pattern_miner/
├── configs/               # Domain-specific YAML configurations (default, insurance)
├── data/
│   └── processed/         # Processed datasets (synthetic_calls, validated labelled_corpus)
├── docs/                  # Project documentation (Architecture, Setup Guide, Handover Document)
├── src/
│   ├── ingestion/         # Dataset loading and schema validation logic
│   ├── features/          # Validation schemas and metrics schemas
│   ├── profiler/          # Dataset statistics, profiling and EDA visualization
│   └── utils/             # Merging YAML configs and custom JSON logging setup
├── outputs/
│   └── profiler/          # Profiler metrics and visualization outputs
├── tests/                 # Unit tests (loader, validator, profiler tests)
├── run_pipeline.py        # Framework main orchestrator runner
└── requirements.txt       # Python package dependencies
```

## Running the Pipeline
To run Phase 1 (Ingestion, Validation, and Profiling) of the pipeline:
```bash
python run_pipeline.py --config configs/insurance.yaml --phase 1
```

## Running Tests
To verify implementation using the test suite:
```bash
pytest tests/
```

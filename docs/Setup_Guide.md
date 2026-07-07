# Setup & Environment Guide

This document describes the steps required to configure your development environment, install project dependencies, run tests, and execute the Conversation Pattern Miner pipeline.

---

## 1. Prerequisites
- **Python 3.11+**
- **Git** client
- A virtual environment tool (`venv` or `conda`)

---

## 2. Setting Up the Environment

### Step 1: Create a Virtual Environment
Navigate to the root directory of the repository and create a Python virtual environment:
```bash
# Using standard venv
python -m venv .venv

# Using Conda
conda create -n sutherland python=3.11
```

### Step 2: Activate the Virtual Environment
Activate the environment based on your operating system:
```bash
# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\activate.ps1

# macOS / Linux
source .venv/bin/activate

# Conda activation
conda activate sutherland
```

### Step 3: Install Dependencies
Install all required libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```

---

## 3. Running & Verifying the Installation

### Verify with Unit Tests
Run the test suite using `pytest` to make sure all modules (loader, validator, profiler, extractor, topic model, clusterer, and analytics) are functioning correctly:
```bash
pytest tests/
```

### Execute the Core NLP, Clustering & Analytics Pipeline
To execute the pipeline up to Phase 3 using the default insurance configuration:
```bash
python run_pipeline.py --config configs/insurance.yaml --phase 3
```

---

## 4. Verification Checklists
Upon running Phase 3, verify the creation of the following outputs inside the `outputs/` folder:

- **Ingestion & Validation**:
  - `data/processed/labelled_corpus.jsonl` (Validated transcripts)
- **Dataset Profile**:
  - `outputs/profiler/dataset_profile.json` (Demographic summary)
  - `outputs/profiler/*.png` (Basic distribution plots)
- **Embeddings Module**:
  - `outputs/embeddings/call_embeddings.npy` (Numpy matrix of shape 384x384)
- **Features Extractor**:
  - `outputs/features/call_features.csv` (Features DataFrame of shape 384x18)
- **Topic Modeling**:
  - `outputs/topics/topic_assignments.json` (Topic IDs and defining keywords)
- **KMeans Clustering**:
  - `outputs/clusters/cluster_assignments.json` (Cluster assignments list)
- **Outcome Analysis**:
  - `outputs/analytics/feature_correlations.csv` (Pearson correlations)
  - `outputs/analytics/cohens_d_analysis.csv` (Cohen's d effect sizes)
  - `outputs/analytics/topic_outcome_analysis.csv` (Topic outcome splits)
  - `outputs/analytics/cluster_outcome_analysis.csv` (Cluster outcome splits)
- **Dimensionality Reduction & Visualizations**:
  - `outputs/visualizations/umap_plot.html` (Interactive, hoverable Plotly HTML scatter map)
  - `outputs/visualizations/umap_plot.png` (Static UMAP plot)
  - `outputs/visualizations/*_boxplot.png` (Outcome-split feature boxplots)
- **Structured Logs**:
  - `logs/pipeline.log` (Execution status recorded in JSON format)

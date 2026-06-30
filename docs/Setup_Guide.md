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
Run the test suite using `pytest` to make sure all components (loader, validator, profiler) are functioning correctly:
```bash
pytest tests/
```

### Execute the Ingestion and Profiling Pipeline
To execute Phase 1 of the pipeline using the default insurance configuration:
```bash
python run_pipeline.py --config configs/insurance.yaml --phase 1
```

---

## 4. Verification Checklists
Upon running Phase 1, verify the creation of the following outputs:

- **Clean Corpus**: A JSONL file is exported to `data/processed/labelled_corpus.jsonl`.
- **JSON Profile**: Summary statistics are saved to `outputs/profiler/dataset_profile.json`.
- **Visualizations**: Check for the three generated PNG charts inside `outputs/profiler/`:
  - `outcome_distribution.png`
  - `call_duration_vs_turns.png`
  - `talk_ratio_distribution.png`
- **Structured Log**: Execution status logs are recorded in JSON format at `logs/pipeline.log`.

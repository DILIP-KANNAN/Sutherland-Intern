# Generalization Guide — Adapting to New Domains

The **Conversation Pattern Miner** is a domain-independent conversational analytics framework. Adapting it to process conversations from a completely different domain (e.g., Banking Sales, Customer Support, Collections, HR Interviews, or Healthcare) does **not** require modifying the core code. 

You only need to supply:
1. A new dataset (JSON list or JSONL formats).
2. A new configuration file in the `configs/` directory.

This guide outlines the step-by-step generalization workflow.

---

## 1. Structure of the Domain Configuration
To declare a new domain, create a new YAML file (e.g., `configs/banking.yaml`). The config loader merges your custom variables with the default framework settings.

Your configuration must specify:
1. **Domain Name**: The target label for tracking logs.
2. **Data Paths**: Location of your raw dataset and validated processed files.
3. **Outcome Labels**: Valid conversion states expected in your data.
4. **Objection Categories**: Supported categories for the domain.

### Example Banking Configuration (`configs/banking.yaml`):
```yaml
domain: "banking_sales"

data:
  raw_path: "data/raw/credit_card_calls.json"
  processed_path: "data/processed/banking_corpus.jsonl"

metadata:
  outcome_labels:
    - "approved"
    - "declined"
    - "follow_up"
  objection_types:
    - "annual_fee"
    - "credit_limit"
    - "security_concern"
    - "partner_consult"
    - "none"
```

---

## 2. Generalization Steps

### Step 1: Format the Raw Dataset
Ensure your raw dataset contains records matching the Pydantic schema constraints. The minimum fields required are:
- `call_id` (str): Unique identifier.
- `outcome` (str): Must match the labels defined in your custom config.
- `transcript_summary` (str): Delimited conversational turns format (e.g. `[T1] Agent: "Hello" | [T2] Customer: "Hi"`).
- Other continuous variables you wish to profile (e.g., tenure, scores).

### Step 2: Write the Configuration File
Save your configuration variables as a YAML file under `configs/`.

### Step 3: Run the Pipeline Orchestrator
Execute the pipeline using your new configuration file:
```bash
# Ingest and validate data
python run_pipeline.py --config configs/banking.yaml --phase 1

# Generate embeddings, style features, and topics
python run_pipeline.py --config configs/banking.yaml --phase 2

# Cluster calls and perform outcome analytics
python run_pipeline.py --config configs/banking.yaml --phase 3
```

### Step 4: Launch the Dashboard
Run Streamlit to explore the newly generated insights:
```bash
streamlit run dashboard/app.py
```
Streamlit will load the statistics directly from your updated `outputs/` folder.

---

## 3. Domain Independence Under the Hood

The framework components are completely domain-independent:
- **`src/preprocessing/cleaner.py`**: Turn-segmentation regex matches conversational dynamics irrespective of the vocabulary.
- **`src/embeddings/embedder.py`**: SentenceTransformers map semantic meanings (e.g. mapping "declined" to "lost" semantics) automatically.
- **`src/clustering/clusterer.py`**: KMeans groups vectors mathematically based on spatial proximity.
- **`src/analytics/outcome_analysis.py`**: Pearson correlations and Cohen's d effect sizes run pure statistics over the engineered columns.
- **`src/topics/labeler.py`**: The LLM prompt dynamically receives the keywords, statistics, and summaries, and creates semantic names appropriate to your new domain.

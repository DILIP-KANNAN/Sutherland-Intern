# Handover Document — Phase 1: Foundation & Labelled Data

This handover document summarizes the architecture, completed components, dataset insights, and next steps for the **Conversation Pattern Miner** framework.

---

## 1. Architecture Overview & Engineering Principles
The framework is designed to be **domain-independent**, **configuration-driven**, and **modular**. Adapting the pipeline to a new conversational domain (e.g., banking, support, collections) requires only updating configuration files and pointing to a new dataset; the source code remains identical.

- **Reusability**: No domain-specific keywords or logic are hardcoded. Everything is driven by parameters defined in domain-specific configuration files.
- **Single Responsibility**: Each module owns exactly one stage of the pipeline: loading, validation, profiling, and plotting.
- **Pipeline Architecture**: Every stage consumes the output of the previous stage and saves intermediate artifacts to the `outputs/` directory.

---

## 2. Implemented Components
The following modules were developed and tested in Phase 1:

- **`src/utils/config_loader.py`**: Merges default configuration settings (`configs/default.yaml`) with domain specific files (`configs/insurance.yaml`).
- **`src/utils/logger.py`**: A structured JSON logger writing logs to `logs/pipeline.log` for machine ingestion, while maintaining human-readable console outputs.
- **`src/features/schemas.py`**: Pydantic schemas enforcing data validation rules on loaded call records.
- **`src/ingestion/loader.py`**: A robust file parser handling standard JSON lists and JSONL formats.
- **`src/ingestion/validator.py`**: Validates parsed records against the Pydantic schema and validates domain configuration (e.g., expected outcomes).
- **`src/profiler/profiler.py`**: Summarizes dataset demographics, performs EDA, and generates visualization plots.

---

## 3. Dataset Insights (Insurance Domain)
Running Phase 1 on the augmented dataset (`data/processed/synthetic_calls.json`, 384 records) yielded the following insights:

- **Volume and Outcomes**: 384 validated records, split into **195 Won** (50.8%), **105 Lost** (27.3%), and **84 No-decision** (21.9%).
- **Conversational Metrics**: The average call length is **8.77 minutes**, average turn count is **7.81 turns**, and the average agent talk ratio is **54.0%**.
- **Common Objections**: The most frequent objection types raised by customers are `time_deferral` (84 instances) and `cancellation_intent` (81 instances).

All generated plots and statistical metrics are exported to [outputs/profiler/](file:///c:/Users/dilip/Desktop/Sutherland_Intern/outputs/profiler/).

---

## 4. Next Steps & Phase 2 Transition
The next engineer/squad taking over should focus on **Phase 2 — Core NLP**:

1. **Turn Preprocessing (`src/preprocessing/`)**: Build segments parsing logic to split `transcript_summary` (e.g., parsing `[T1] Agent: "..." | [T2] Customer: "..."`) into clean structured turns.
2. **Conversational Feature Engineering (`src/features/extractor.py`)**: Extract dynamics like agent/customer talk ratio, monologue lengths, question counts, and objection handling markers.
3. **Semantic Embedding Generation (`src/embeddings/embedder.py`)**: Implement call-level sentence embeddings pooling using `sentence-transformers`.
4. **Topic Discovery (`src/topics/topic_model.py`)**: Train a `BERTopic` model to classify conversational themes.

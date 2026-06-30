import os
import tempfile
import json
import pytest
from src.ingestion.loader import load_raw_dataset

def test_load_raw_dataset_json_list():
    sample_data = [{"call_id": "C1", "outcome": "won", "transcript_summary": "Agent: hello"}]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
        json.dump(sample_data, f)
        temp_path = f.name
        
    try:
        loaded = load_raw_dataset(temp_path)
        assert len(loaded) == 1
        assert loaded[0]["call_id"] == "C1"
    finally:
        os.remove(temp_path)

def test_load_raw_dataset_jsonl():
    sample_lines = (
        '{"call_id": "C1", "outcome": "won", "transcript_summary": "Agent: hello"}\n'
        '{"call_id": "C2", "outcome": "lost", "transcript_summary": "Agent: bye"}'
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as f:
        f.write(sample_lines)
        temp_path = f.name
        
    try:
        loaded = load_raw_dataset(temp_path)
        assert len(loaded) == 2
        assert loaded[0]["call_id"] == "C1"
        assert loaded[1]["call_id"] == "C2"
    finally:
        os.remove(temp_path)

def test_load_raw_dataset_not_found():
    with pytest.raises(FileNotFoundError):
        load_raw_dataset("non_existent_file.json")

from src.ingestion.validator import validate_dataset

def test_validate_dataset_valid():
    sample_data = [{
        "call_id": "CALL-1001",
        "outcome": "won",
        "transcript_summary": "Agent: Hello | Customer: Hi"
    }]
    valid, invalid = validate_dataset(sample_data, expected_outcome_labels=["won", "lost"])
    assert len(valid) == 1
    assert len(invalid) == 0
    assert valid[0].call_id == "CALL-1001"

def test_validate_dataset_invalid_outcome():
    sample_data = [{
        "call_id": "CALL-1002",
        "outcome": "refunded",  # not in expected outcome labels
        "transcript_summary": "Agent: Hello"
    }]
    valid, invalid = validate_dataset(sample_data, expected_outcome_labels=["won", "lost"])
    assert len(valid) == 0
    assert len(invalid) == 1

def test_validate_dataset_missing_required():
    sample_data = [{
        "call_id": "CALL-1003",
        # missing outcome and transcript_summary
    }]
    valid, invalid = validate_dataset(sample_data)
    assert len(valid) == 0
    assert len(invalid) == 1

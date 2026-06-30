import os
import shutil
import tempfile
from src.features.schemas import CallRecord
from src.profiler.profiler import profile_dataset

def test_profile_dataset():
    sample_records = [
        CallRecord(
            call_id="CALL-1", 
            outcome="won", 
            transcript_summary="A: hello", 
            num_turns=5, 
            call_duration_seconds=120, 
            agent_talk_ratio=0.5
        ),
        CallRecord(
            call_id="CALL-2", 
            outcome="lost", 
            transcript_summary="A: hello", 
            num_turns=10, 
            call_duration_seconds=240, 
            agent_talk_ratio=0.6
        )
    ]
    
    temp_dir = tempfile.mkdtemp()
    try:
        profile = profile_dataset(sample_records, temp_dir)
        
        # Verify statistical results
        assert "corpus_summary" in profile
        assert profile["corpus_summary"]["total_calls"] == 2
        assert profile["corpus_summary"]["avg_turns"] == 7.5
        assert profile["corpus_summary"]["avg_agent_talk_ratio"] == 0.55
        
        # Verify file outputs
        assert os.path.exists(os.path.join(temp_dir, "dataset_profile.json"))
        assert os.path.exists(os.path.join(temp_dir, "outcome_distribution.png"))
        assert os.path.exists(os.path.join(temp_dir, "call_duration_vs_turns.png"))
        assert os.path.exists(os.path.join(temp_dir, "talk_ratio_distribution.png"))
    finally:
        shutil.rmtree(temp_dir)

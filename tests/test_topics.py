import os
import shutil
import tempfile
from src.features.schemas import CallRecord
from src.topics.topic_model import TopicModel

def test_topic_model():
    # Setup mock transcripts to train the model
    records = [
        CallRecord(
            call_id=f"CALL-{i}", 
            outcome="won", 
            transcript_summary=f"Hello, this is a call regarding annual renewal for package {i}."
        )
        for i in range(15)
    ]
    
    # Initialize with small min_topic_size to fit on mock data
    topic_model = TopicModel(min_topic_size=2, nr_topics=4)
    topics, topic_info = topic_model.fit_transform(records)
    
    assert len(topics) == 15
    assert len(topic_info) > 0
    
    temp_dir = tempfile.mkdtemp()
    try:
        json_path = topic_model.save_model(temp_dir, topics, topic_info)
        assert os.path.exists(json_path)
    finally:
        shutil.rmtree(temp_dir)

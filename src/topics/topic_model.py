import os
import json
import logging
from typing import List, Dict, Any, Tuple
from bertopic import BERTopic
from src.features.schemas import CallRecord

logger = logging.getLogger("pattern_miner")

class TopicModel:
    def __init__(self, min_topic_size: int = 5, nr_topics: int = 8):
        """
        Initializes the BERTopic model.
        """
        logger.info(f"Initializing BERTopic model with min_topic_size={min_topic_size}, nr_topics={nr_topics}")
        self.nr_topics = nr_topics
        self.model = BERTopic(
            nr_topics=self.nr_topics, 
            min_topic_size=min_topic_size, 
            verbose=False
        )
        
    def fit_transform(
        self, 
        records: List[CallRecord], 
        embeddings = None
    ) -> Tuple[List[int], Dict[int, List[Tuple[str, float]]]]:
        """
        Fits BERTopic on the call transcript summaries and returns topic assignments
        along with topic representation keywords.
        """
        docs = [record.transcript_summary for record in records]
        
        logger.info("Fitting and transforming call documents through BERTopic pipeline...")
        topics, _ = self.model.fit_transform(docs, embeddings=embeddings)
        
        # Retrieve representation words for each topic
        topic_info = self.model.get_topics()
        
        return topics, topic_info
        
    def save_model(
        self, 
        output_dir: str, 
        topics: List[int], 
        topic_info: Dict[int, List[Tuple[str, float]]]
    ) -> str:
        """
        Saves topic model assignments and definitions to a JSON file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Structure the keywords for easy loading by the dashboard
        formatted_topics = {}
        for topic_id, words in topic_info.items():
            formatted_topics[str(topic_id)] = [
                {"word": word, "weight": float(weight)} for word, weight in words
            ]
            
        output_data = {
            "topic_assignments": [int(t) for t in topics],
            "topic_definitions": formatted_topics
        }
        
        json_path = os.path.join(output_dir, "topic_assignments.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Topic assignments saved successfully to {json_path}")
        return json_path

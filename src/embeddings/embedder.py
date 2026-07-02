import os
import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src.features.schemas import CallRecord

logger = logging.getLogger("pattern_miner")

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the sentence transformer embedding model.
        """
        logger.info(f"Loading sentence-transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(
        self, 
        records: List[CallRecord], 
        output_path: str = None, 
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Generates call-level semantic embeddings for all validated records.
        """
        if output_path and os.path.exists(output_path) and not force_recompute:
            logger.info(f"Found cached call embeddings at {output_path}. Loading...")
            return np.load(output_path)
            
        logger.info(f"Generating semantic embeddings for {len(records)} calls...")
        # Convert raw transcript summary to vector embedding representation
        texts = [record.transcript_summary for record in records]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings_arr = np.array(embeddings)
        
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            np.save(output_path, embeddings_arr)
            logger.info(f"Successfully exported call embeddings of shape {embeddings_arr.shape} to {output_path}")
            
        return embeddings_arr

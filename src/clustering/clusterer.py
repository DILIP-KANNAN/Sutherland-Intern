import os
import json
import logging
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger("pattern_miner")

class Clusterer:
    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        """
        Initializes the KMeans clustering module.
        """
        logger.info(f"Initializing KMeans clusterer with n_clusters={n_clusters}")
        self.n_clusters = n_clusters
        self.model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=random_state, 
            n_init='auto'
        )
        
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fits KMeans on the dense embeddings matrix and returns cluster IDs.
        """
        logger.info(f"Fitting KMeans on embedding matrix of shape {embeddings.shape}...")
        return self.model.fit_predict(embeddings)
        
    def save_assignments(self, output_path: str, assignments: np.ndarray) -> str:
        """
        Saves cluster assignments to a JSON file.
        """
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        data = {
            "cluster_assignments": [int(x) for x in assignments]
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Cluster assignments exported successfully to {output_path}")
        return output_path

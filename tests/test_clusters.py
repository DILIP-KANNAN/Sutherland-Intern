import numpy as np
from src.clustering.clusterer import Clusterer

def test_clusterer():
    # Create mock 10x384 embedding matrix
    mock_embeddings = np.random.rand(10, 384)
    clusterer = Clusterer(n_clusters=3)
    clusters = clusterer.fit_predict(mock_embeddings)
    
    assert len(clusters) == 10
    # Ensure generated clusters are within correct range (0 to 2)
    assert all(0 <= c < 3 for c in clusters)

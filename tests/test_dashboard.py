import os
import json
import pandas as pd
from dashboard.app import load_features, load_labels_comparison, load_cluster_profiles, load_cohens_d

def test_dashboard_files_exist():
    # Verify that the dashboard script is present
    assert os.path.exists("dashboard/app.py")
    
def test_dashboard_data_loading():
    # Verify loading functions parse the data structures successfully
    features = load_features()
    assert isinstance(features, pd.DataFrame)
    
    labels = load_labels_comparison()
    assert isinstance(labels, dict)
    
    profiles = load_cluster_profiles()
    assert isinstance(profiles, dict)
    
    cohens_d = load_cohens_d()
    assert isinstance(cohens_d, pd.DataFrame)

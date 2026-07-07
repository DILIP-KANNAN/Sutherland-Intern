import os
import shutil
import tempfile
import pandas as pd
from src.analytics.outcome_analysis import (
    calculate_cohens_d,
    generate_detailed_cluster_profiles,
    export_decoupled_outcome_stats
)

def test_calculate_cohens_d():
    # Setup two mock numerical distributions
    group1 = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0])
    group2 = pd.Series([5.0, 6.0, 5.0, 7.0, 6.0])
    
    d_val = calculate_cohens_d(group1, group2)
    
    # Check that effect size is computed correctly and indicates positive separation
    assert d_val > 0
    assert abs(d_val - 5.8) < 0.1

def test_generate_detailed_cluster_profiles():
    # Setup mock features DataFrame
    df = pd.DataFrame({
        "outcome": ["won", "lost", "no-decision", "won", "lost"],
        "talk_ratio": [0.55, 0.48, 0.50, 0.58, 0.46]
    })
    topics = [1, 2, 1, 2, 1]
    clusters = [0, 1, 0, 1, 0]
    
    temp_dir = tempfile.mkdtemp()
    try:
        generate_detailed_cluster_profiles(df, topics, clusters, temp_dir)
        json_path = os.path.join(temp_dir, "cluster_outcome_profiles.json")
        assert os.path.exists(json_path)
    finally:
        shutil.rmtree(temp_dir)

def test_export_decoupled_outcome_stats():
    # Setup mock features DataFrame
    df = pd.DataFrame({
        "outcome": ["won", "lost", "no-decision", "won", "lost"],
        "talk_ratio": [0.55, 0.48, 0.50, 0.58, 0.46]
    })
    
    temp_dir = tempfile.mkdtemp()
    try:
        export_decoupled_outcome_stats(df, temp_dir)
        for outcome in ["won", "lost", "no_decision"]:
            csv_path = os.path.join(temp_dir, f"{outcome}_outcome_stats.csv")
            assert os.path.exists(csv_path)
    finally:
        shutil.rmtree(temp_dir)

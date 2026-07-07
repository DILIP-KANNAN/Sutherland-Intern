import pandas as pd
from src.analytics.outcome_analysis import calculate_cohens_d

def test_calculate_cohens_d():
    # Setup two mock numerical distributions
    group1 = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0])
    group2 = pd.Series([5.0, 6.0, 5.0, 7.0, 6.0])
    
    d_val = calculate_cohens_d(group1, group2)
    
    # Check that effect size is computed correctly and indicates positive separation
    assert d_val > 0
    assert abs(d_val - 5.8) < 0.1


import os
import logging
from typing import List
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("pattern_miner")

def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calculates Cohen's d effect size between two groups.
    Formula: d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    if n1 <= 1 or n2 <= 1:
        return 0.0
        
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
        
    return (group1.mean() - group2.mean()) / pooled_std

def perform_outcome_analysis(
    features_df: pd.DataFrame, 
    topics: List[int], 
    clusters: List[int], 
    output_dir: str
):
    """
    Performs comprehensive statistical correlation, effect sizes, 
    and distribution splits for both features and topics grouped by outcome.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone and merge columns
    df = features_df.copy()
    df["topic_id"] = topics
    df["cluster_id"] = clusters
    
    # 1. Feature Correlations & Cohen's d
    # Subset for won vs lost binary mapping
    df_binary = df[df["outcome"].isin(["won", "lost"])].copy()
    df_binary["outcome_numeric"] = df_binary["outcome"].map({"won": 1, "lost": 0})
    
    numeric_cols = df_binary.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifiers and group IDs
    numeric_cols = [
        c for c in numeric_cols 
        if c not in ["topic_id", "cluster_id", "outcome_numeric", "call_id"]
    ]
    
    correlations = []
    cohens_d_list = []
    
    for col in numeric_cols:
        if df_binary[col].std() == 0:
            continue
            
        # Pearson correlation coefficient and p-value
        corr, p_val = stats.pearsonr(df_binary[col], df_binary["outcome_numeric"])
        correlations.append({
            "feature": col,
            "pearson_correlation": round(corr, 4),
            "p_value": round(p_val, 6)
        })
        
        # Cohen's d effect size between won and lost calls
        won_group = df_binary[df_binary["outcome"] == "won"][col]
        lost_group = df_binary[df_binary["outcome"] == "lost"][col]
        d_val = calculate_cohens_d(won_group, lost_group)
        
        cohens_d_list.append({
            "feature": col,
            "won_mean": round(float(won_group.mean()), 4),
            "lost_mean": round(float(lost_group.mean()), 4),
            "cohens_d": round(float(d_val), 4)
        })
        
    corr_df = pd.DataFrame(correlations).sort_values(by="pearson_correlation", key=abs, ascending=False)
    cohens_d_df = pd.DataFrame(cohens_d_list).sort_values(by="cohens_d", key=abs, ascending=False)
    
    corr_df.to_csv(os.path.join(output_dir, "feature_correlations.csv"), index=False)
    cohens_d_df.to_csv(os.path.join(output_dir, "cohens_d_analysis.csv"), index=False)
    
    # 2. Topic Outcome Splits
    topic_analysis = []
    for t_id in df["topic_id"].unique():
        t_group = df[df["topic_id"] == t_id]
        total = len(t_group)
        won = len(t_group[t_group["outcome"] == "won"])
        lost = len(t_group[t_group["outcome"] == "lost"])
        no_decision = len(t_group[t_group["outcome"] == "no-decision"])
        
        topic_analysis.append({
            "topic_id": int(t_id),
            "total_calls": total,
            "won_count": won,
            "won_rate": round(won / total, 4) if total > 0 else 0.0,
            "lost_count": lost,
            "lost_rate": round(lost / total, 4) if total > 0 else 0.0,
            "no_decision_count": no_decision,
            "no_decision_rate": round(no_decision / total, 4) if total > 0 else 0.0
        })
    topic_df = pd.DataFrame(topic_analysis).sort_values(by="topic_id")
    topic_df.to_csv(os.path.join(output_dir, "topic_outcome_analysis.csv"), index=False)
    
    # 3. Cluster Outcome Splits
    cluster_analysis = []
    for c_id in df["cluster_id"].unique():
        c_group = df[df["cluster_id"] == c_id]
        total = len(c_group)
        won = len(c_group[c_group["outcome"] == "won"])
        lost = len(c_group[c_group["outcome"] == "lost"])
        no_decision = len(c_group[c_group["outcome"] == "no-decision"])
        
        cluster_analysis.append({
            "cluster_id": int(c_id),
            "total_calls": total,
            "won_count": won,
            "won_rate": round(won / total, 4) if total > 0 else 0.0,
            "lost_count": lost,
            "lost_rate": round(lost / total, 4) if total > 0 else 0.0,
            "no_decision_count": no_decision,
            "no_decision_rate": round(no_decision / total, 4) if total > 0 else 0.0
        })
    cluster_df = pd.DataFrame(cluster_analysis).sort_values(by="cluster_id")
    cluster_df.to_csv(os.path.join(output_dir, "cluster_outcome_analysis.csv"), index=False)
    
    logger.info("Statistical reports generated and exported successfully.")

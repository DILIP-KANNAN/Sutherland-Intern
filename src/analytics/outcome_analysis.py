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

def generate_detailed_cluster_profiles(
    features_df: pd.DataFrame, 
    topics: List[int], 
    clusters: List[int], 
    output_dir: str
):
    """
    Computes detailed summary metrics (mean, median, standard deviation) for key
    conversational features grouped by cluster and outcome, saved as nested JSON.
    """
    import json
    df = features_df.copy()
    df["topic_id"] = topics
    df["cluster_id"] = clusters
    
    key_features = [
        "talk_ratio", "num_turns", "call_duration_seconds", "customer_sentiment",
        "resolution_score", "num_objections", "discount_offered_pct", 
        "max_monologue_turns", "avg_turn_length"
    ]
    
    # Check what features exist
    features_to_calc = [f for f in key_features if f in df.columns]
    
    profiles = {}
    
    unique_clusters = sorted(df["cluster_id"].unique())
    unique_outcomes = ["won", "lost", "no-decision"]
    
    for c_id in unique_clusters:
        c_str = f"cluster_{c_id}"
        profiles[c_str] = {}
        c_group = df[df["cluster_id"] == c_id]
        
        for outcome in unique_outcomes:
            o_group = c_group[c_group["outcome"] == outcome]
            count = len(o_group)
            
            profiles[c_str][outcome] = {
                "total_calls": count,
                "metrics": {}
            }
            
            if count > 0:
                for col in features_to_calc:
                    mean_val = o_group[col].mean()
                    med_val = o_group[col].median()
                    std_val = o_group[col].std() if count > 1 else 0.0
                    
                    profiles[c_str][outcome]["metrics"][col] = {
                        "mean": round(float(mean_val), 4) if not pd.isna(mean_val) else 0.0,
                        "median": round(float(med_val), 4) if not pd.isna(med_val) else 0.0,
                        "std": round(float(std_val), 4) if not pd.isna(std_val) else 0.0
                    }
                    
    json_path = os.path.join(output_dir, "cluster_outcome_profiles.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    logger.info(f"Detailed cluster outcome profiles exported to {json_path}")

def export_decoupled_outcome_stats(features_df: pd.DataFrame, output_dir: str):
    """
    Filters and exports separate summary CSVs for each outcome type (won, lost, no-decision).
    """
    df = features_df.copy()
    unique_outcomes = ["won", "lost", "no-decision"]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["call_id"]]
    
    for outcome in unique_outcomes:
        o_group = df[df["outcome"] == outcome]
        summary_rows = []
        
        for col in numeric_cols:
            if len(o_group) > 0:
                mean_val = o_group[col].mean()
                median_val = o_group[col].median()
                std_val = o_group[col].std() if len(o_group) > 1 else 0.0
                min_val = o_group[col].min()
                max_val = o_group[col].max()
            else:
                mean_val = median_val = std_val = min_val = max_val = 0.0
                
            summary_rows.append({
                "feature": col,
                "mean": round(float(mean_val), 4) if not pd.isna(mean_val) else 0.0,
                "median": round(float(median_val), 4) if not pd.isna(median_val) else 0.0,
                "std": round(float(std_val), 4) if not pd.isna(std_val) else 0.0,
                "min": round(float(min_val), 4) if not pd.isna(min_val) else 0.0,
                "max": round(float(max_val), 4) if not pd.isna(max_val) else 0.0
            })
            
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, f"{outcome.replace('-', '_')}_outcome_stats.csv")
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Decoupled summary stats for outcome '{outcome}' saved to {csv_path}")


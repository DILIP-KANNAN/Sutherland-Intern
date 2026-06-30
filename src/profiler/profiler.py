import os
import json
import logging
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from src.features.schemas import CallRecord

logger = logging.getLogger("pattern_miner")

def profile_dataset(records: List[CallRecord], output_dir: str) -> Dict[str, Any]:
    """
    Analyzes the loaded dataset, calculates summary metrics, performs EDA,
    and saves JSON report and visualizations to the outputs directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert records to DataFrame for analysis
    data_dicts = [record.model_dump() for record in records]
    df = pd.DataFrame(data_dicts)
    
    total_calls = len(df)
    logger.info(f"Running profile analysis on {total_calls} records...")
    
    # Summary of outcomes
    outcome_counts = df['outcome'].value_counts().to_dict()
    outcome_pct = (df['outcome'].value_counts(normalize=True) * 100).to_dict()
    
    # Numerical summaries
    avg_turns = float(df['num_turns'].mean()) if 'num_turns' in df else 0.0
    avg_duration = float(df['call_duration_seconds'].mean()) if 'call_duration_seconds' in df else 0.0
    avg_sentiment = float(df['customer_sentiment_score'].mean()) if 'customer_sentiment_score' in df else 0.0
    avg_talk_ratio = float(df['agent_talk_ratio'].mean()) if 'agent_talk_ratio' in df else 0.0
    
    # Categorical counts
    objection_counts = df['objection_type'].value_counts().to_dict() if 'objection_type' in df else {}
    opener_quality = df['opener_quality'].value_counts().to_dict() if 'opener_quality' in df else {}
    agent_response_quality = df['agent_response_quality'].value_counts().to_dict() if 'agent_response_quality' in df else {}
    
    profile = {
        "corpus_summary": {
            "total_calls": total_calls,
            "outcome_counts": outcome_counts,
            "outcome_percentages": {k: round(v, 2) for k, v in outcome_pct.items()},
            "avg_call_length_minutes": round((avg_duration / 60.0), 2),
            "avg_turns": round(avg_turns, 2),
            "avg_customer_sentiment_score": round(avg_sentiment, 2),
            "avg_agent_talk_ratio": round(avg_talk_ratio, 2)
        },
        "objection_distribution": objection_counts,
        "opener_quality_distribution": opener_quality,
        "agent_response_quality_distribution": agent_response_quality
    }
    
    # Save profile JSON report
    profile_json_path = os.path.join(output_dir, "dataset_profile.json")
    with open(profile_json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    logger.info(f"Saved dataset profile JSON to {profile_json_path}")
    
    # EDA Plots
    # 1. Outcome Distribution
    plt.figure(figsize=(6, 4))
    colors = []
    for val in df['outcome'].value_counts().index:
        if val == 'won':
            colors.append('#2E7D32')  # dark green
        elif val == 'lost':
            colors.append('#C62828')  # dark red
        else:
            colors.append('#EF6C00')  # dark orange
            
    df['outcome'].value_counts().plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Call Outcome Distribution')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Call Duration vs Turn Count
    plt.figure(figsize=(7, 5))
    outcome_colors = {'won': '#2E7D32', 'lost': '#C62828', 'no-decision': '#EF6C00'}
    for outcome, group in df.groupby('outcome'):
        plt.scatter(
            group['num_turns'], 
            group['call_duration_seconds'], 
            label=outcome, 
            color=outcome_colors.get(outcome, 'gray'), 
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
    plt.title('Call Duration vs Turn Count by Outcome')
    plt.xlabel('Number of Turns')
    plt.ylabel('Call Duration (seconds)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "call_duration_vs_turns.png"), dpi=150)
    plt.close()
    
    # 3. Talk Ratio Distribution
    plt.figure(figsize=(6, 4))
    df['agent_talk_ratio'].plot(kind='hist', bins=20, color='#1976D2', edgecolor='black', alpha=0.7)
    plt.title('Agent Talk Ratio Distribution')
    plt.xlabel('Agent Talk Ratio')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "talk_ratio_distribution.png"), dpi=150)
    plt.close()
    
    logger.info("Saved EDA visualization plots successfully.")
    return profile

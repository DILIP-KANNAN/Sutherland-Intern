import os
import logging
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import umap

logger = logging.getLogger("pattern_miner")

def generate_visualizations(
    embeddings: np.ndarray, 
    features_df: pd.DataFrame, 
    clusters: List[int], 
    output_dir: str
):
    """
    Reduces call embeddings to 2D using UMAP, projects them on an interactive
    Plotly HTML map, and generates static feature comparisons split by outcome.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the dataframe and add cluster column
    df = features_df.copy()
    df["cluster"] = [f"Cluster {c}" for c in clusters]
    
    # 1. UMAP Dimensionality Reduction
    logger.info("Reducing call embeddings to 2D UMAP space...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    df["umap_x"] = embedding_2d[:, 0]
    df["umap_y"] = embedding_2d[:, 1]
    
    # 2. Interactive UMAP Plot (Plotly Express)
    logger.info("Generating interactive UMAP Plotly scatter map...")
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="cluster",
        symbol="outcome",
        hover_data=["call_id", "outcome", "cluster", "talk_ratio", "num_turns"],
        title="Interactive 2D UMAP Topic & Cluster Map",
        labels={"umap_x": "UMAP Dimension 1", "umap_y": "UMAP Dimension 2"},
        color_discrete_sequence=px.colors.qualitative.D3
    )
    
    fig.update_layout(
        legend_title_text='Groupings',
        template='plotly_white',
        hovermode="closest"
    )
    
    html_path = os.path.join(output_dir, "umap_plot.html")
    fig.write_html(html_path)
    logger.info(f"Interactive UMAP map exported successfully to {html_path}")
    
    # 3. Static UMAP Plot (Matplotlib)
    logger.info("Generating static UMAP scatter plot...")
    plt.figure(figsize=(8, 6))
    
    # Color mapping for clusters
    unique_clusters = sorted(df["cluster"].unique())
    cmap = plt.get_cmap("tab10")
    cluster_colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    
    # Marker mapping for outcomes
    outcome_markers = {'won': 'o', 'lost': 'X', 'no-decision': '^'}
    
    for outcome, out_group in df.groupby('outcome'):
        marker = outcome_markers.get(outcome, 'o')
        for cluster, cl_group in out_group.groupby('cluster'):
            plt.scatter(
                cl_group["umap_x"],
                cl_group["umap_y"],
                label=f"{cluster} ({outcome})",
                color=cluster_colors[cluster],
                marker=marker,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                s=60
            )
            
    plt.title("Static 2D UMAP Call Projections")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "umap_plot.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    
    # 4. Feature Comparison Boxplots
    logger.info("Generating outcome-split boxplots for features...")
    
    features_to_compare = [
        ("talk_ratio", "Agent Talk Ratio by Outcome", "talk_ratio_boxplot.png"),
        ("max_monologue_turns", "Maximum Monologue Turns by Outcome", "max_monologue_boxplot.png"),
        ("customer_sentiment", "Customer Sentiment Score by Outcome", "sentiment_boxplot.png"),
        ("discount_offered_pct", "Discount Offered Percentage by Outcome", "discount_boxplot.png")
    ]
    
    for col, title, filename in features_to_compare:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            
            # Map boxplot styling
            box_data = [group[col].values for name, group in df.groupby("outcome")]
            box_names = [name for name, group in df.groupby("outcome")]
            
            bp = plt.boxplot(box_data, tick_labels=box_names, patch_artist=True)

            
            for box in bp['boxes']:
                box.set(facecolor='#BBDEFB', color='#1976D2', linewidth=1.5)
            for median in bp['medians']:
                median.set(color='#D32F2F', linewidth=2.0)
                
            plt.title(title)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, filename), dpi=150)
            plt.close()
            
    logger.info("Successfully exported all visualizations.")

def generate_cluster_specific_plots(features_df: pd.DataFrame, clusters: List[int], output_dir: str):
    """
    For each cluster 0-5, exports a 2x2 multi-panel plot comparing conversational 
    features split by outcome (won, lost, no-decision).
    """
    df = features_df.copy()
    df["cluster_id"] = clusters
    
    unique_clusters = sorted(df["cluster_id"].unique())
    features_to_plot = [
        ("talk_ratio", "Agent Talk Ratio"),
        ("max_monologue_turns", "Max Monologue Turns"),
        ("customer_sentiment", "Customer Sentiment"),
        ("num_objections", "Number of Objections")
    ]
    
    color_map = {'won': '#BBDEFB', 'lost': '#FFCDD2', 'no-decision': '#FFE0B2'}
    border_map = {'won': '#1976D2', 'lost': '#D32F2F', 'no-decision': '#F57C00'}
    
    for c_id in unique_clusters:
        c_group = df[df["cluster_id"] == c_id]
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"Cluster {c_id} Conversational Profile by Outcome", fontsize=14, fontweight='bold')
        
        # Check outcomes present in cluster
        outcomes_present = [o for o in ['won', 'lost', 'no-decision'] if o in c_group['outcome'].values]
        
        for idx, (col, title) in enumerate(features_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            if col in c_group.columns and len(outcomes_present) > 0:
                # Prepare boxplot data grouped by outcome
                box_data = [c_group[c_group["outcome"] == o][col].values for o in outcomes_present]
                
                # Plot boxplot
                bp = ax.boxplot(box_data, tick_labels=outcomes_present, patch_artist=True)
                
                # Color code
                for patch, outcome in zip(bp['boxes'], outcomes_present):
                    patch.set(
                        facecolor=color_map.get(outcome, '#E0E0E0'), 
                        color=border_map.get(outcome, '#757575'), 
                        linewidth=1.5
                    )
                for median in bp['medians']:
                    median.set(color='black', linewidth=1.5)
            else:
                ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center')
                
            ax.set_title(title, fontsize=11, fontweight='semibold')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"cluster_{c_id}_profile.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        logger.info(f"Multi-panel plot saved for Cluster {c_id} to {png_path}")


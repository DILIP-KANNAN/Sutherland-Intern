import os
import argparse
import json
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.ingestion.loader import load_raw_dataset
from src.ingestion.validator import validate_dataset
from src.profiler.profiler import profile_dataset

def main():
    parser = argparse.ArgumentParser(description="Conversation Pattern Miner Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/insurance.yaml", 
        help="Path to domain configuration YAML"
    )
    parser.add_argument(
        "--phase", 
        type=int, 
        default=1, 
        help="Pipeline phase to execute (1-4)"
    )
    args = parser.parse_args()
    
    # Initialize logger
    logger = setup_logger("logs/pipeline.log")
    logger.info("Initializing Conversation Pattern Miner pipeline...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config} for domain: {config.get('domain', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return
        
    # Phase 1: Ingestion, Validation, and Profiling
    if args.phase >= 1:
        logger.info("--- Executing Phase 1: Ingestion, Validation, and Profiling ---")
        
        raw_path = config.get("data", {}).get("raw_path")
        if not raw_path:
            logger.error("No raw_path configured in settings.")
            return
            
        logger.info(f"Loading raw dataset from {raw_path}")
        try:
            raw_data = load_raw_dataset(raw_path)
            logger.info(f"Loaded {len(raw_data)} raw records.")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return
            
        # Empty dataset check
        if not raw_data:
            logger.warning("Empty corpus loaded. Raising warning and exiting early.")
            return
            
        # Validate dataset
        expected_outcomes = config.get("metadata", {}).get("outcome_labels")
        valid_records, invalid_records = validate_dataset(raw_data, expected_outcomes)
        
        # Save validated dataset to processed path
        processed_path = config.get("data", {}).get("processed_path")
        if not processed_path:
            logger.error("No processed_path configured in settings.")
            return
            
        logger.info(f"Saving validated dataset to {processed_path}")
        try:
            processed_dir = os.path.dirname(processed_path)
            if processed_dir:
                os.makedirs(processed_dir, exist_ok=True)
            with open(processed_path, "w", encoding="utf-8") as f:
                for record in valid_records:
                    f.write(json.dumps(record.model_dump()) + "\n")
            logger.info(f"Successfully saved {len(valid_records)} records to {processed_path}")
        except Exception as e:
            logger.error(f"Error saving validated corpus: {e}")
            return
            
        # Profile dataset & perform EDA
        output_dir = "outputs/profiler"
        try:
            profile_dataset(valid_records, output_dir)
            logger.info(f"Phase 1 completed successfully. Profile generated in {output_dir}")
        except Exception as e:
            logger.error(f"Error profiling dataset: {e}")
            return
            
    # Phase 2: Embeddings, Features & Topics
    if args.phase >= 2:
        logger.info("--- Executing Phase 2: Embeddings, Features, and Topics ---")
        
        # If valid_records is empty (e.g., if we run with --phase 2 directly), load it from processed_path
        if not valid_records:
            processed_path = config.get("data", {}).get("processed_path")
            if not processed_path:
                logger.error("No processed_path configured in settings.")
                return
            logger.info(f"Loading validated corpus from {processed_path}")
            try:
                raw_data = load_raw_dataset(processed_path)
                expected_outcomes = config.get("metadata", {}).get("outcome_labels")
                valid_records, _ = validate_dataset(raw_data, expected_outcomes)
            except Exception as e:
                logger.error(f"Error loading validated corpus for Phase 2: {e}")
                return
                
        # 1. Embeddings Module
        from src.embeddings.embedder import Embedder
        embedding_model = config.get("pipeline", {}).get("embedding_model", "all-MiniLM-L6-v2")
        try:
            embedder = Embedder(model_name=embedding_model)
            embeddings_path = "outputs/embeddings/call_embeddings.npy"
            embeddings = embedder.generate_embeddings(valid_records, output_path=embeddings_path)
            logger.info(f"Call embeddings step complete. Shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating call embeddings: {e}")
            return
            
        # 2. Features Module
        from src.features.extractor import extract_features
        try:
            features_path = "outputs/features/call_features.csv"
            features_df = extract_features(valid_records, output_path=features_path)
            logger.info(f"Conversational features step complete. Shape: {features_df.shape}")
        except Exception as e:
            logger.error(f"Error extracting conversational features: {e}")
            return
            
        # 3. Topic Modeling Module (BERTopic)
        from src.topics.topic_model import TopicModel
        min_topic_size = config.get("pipeline", {}).get("topics", {}).get("min_topic_size", 5)
        try:
            topic_model = TopicModel(min_topic_size=min_topic_size, nr_topics=8)
            topics, topic_info = topic_model.fit_transform(valid_records, embeddings=embeddings)
            topics_dir = "outputs/topics"
            topic_model.save_model(topics_dir, topics, topic_info)
            logger.info(f"Topic modeling step complete. Assigned {len(topics)} calls to 8 topics.")
        except Exception as e:
            logger.error(f"Error running BERTopic model: {e}")
            return
            
        logger.info("Phase 2 execution finished successfully.")

    # Phase 3: Clustering & Outcome Analysis
    if args.phase >= 3:
        logger.info("--- Executing Phase 3: Clustering & Outcome Analysis ---")
        
        import numpy as np
        import pandas as pd
        
        # Load necessary objects if they are not in memory (useful if running with --phase 3 directly)
        if not valid_records:
            processed_path = config.get("data", {}).get("processed_path")
            if not processed_path:
                logger.error("No processed_path configured in settings.")
                return
            logger.info(f"Loading validated corpus from {processed_path}")
            try:
                raw_data = load_raw_dataset(processed_path)
                expected_outcomes = config.get("metadata", {}).get("outcome_labels")
                valid_records, _ = validate_dataset(raw_data, expected_outcomes)
            except Exception as e:
                logger.error(f"Error loading validated corpus: {e}")
                return
                
        embeddings_path = "outputs/embeddings/call_embeddings.npy"
        if 'embeddings' not in locals() or embeddings is None:
            if os.path.exists(embeddings_path):
                logger.info(f"Loading cached embeddings from {embeddings_path}")
                embeddings = np.load(embeddings_path)
            else:
                logger.error(f"Call embeddings not found at {embeddings_path}. Please run Phase 2 first.")
                return
                
        features_path = "outputs/features/call_features.csv"
        if 'features_df' not in locals() or features_df is None:
            if os.path.exists(features_path):
                logger.info(f"Loading features DataFrame from {features_path}")
                features_df = pd.read_csv(features_path)
            else:
                logger.error(f"Conversational features not found at {features_path}. Please run Phase 2 first.")
                return
                
        topics_json_path = "outputs/topics/topic_assignments.json"
        if 'topics' not in locals() or topics is None:
            if os.path.exists(topics_json_path):
                logger.info(f"Loading topic assignments from {topics_json_path}")
                with open(topics_json_path, "r", encoding="utf-8") as f:
                    topics_data = json.load(f)
                topics = topics_data.get("topic_assignments", [])
            else:
                logger.error(f"Topic assignments not found at {topics_json_path}. Please run Phase 2 first.")
                return
                
        # 1. KMeans Clustering
        from src.clustering.clusterer import Clusterer
        n_clusters = config.get("pipeline", {}).get("clustering", {}).get("n_clusters", 6)
        cluster_assignments_path = "outputs/clusters/cluster_assignments.json"
        try:
            clusterer = Clusterer(n_clusters=n_clusters)
            clusters = clusterer.fit_predict(embeddings)
            clusterer.save_assignments(cluster_assignments_path, clusters)
            logger.info(f"Clustering complete. Assigned {len(clusters)} calls into {n_clusters} clusters.")
        except Exception as e:
            logger.error(f"Error executing KMeans clustering: {e}")
            return
            
        # 2. Outcome Analysis
        from src.analytics.outcome_analysis import (
            perform_outcome_analysis,
            generate_detailed_cluster_profiles,
            export_decoupled_outcome_stats
        )
        analytics_dir = "outputs/analytics"
        try:
            perform_outcome_analysis(features_df, topics, list(clusters), analytics_dir)
            generate_detailed_cluster_profiles(features_df, topics, list(clusters), analytics_dir)
            export_decoupled_outcome_stats(features_df, analytics_dir)
            logger.info(f"Outcome correlation analysis complete. Reports saved in {analytics_dir}")
        except Exception as e:
            logger.error(f"Error performing outcome analysis: {e}")
            return
            
        # 3. Visualizations
        from src.visualization.visualizer import (
            generate_visualizations,
            generate_cluster_specific_plots
        )
        visualizations_dir = "outputs/visualizations"
        try:
            generate_visualizations(embeddings, features_df, list(clusters), visualizations_dir)
            generate_cluster_specific_plots(features_df, list(clusters), visualizations_dir)
            logger.info(f"Visualization plots generated successfully in {visualizations_dir}")
        except Exception as e:
            logger.error(f"Error generating visual plots: {e}")
            return

            
        # 4. Dual-Method Segment Labeling
        from src.topics.labeler import generate_and_save_labels
        labels_output_path = "outputs/topics/labels_comparison.json"
        
        # Load topic definitions
        topic_definitions = {}
        if os.path.exists(topics_json_path):
            try:
                with open(topics_json_path, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                topic_definitions = t_data.get("topic_definitions", {})
            except Exception as e:
                logger.warning(f"Could not load topic definitions from {topics_json_path}: {e}")
                
        try:
            generate_and_save_labels(
                valid_records,
                embeddings,
                features_df,
                topics,
                list(clusters),
                topic_definitions,
                labels_output_path
            )
            logger.info(f"Segment labeling complete. Comparison matrix saved to {labels_output_path}")
        except Exception as e:
            logger.error(f"Error executing segment labeling: {e}")
            return
            
        logger.info("Phase 3 execution finished successfully.")


if __name__ == "__main__":
    main()



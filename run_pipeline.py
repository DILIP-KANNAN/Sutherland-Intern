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

if __name__ == "__main__":
    main()


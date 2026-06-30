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

if __name__ == "__main__":
    main()

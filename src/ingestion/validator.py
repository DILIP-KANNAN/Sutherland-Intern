import logging
from typing import List, Dict, Any, Tuple
from src.features.schemas import CallRecord
from pydantic import ValidationError

logger = logging.getLogger("pattern_miner")

def validate_dataset(
    data: List[Dict[str, Any]], 
    expected_outcome_labels: List[str] = None
) -> Tuple[List[CallRecord], List[Dict[str, Any]]]:
    """
    Validates loaded raw data items against the CallRecord Pydantic schema.
    Returns (list of valid CallRecord objects, list of invalid dictionaries).
    """
    valid_records = []
    invalid_records = []
    
    for i, item in enumerate(data):
        call_id = item.get("call_id", f"UNKNOWN_RECORD_INDEX_{i}")
        try:
            # Parse and validate with Pydantic
            record = CallRecord(**item)
            
            # Validate domain-specific expectations
            if expected_outcome_labels and record.outcome not in expected_outcome_labels:
                logger.warning(
                    f"Call {call_id} validation failed: outcome '{record.outcome}' "
                    f"is not one of expected labels {expected_outcome_labels}."
                )
                invalid_records.append(item)
                continue
                
            valid_records.append(record)
        except ValidationError as e:
            logger.warning(f"Call {call_id} failed validation schema: {e.errors()}")
            invalid_records.append(item)
            
    logger.info(f"Dataset validation summary: {len(valid_records)} valid, {len(invalid_records)} invalid records.")
    return valid_records, invalid_records

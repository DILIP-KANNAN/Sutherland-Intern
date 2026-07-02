import logging
import pandas as pd
from typing import List, Dict, Any
from src.features.schemas import CallRecord
from src.preprocessing.cleaner import parse_transcript_summary

logger = logging.getLogger("pattern_miner")

def extract_features(records: List[CallRecord], output_path: str = None) -> pd.DataFrame:
    """
    Extracts at least 12 (specifically 16) conversational and metadata features 
    for each call and exports them as a pandas DataFrame.
    """
    logger.info(f"Extracting features for {len(records)} calls...")
    
    extracted_data = []
    
    for record in records:
        # Parse transcript turns
        turns = parse_transcript_summary(record.transcript_summary)
        
        # 1. Initialize text turn feature accumulators
        agent_question_count = 0
        customer_question_count = 0
        total_chars = 0
        
        max_monologue = 0
        current_speaker = None
        current_run = 0
        
        for turn in turns:
            text = turn.get("text", "")
            speaker = turn.get("speaker", "")
            
            total_chars += len(text)
            
            # Question counts
            if speaker == "Agent":
                agent_question_count += text.count("?")
            elif speaker == "Customer":
                customer_question_count += text.count("?")
                
            # Monologue turns tracking
            if current_speaker == speaker:
                current_run += 1
            else:
                max_monologue = max(max_monologue, current_run)
                current_speaker = speaker
                current_run = 1
                
        max_monologue = max(max_monologue, current_run)
        
        # Avoid division by zero
        avg_turn_length = (total_chars / len(turns)) if len(turns) > 0 else 0.0
        
        # 2. Build feature dictionary (16 distinct features)
        features = {
            "call_id": record.call_id,
            "outcome": record.outcome,
            
            # Metadata-driven features
            "talk_ratio": record.agent_talk_ratio if record.agent_talk_ratio is not None else 0.0,
            "num_turns": record.num_turns if record.num_turns is not None else 0,
            "call_duration_seconds": record.call_duration_seconds if record.call_duration_seconds is not None else 0,
            "customer_sentiment": record.customer_sentiment_score if record.customer_sentiment_score is not None else 0.0,
            "resolution_score": record.resolution_score if record.resolution_score is not None else 0.0,
            "num_objections": record.num_objections_raised if record.num_objections_raised is not None else 0,
            "prior_claims": record.prior_claims if record.prior_claims is not None else 0,
            "years_as_customer": record.years_as_customer if record.years_as_customer is not None else 0,
            "prior_premium_gbp": record.prior_premium_gbp if record.prior_premium_gbp is not None else 0.0,
            "quoted_premium_gbp": record.quoted_premium_gbp if record.quoted_premium_gbp is not None else 0.0,
            "premium_change_pct": record.premium_change_pct if record.premium_change_pct is not None else 0.0,
            "discount_offered_pct": record.discount_offered_pct if record.discount_offered_pct is not None else 0.0,
            
            # Transcript-driven parsed features
            "agent_question_count": agent_question_count,
            "customer_question_count": customer_question_count,
            "avg_turn_length": round(avg_turn_length, 2),
            "max_monologue_turns": max_monologue
        }
        
        extracted_data.append(features)
        
    df = pd.DataFrame(extracted_data)
    
    if output_path:
        out_dir = pd.io.common.os.path.dirname(output_path)
        if out_dir:
            pd.io.common.os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved feature DataFrame of shape {df.shape} to {output_path}")
        
    return df

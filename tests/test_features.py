import pandas as pd
from src.features.schemas import CallRecord
from src.features.extractor import extract_features

def test_extract_features():
    # Setup record with questions and consecutive turns
    records = [
        CallRecord(
            call_id="CALL-1",
            outcome="won",
            transcript_summary="[T1] Agent: \"Hello?\" | [T2] Customer: \"Hi? How are you?\" | [T3] Customer: \"I need help.\"",
            num_turns=3,
            call_duration_seconds=120,
            agent_talk_ratio=0.4,
            customer_sentiment_score=0.8,
            resolution_score=0.9,
            num_objections_raised=0,
            prior_claims=1,
            years_as_customer=2,
            prior_premium_gbp=300.0,
            quoted_premium_gbp=320.0,
            premium_change_pct=6.7,
            discount_offered_pct=0.0
        )
    ]
    
    df = extract_features(records)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "agent_question_count" in df.columns
    assert "customer_question_count" in df.columns
    assert "max_monologue_turns" in df.columns
    
    # Check text parsed features
    assert df.loc[0, "agent_question_count"] == 1       # T1 has '?'
    assert df.loc[0, "customer_question_count"] == 2    # T2 has two '?'
    assert df.loc[0, "max_monologue_turns"] == 2        # T2 and T3 are both Customer (consecutive)
    assert len(df.columns) == 18                        # 16 features + call_id + outcome keys

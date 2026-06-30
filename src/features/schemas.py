from pydantic import BaseModel, Field
from typing import List, Optional

class TranscriptTurn(BaseModel):
    """
    Represents a single turn in a transcript.
    Example: [T1] Agent: "Hello, just a call..."
    """
    turn_id: str
    speaker: str
    text: str

class CallRecord(BaseModel):
    """
    Schema for a call record loaded from the corpus.
    """
    call_id: str
    date: Optional[str] = None
    shift: Optional[str] = None
    channel: Optional[str] = None
    region: Optional[str] = None
    insurance_type: Optional[str] = None
    agent_name: Optional[str] = None
    customer_name: Optional[str] = None
    years_as_customer: Optional[int] = None
    prior_claims: Optional[int] = None
    prior_premium_gbp: Optional[float] = None
    quoted_premium_gbp: Optional[float] = None
    premium_change_pct: Optional[float] = None
    discount_offered_pct: Optional[float] = None
    final_premium_gbp: Optional[float] = None
    scenario_type: Optional[str] = None
    outcome: str
    contact_attempt_number: Optional[int] = None
    call_duration_seconds: Optional[int] = None
    num_turns: Optional[int] = None
    num_objections_raised: Optional[int] = None
    opener_quality: Optional[str] = None
    agent_response_quality: Optional[str] = None
    discovery_question_used: Optional[bool] = None
    discount_offered: Optional[bool] = None
    upsell_attempted: Optional[bool] = None
    objection_type: Optional[str] = None
    agent_talk_ratio: Optional[float] = None
    customer_sentiment_score: Optional[float] = None
    resolution_score: Optional[float] = None
    transcript_summary: str

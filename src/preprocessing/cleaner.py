import re
from typing import List, Dict

# Regex pattern to match: [T1] Agent: "Hello..." or [T1] Agent: Hello...
TURN_PATTERN = re.compile(r"^\[(T\d+)\]\s+([^:]+):\s*\"?(.*?)\"?$")

def parse_transcript_summary(transcript_summary: str) -> List[Dict[str, str]]:
    """
    Parses a concatenated transcript summary into a list of individual turns.
    Example:
      "[T1] Agent: \"Hello\" | [T2] Customer: \"Hi\""
      -> [{"turn_id": "T1", "speaker": "Agent", "text": "Hello"}, ...]
    """
    if not transcript_summary:
        return []
        
    turns = []
    raw_turns = transcript_summary.split(" | ")
    for raw_turn in raw_turns:
        raw_turn = raw_turn.strip()
        if not raw_turn:
            continue
        match = TURN_PATTERN.match(raw_turn)
        if match:
            turn_id, speaker, text = match.groups()
            turns.append({
                "turn_id": turn_id,
                "speaker": speaker.strip(),
                "text": text.strip().strip('"').strip("'")
            })
            
    return turns

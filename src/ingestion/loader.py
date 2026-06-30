import json
import os
from typing import List, Dict, Any

def load_raw_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads raw datasets. Supports both standard JSON lists and JSONL (JSON lines) formats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    if not content:
        return []
        
    # Attempt to parse as standard JSON list first
    if content.startswith("["):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
            
    # Fallback: Parse as JSONL (JSON lines)
    data = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
            
    return data

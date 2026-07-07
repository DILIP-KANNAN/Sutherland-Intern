import os
import re
import time
import json
import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("pattern_miner")

def load_dotenv_custom(dotenv_path: str = ".env"):
    """
    Manually parses the local .env file if it exists and sets environment variables.
    """
    if os.path.exists(dotenv_path):
        logger.info(f"Custom parsing local env file at: {dotenv_path}")
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                val = val.strip().strip("'").strip('"')
                os.environ[key.strip()] = val
                
load_dotenv_custom()

def anonymize_text(text: str) -> str:
    """
    Anonymizes sensitive details in conversation text such as emails, phone numbers,
    policy numbers, postcodes, and greetings that mention names.
    """
    if not isinstance(text, str):
        return ""
        
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", text)
    text = re.sub(r"\b\d{3,5}[-\s]?\d{3,4}[-\s]?\d{3,4}\b", "[PHONE]", text)
    text = re.sub(r"\b[A-Z]{3,4}-\d{5,10}\b", "[ID_REF]", text)
    text = re.sub(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", "[POSTCODE]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{5}(-\d{4})?\b", "[POSTCODE]", text)
    text = re.sub(r"(?i)\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", "my name is [NAME]", text)
    text = re.sub(r"(?i)\bthis is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", "this is [NAME]", text)
    text = re.sub(r"(?i)\bspeaking to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", "speaking to [NAME]", text)
    
    return text

class HeuristicLabeler:
    @staticmethod
    def generate_label(keywords: List[str], win_rate: float, lost_rate: float) -> str:
        """
        Extracts semantic titles based on keyword sets and outcome splits.
        """
        kw_set = {w.lower() for w in keywords}
        category = "General Dialogue"
        
        if kw_set.intersection({"confirm", "confirmation", "done", "address", "email", "setup", "postcode"}):
            category = "Verification & Closure"
        elif kw_set.intersection({"premium", "quote", "price", "monthly", "cost", "annual", "competitor", "match", "cheaper"}):
            category = "Price & Renewal Negotiation"
        elif kw_set.intersection({"speaking", "identity", "policyholder", "husband", "wife", "verify", "verification"}):
            category = "Identity Check"
        elif kw_set.intersection({"renew", "renewal", "extend", "loyalty"}):
            category = "Renewal Retention"
            
        if win_rate >= 0.80:
            outcome_desc = "High-Conversion"
        elif lost_rate >= 0.80:
            outcome_desc = "High-Objection/Churn"
        elif win_rate < 0.20 and lost_rate < 0.20:
            outcome_desc = "Deferred/Follow-up"
        else:
            outcome_desc = "Mixed/Standard"
            
        top_kws = ", ".join(keywords[:2])
        return f"{outcome_desc} {category} ({top_kws})"

class LLMLabeler:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            logger.info("GEMINI_API_KEY found. Configuring LLMLabeler...")
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                logger.error(f"Error configuring Gemini SDK: {e}")
                self.model = None
        else:
            logger.warning("GEMINI_API_KEY environment variable is unset. LLM labeling will run in fallback mode.")
            
    def generate_batch_labels(self, items: List[Dict[str, Any]], segment_type: str) -> Dict[str, str]:
        """
        Generates labels in batch mode to avoid hitting the 5 RPM free-tier rate limits.
        Sends a single prompt for all items and returns a mapping dictionary of segment ID -> label.
        """
        if not self.model:
            return {str(item["id"]): "LLM Label Unavailable (GEMINI_API_KEY unset)" for item in items}
            
        logger.info(f"Batching {len(items)} {segment_type} labeling requests into a single Gemini prompt...")
        
        # Format the items for the prompt
        prompt_items = []
        for item in items:
            safe_examples = [anonymize_text(ex) for ex in item["examples"][:2]]
            prompt_items.append({
                "id": str(item["id"]),
                "keywords": item["keywords"],
                "win_rate": f"{item['win_rate']:.2%}",
                "loss_rate": f"{item['lost_rate']:.2%}",
                "examples": safe_examples
            })
            
        prompt = f"""
        You are an expert customer conversation analyst.
        Your task is to assign a concise, professional semantic label of 3 to 5 words to each of the following {segment_type} groups.
        
        Groups to label:
        {json.dumps(prompt_items, indent=2)}
        
        Requirements for each label:
        - The label must be brief (3 to 5 words).
        - It should clearly summarize the primary theme (e.g., 'Verification & Payment Close' or 'Spouse Deferrals & Follow-ups').
        - Do not use quotes or special characters inside the label strings.
        
        You MUST return ONLY a valid JSON object matching the following structure, with no extra text or explanations:
        {{
          "labels": {{
            "0": "Label text for group 0",
            "1": "Label text for group 1",
            ...
          }}
        }}
        """
        
        try:
            # Let's request JSON output structure
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            raw_response = response.text.strip()
            
            # Parse response
            parsed = json.loads(raw_response)
            labels = parsed.get("labels", {})
            
            # Map all strings back
            result = {}
            for item in items:
                idx = str(item["id"])
                result[idx] = labels.get(idx, f"LLM Generation Missed ID {idx}")
            return result
        except Exception as e:
            logger.error(f"Gemini batch API call failed for {segment_type}s: {e}")
            # Return fallback indicator
            return {str(item["id"]): f"LLM Call Failed ({str(e)[:50]})" for item in items}

def generate_and_save_labels(
    valid_records: List[Any],
    embeddings: np.ndarray,
    features_df: pd.DataFrame,
    topics: List[int],
    clusters: List[int],
    topic_definitions: Dict[str, List[Dict[str, Any]]],
    output_path: str
):
    """
    Orchestrates batch labeling of segments, generating both heuristic
    and LLM labels, and saving the results comparison dictionary.
    """
    logger.info("Initializing segment labeling pipeline...")
    
    df = features_df.copy()
    df["topic_id"] = topics
    df["cluster_id"] = clusters
    df["transcript"] = [rec.transcript_summary for rec in valid_records]
    
    llm_labeler = LLMLabeler()
    
    output_data = {
        "topics": {},
        "clusters": {}
    }
    
    # --- Part 1: Gather Topic Data for Batch ---
    topic_batch_items = []
    unique_topics = sorted(list(set(topics)))
    
    for t_id in unique_topics:
        t_group = df[df["topic_id"] == t_id]
        total = len(t_group)
        
        win_rate = len(t_group[t_group["outcome"] == "won"]) / total if total > 0 else 0.0
        lost_rate = len(t_group[t_group["outcome"] == "lost"]) / total if total > 0 else 0.0
        
        # Get keywords
        if str(t_id) in topic_definitions:
            kw_list = [w["word"] for w in topic_definitions[str(t_id)][:5]]
        else:
            if total > 0:
                try:
                    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
                    vectorizer.fit(t_group["transcript"])
                    kw_list = list(vectorizer.get_feature_names_out())
                except:
                    kw_list = ["call", "transcript"]
            else:
                kw_list = ["empty", "topic"]
                
        examples = t_group["transcript"].head(2).tolist()
        
        topic_batch_items.append({
            "id": t_id,
            "keywords": kw_list,
            "win_rate": win_rate,
            "lost_rate": lost_rate,
            "examples": examples
        })
        
    # Generate LLM Batch Labels for Topics
    topic_llm_labels = llm_labeler.generate_batch_labels(topic_batch_items, "topic")
    
    # Merge and populate Heuristics
    for item in topic_batch_items:
        t_id = item["id"]
        h_label = HeuristicLabeler.generate_label(item["keywords"], item["win_rate"], item["lost_rate"])
        l_label = topic_llm_labels.get(str(t_id), "LLM Label Not Found")
        
        output_data["topics"][str(t_id)] = {
            "keywords": item["keywords"],
            "total_calls": len(df[df["topic_id"] == t_id]),
            "win_rate": round(item["win_rate"], 4),
            "heuristic_label": h_label,
            "llm_label": l_label
        }
        logger.info(f"Topic {t_id} Heuristic: '{h_label}' | LLM: '{l_label}'")
        
    # --- Part 2: Gather Cluster Data for Batch ---
    # Ensure sleep between batch calls to stay safe on free tier rate limits
    if llm_labeler.model:
        time.sleep(2.0)
        
    cluster_batch_items = []
    unique_clusters = sorted(list(set(clusters)))
    
    for c_id in unique_clusters:
        c_group = df[df["cluster_id"] == c_id]
        total = len(c_group)
        
        win_rate = len(c_group[c_group["outcome"] == "won"]) / total if total > 0 else 0.0
        lost_rate = len(c_group[c_group["outcome"] == "lost"]) / total if total > 0 else 0.0
        
        # Get keywords via TF-IDF
        if total > 0:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
                vectorizer.fit(c_group["transcript"])
                kw_list = list(vectorizer.get_feature_names_out())
            except:
                kw_list = ["group", "cluster"]
        else:
            kw_list = ["empty", "cluster"]
            
        examples = c_group["transcript"].head(2).tolist()
        
        cluster_batch_items.append({
            "id": c_id,
            "keywords": kw_list,
            "win_rate": win_rate,
            "lost_rate": lost_rate,
            "examples": examples
        })
        
    # Generate LLM Batch Labels for Clusters
    cluster_llm_labels = llm_labeler.generate_batch_labels(cluster_batch_items, "cluster")
    
    # Merge and populate Heuristics
    for item in cluster_batch_items:
        c_id = item["id"]
        h_label = HeuristicLabeler.generate_label(item["keywords"], item["win_rate"], item["lost_rate"])
        l_label = cluster_llm_labels.get(str(c_id), "LLM Label Not Found")
        
        output_data["clusters"][str(c_id)] = {
            "keywords": item["keywords"],
            "total_calls": len(df[df["cluster_id"] == c_id]),
            "win_rate": round(item["win_rate"], 4),
            "heuristic_label": h_label,
            "llm_label": l_label
        }
        logger.info(f"Cluster {c_id} Heuristic: '{h_label}' | LLM: '{l_label}'")
        
    # Save outputs
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    logger.info(f"Label comparison matrix saved successfully to {output_path}")

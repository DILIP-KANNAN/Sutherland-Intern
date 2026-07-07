from src.topics.labeler import anonymize_text, HeuristicLabeler, LLMLabeler

def test_anonymize_text():
    raw_text = "Hello, my name is John Smith. My email is john.smith@company.co.uk and my number is 07912 345 678. My policy reference is POL-98765."
    clean_text = anonymize_text(raw_text)
    
    assert "[EMAIL]" in clean_text
    assert "[PHONE]" in clean_text
    assert "[ID_REF]" in clean_text
    assert "John Smith" not in clean_text or "[NAME]" in clean_text
    
def test_heuristic_labeler():
    keywords = ["confirm", "address", "postcode", "email"]
    label = HeuristicLabeler.generate_label(keywords, win_rate=0.9, lost_rate=0.0)
    
    assert "High-Conversion" in label
    assert "Verification & Closure" in label
    assert "confirm" in label
    
def test_llm_labeler_fallback():
    # If GEMINI_API_KEY is unset, it should return fallback label mapping and not crash
    labeler = LLMLabeler()
    items = [{"id": 0, "keywords": ["test"], "win_rate": 0.5, "lost_rate": 0.5, "examples": ["Hello test"]}]
    labels = labeler.generate_batch_labels(items, "test")
    
    if not labeler.api_key:
        assert "GEMINI_API_KEY unset" in labels["0"]


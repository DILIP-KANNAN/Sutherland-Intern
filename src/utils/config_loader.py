import os
import yaml

def load_config(domain_config_path: str = None) -> dict:
    """
    Loads and merges default and domain-specific configuration files.
    """
    # Find project base directory (parent of src folder)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_config_path = os.path.join(base_dir, "configs", "default.yaml")
    
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default config not found at {default_config_path}")
        
    with open(default_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
        
    if domain_config_path:
        # Resolve path relative to base directory if not absolute
        if not os.path.isabs(domain_config_path):
            resolved_domain_path = os.path.join(base_dir, domain_config_path)
            # If not found directly, try relative to configs directory
            if not os.path.exists(resolved_domain_path):
                resolved_domain_path = os.path.join(base_dir, "configs", domain_config_path)
        else:
            resolved_domain_path = domain_config_path
            
        if os.path.exists(resolved_domain_path):
            with open(resolved_domain_path, "r", encoding="utf-8") as f:
                domain_config = yaml.safe_load(f) or {}
                # Merge dictionaries
                for key, val in domain_config.items():
                    if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                        config[key].update(val)
                    else:
                        config[key] = val
        else:
            raise FileNotFoundError(f"Domain config file not found at {resolved_domain_path}")
            
    return config

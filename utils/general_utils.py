"""
General utility functions for CI-LLM project.
Provides configuration loading and other common utilities.
"""

import os
import yaml
from typing import Dict, Optional


def load_config(config_path: str, override_config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file(s).
    
    Args:
        config_path: Path to the main configuration YAML file
        override_config_path: Optional path to an override configuration file
        
    Returns:
        Configuration dictionary with merged settings
        
    Raises:
        FileNotFoundError: If the main config file doesn't exist
        yaml.YAMLError: If there's an error parsing the YAML files
    """
    # Load main configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # Load and merge override configuration if provided
    if override_config_path and os.path.exists(override_config_path):
        with open(override_config_path, 'r') as f:
            override_config = yaml.safe_load(f)
        
        if override_config:
            # Merge override config into main config
            config = merge_configs(config, override_config)
            print(f"Applied configuration overrides from: {override_config_path}")
    
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Recursively merge override configuration into base configuration.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override the value
            merged[key] = value
    
    return merged


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory to create
    """
    os.makedirs(directory_path, exist_ok=True)


def get_project_root() -> str:
    """
    Get the project root directory path.
    
    Returns:
        Absolute path to the project root
    """
    # Assume this file is in utils/ subdirectory of project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def validate_config_paths(config: Dict) -> None:
    """
    Validate that required paths in configuration exist or can be created.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required paths are invalid
    """
    # Check if dataset path exists (for input data)
    dataset_path = config.get('dataset_name')
    if dataset_path and not os.path.exists(dataset_path):
        print(f"Warning: Dataset path does not exist: {dataset_path}")
    
    # Ensure output directories can be created
    output_paths = [
        config.get('processed_data_path_train'),
        config.get('processed_data_path_eval'),
        config.get('output_dir')
    ]
    
    for path in output_paths:
        if path:
            # Get parent directory and ensure it exists
            parent_dir = os.path.dirname(path)
            if parent_dir:
                ensure_directory_exists(parent_dir) 
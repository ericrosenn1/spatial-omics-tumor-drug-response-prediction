"""
config.py

Purpose:
Load and validate YAML configuration files for the spatial feature identification
pipeline.
"""


# =========================
# Imports
# =========================

from pathlib import Path
import yaml



# =========================
# Config loading
# =========================

def load_config(config_path):
    """Load a YAML config file and convert key paths to Path objects."""
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["config_path"] = config_path
    cfg["input_root"] = Path(cfg["input_root"])
    cfg["output_root"] = Path(cfg["output_root"])

    if "sample_glob" not in cfg:
        cfg["sample_glob"] = "SAMPLE_*"

    if "steps" not in cfg:
        cfg["steps"] = {}

    return cfg



# =========================
# Config validation
# =========================

def validate_config(cfg):
    """Check that required config fields exist and input paths are valid."""
    required = ["input_root", "output_root", "sample_glob"]

    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    if not cfg["input_root"].exists():
        raise FileNotFoundError(f"Input root does not exist: {cfg['input_root']}")

    cfg["output_root"].mkdir(parents=True, exist_ok=True)

    return cfg


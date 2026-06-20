"""Module for loading configuration from different sources."""

import tomllib
from typing import Any


def load_toml(file_path: str, **overrides: Any) -> dict[str, Any]:
    """Load configuration from a TOML file and apply overrides."""
    with open(file_path, "rb") as f:
        config = tomllib.load(f)

    # Apply overrides
    for key, value in overrides.items():
        config[key] = value

    return config

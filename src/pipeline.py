"""High-level pipeline orchestration module."""
from .training import run_pipeline


def run(overrides: dict):
    return run_pipeline(overrides)

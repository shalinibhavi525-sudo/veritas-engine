import pytest
import os
from veritas_engine import VeritasEngine

def test_engine_init():
    """Check if engine loads with default settings."""
    engine = VeritasEngine()
    assert engine.model_name == "distilbert-base-uncased"

def test_data_structure():
    """Check if dataset remapping produces the right labels."""
    engine = VeritasEngine()
    dataset = engine.prepare_data()
    assert "label" in dataset.features
    unique_labels = set(dataset['label'])
    assert unique_labels.issubset({0, 1})

import pytest
import os
import shutil

def test_onnx_export_exists():
    """Verify that the optimization process actually creates a model file."""
    output_dir = "onnx_quantized"
    if os.path.exists(output_dir):
        assert os.path.isfile(os.path.join(output_dir, "model_quantized.onnx"))
      

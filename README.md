# Veritas Engine: Hardware-Aware Transformer Optimization 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Optimization: ONNX Runtime](https://img.shields.io/badge/Optimization-ONNX%20Runtime-0078D4.svg)](https://onnxruntime.ai/)

**Veritas Engine** is a specialized optimization pipeline designed to compress Transformer-based models for real-time, client-side inference. It specifically targets the "Last Mile" of internet connectivity, moving away from cloud-dependent APIs toward high-speed, 8-bit integer (INT8) local models.

---

## 🌍 The Problem: The "Connectivity-Blind" Gap
In regions with unstable network infrastructure (such as the reserved forests of Tripura, India), cloud-based AI verification is often unusable. Veritas Engine bridges this digital divide by optimizing state-of-the-art NLP models to run locally on consumer-grade CPUs.

### Key Performance Benchmarks:
| Metric | Baseline (DistilBERT FP32) | Veritas Optimized (INT8 ONNX) |
| :--- | :--- | :--- |
| **Model Size** | 255.45 MB | **64.45 MB** (75% smaller) |
| **Latency** | 52.73 ms | **23.58 ms** (55% faster) |
| **Accuracy** | 62.77% | **61.99%** (98.7% retained) |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/veritas-engine.git
cd veritas-engine

# Install required libraries
pip install transformers optimum[onnxruntime] torch pandas datasets matplotlib seaborn
```

---

## 🖥️ Usage

### 1. Run the Optimization Pipeline
The main script `veritas_engine.py` handles the training, quantization, and ONNX export.

```python
from veritas_engine import VeritasEngine

# Initialize the engine
engine = VeritasEngine(model_name="distilbert-base-uncased")

# Step 1: Prepare the dataset
dataset = engine.prepare_data()

# Step 2: Execute optimization (Quantization + ONNX Export)
engine.optimize()
```

### 2. Verify Local Inference
Once optimized, you can run the engine locally using ONNX Runtime:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Load the 64MB squeezed model
model = ORTModelForSequenceClassification.from_pretrained("onnx_quantized", file_name="model_quantized.onnx")
tokenizer = AutoTokenizer.from_pretrained("./my_fine_tuned_model")

# Test a claim
inputs = tokenizer("The unemployment rate dropped this month.", return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
```

---

## 📊 Methodology
The Veritas Engine follows a three-stage compression architecture:
1. **Fine-Tuning:** DistilBERT is trained on remapped binary labels from the LIAR dataset.
2. **INT8 Quantization:** Linear layers are dynamically quantized to 8-bit precision to save 75% space.
3. **ONNX Graph Optimization:** The model is converted to a static computation graph for hardware acceleration (AVX512_VNNI).

---

## ✍️ Author
**Shambhavi Singh** - Independent Researcher  
[Email](mailto:shalinibhavi525@gmail.com) | [ORCID](https://orcid.org/0009-0009-9448-5638)

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

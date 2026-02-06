# veritas-engine
# Veritas: AI for the "Last Mile" 🌍🏛️

[![Status](https://img.shields.io/badge/JOSS-Under%20Review-blue)](https://joss.theoj.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Veritas** is a hardware-aware optimization engine designed to democratize misinformation detection. It compresses Large Language Models (LLMs) into lightweight, offline-capable artifacts that run on standard consumer CPUs—specifically designed for users in "connectivity-blind" regions.

## 🚀 The Mission: Democratizing Truth
In regions with limited network infrastructure, like the reserved forests of Tripura, India, cloud-based fact-checking is often impossible. Veritas bridges this digital divide by shifting inference from power-hungry cloud GPUs to local, low-resource CPUs.

### Key Performance Metrics:
- **74.8% Reduction in Size:** 255.45 MB → **64.45 MB** (Fits within Chrome Extension limits).
- **55.2% Faster Inference:** 52.73 ms → **23.58 ms** (Synchronous real-time scrolling).
- **98.7% Accuracy Retention:** Preserves baseline predictive power while using 8-bit integer precision.

---

## 🛠️ Features
- **Hardware-Aware Compression:** Targets `AVX512_VNNI` instructions for maximum CPU throughput.
- **Dynamic Quantization:** Converts FP32 weights to INT8 to reduce memory bandwidth bottlenecks.
- **ONNX Runtime Optimization:** Operator fusion and graph optimization for seamless browser integration.
- **Dataset Remapping:** Automated pipeline to adapt the LIAR dataset for binary safety filters.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/veritas-optimization-engine.git
cd veritas-optimization-engine

# Install dependencies
pip install transformers optimum[onnxruntime] torch pandas datasets
🖥️ Usage
1. Run the Optimization Pipeline
To fine-tune and compress the DistilBERT model into an ONNX artifact:
code
Python
from veritas_engine import VeritasEngine

# Initialize and run the squeeze
engine = VeritasEngine()
engine.optimize()
2. Local Inference Test
Run a real-time check on a political claim using the optimized engine:
code
Python
# (See cell 8 from the notebook for the full inference script)
📊 Research Context
This software is the implementation of the paper: Democratizing Truth: Optimizing Transformer Models for Client-Side Misinformation Detection in Resource-Constrained Environments.
Veritas was built to solve the Latency and Privacy Bottleneck, ensuring that AI safety tools are a local right, not a cloud-dependent luxury.
🤝 Contributing
Contributions are welcome! If you have ideas for improving quantization-aware training (QAT) or extending support to mobile ARM architectures (e.g., for Android deployment), please open an issue or pull request.
📜 License
Distributed under the MIT License. See LICENSE for more information.
✍️ Author
Shambhavi Singh - Independent Researcher
shalinibhavi525@gmail.com

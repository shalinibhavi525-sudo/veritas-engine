---
title: 'Veritas: A Hardware-Aware Optimization Pipeline for Client-Side Misinformation Detection'
tags:
  - Python
  - NLP
  - ONNX
  - Edge AI
  - Information Integrity
authors:
  - name: Shambhavi Singh
    orcid: 0009-0002-3401-4432
    affiliation: Independent Researcher
date: 6 February 2026
bibliography: paper.bib
---

# Summary

Digital misinformation proliferates fastest in regions with limited network infrastructure. While Transformer-based models [@devlin-etal-2019-bert] provide state-of-the-art detection, their size and latency prohibit deployment in "connectivity-blind" environments. `Veritas` is a Python-based optimization engine that compresses DistilBERT architectures [@sanh2019distilbert] for local, offline inference. By leveraging Dynamic Quantization and ONNX Runtime optimizations—specifically targeting the x86 AVX512_VNNI instruction set—`Veritas` achieves a 74.8% reduction in model size, successfully bringing complex NLP tasks under the 100 MB threshold required for seamless browser extension deployment.

# Statement of Need

In areas like the reserved forest regions of Tripura, India, internet connectivity is often restricted to unstable 3G or satellite links. In these contexts, cloud-based verification (e.g., via LLM APIs) is frequently inaccessible. `Veritas` addresses this "Digital Divide" by providing a pipeline for "AI in the Last Mile." 

The software automates the transition from high-memory PyTorch models to hardware-accelerated ONNX artifacts. This ensures that privacy-preserving, real-time misinformation detection can occur entirely client-side, removing the need for data transmission and reducing the carbon footprint of inference.

# Functionality

The `Veritas` engine performs three critical tasks:
1. **Dataset Remapping:** Adapts granular fact-checking datasets like LIAR [@wang-2017-liar] for binary classification suitable for real-time user warnings.
2. **Hardware-Aware Compression:** Implements INT8 quantization optimized for standard consumer CPUs, building upon established integer-arithmetic-only inference schemes [@jacob2018quantization].
3. **Graph Optimization:** Fuses computational kernels into a static ONNX graph to minimize memory-to-cache traffic.

# Mentions

This software utilizes the `Optimum` library for ONNX integration and the `Transformers` library by Hugging Face [@wolf-etal-2020-transformers]. 

# References

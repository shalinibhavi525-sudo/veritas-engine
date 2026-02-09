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
date: 9 February 2026
bibliography: paper.bib
---

# Summary

Digital misinformation proliferates fastest in regions with limited network infrastructure. While Large Language Models (LLMs) provide state-of-the-art detection, their computational requirements often prohibit deployment in "connectivity-blind" environments. `Veritas` is an open-source optimization engine that compresses Transformer-based architectures into lightweight, offline-capable artifacts. By leveraging Dynamic Quantization and ONNX Runtime optimizations, `Veritas` enables real-time misinformation filtering on standard consumer CPUs, allowing AI safety tools to operate without internet dependency or high-end hardware.

# Statement of Need

In regions like the reserved forest areas of Tripura, India, internet connectivity is often restricted to unstable 3G or satellite links. In these contexts, cloud-based verification via APIs (e.g., Gemini or GPT-4) is frequently inaccessible. `Veritas` addresses this "Digital Divide" by providing a pipeline for "AI in the Last Mile." 

The software automates the transition from high-memory PyTorch models to hardware-accelerated INT8 artifacts. This ensures that privacy-preserving, real-time misinformation detection can occur entirely client-side, removing the need for data transmission and ensuring that information integrity remains a local right rather than a cloud-dependent luxury.

# State of the Field

Existing NLP frameworks like Hugging Face `Transformers` [@wolf-etal-2020-transformers] and `Optimum` provide the foundational blocks for model optimization. However, applying these to specific safety domains like the LIAR dataset [@wang-2017-liar] often requires significant manual boilerplate code for label remapping and hardware-specific instruction targeting (e.g., AVX512_VNNI).

Unlike general-purpose compression tools, `Veritas` provides a purpose-built "build vs. contribute" justification: existing alternatives are often too generalized for non-specialist deployment in browser environments. `Veritas` simplifies this by providing a unified class-based engine that handles the entire lifecycle from dataset remapping to VNNI-optimized ONNX export, specifically targeting the 100 MB memory threshold required for web extension deployment.

# Software Design

The design of `Veritas` was governed by the critical trade-off between **predictive performance** and **inference latency**. We chose **Dynamic INT8 Quantization** over Static Quantization to preserve accuracy in nuanced linguistic tasks where activation ranges vary significantly. 

The architecture utilizes a **three-stage pipeline**:
1. **Linguistic Refactoring:** Remapping six-class fact-checking data into binary safety signals.
2. **Precision Reduction:** Converting 32-bit weights to 8-bit integers to minimize memory-to-cache traffic.
3. **Graph Optimization:** Using ONNX Runtime to fuse operators (e.g., LayerNorm and Attention kernels), which is mathematically significant for bypassing the "memory wall" bottleneck typical of Transformer models on CPUs.

# Research Impact Statement

`Veritas` demonstrates credible significance through rigorous benchmarking on consumer-grade hardware. Our results show a **74.8% reduction in model size** (255.45 MB to 64.45 MB) and a **55.2% reduction in latency** (52.73 ms to 23.58 ms). These benchmarks prove that state-of-the-art NLP can be deployed synchronously on the "Edge," enabling non-blocking user interaction even in environments with zero connectivity. The software is community-ready, providing reproducible scripts and a pre-configured optimization environment.

# AI Usage Disclosure

Generative AI (specifically LLMs) was utilized in the refactoring of the software's documentation and the formatting of the `paper.md` file to ensure compliance with JOSS standards. Technical logic, quantization parameters, and hardware-specific optimization targets were manually engineered and verified through empirical benchmarking. All AI-generated text was reviewed for scientific accuracy and technical validity.

# References

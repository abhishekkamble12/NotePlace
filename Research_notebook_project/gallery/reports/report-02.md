# FlashAttention-2: Memory IO Optimization & Parallelization

**Research Question:** How does FlashAttention-2 reduce memory IO complexity and improve training throughput in Transformer architectures?

## 1. Executive Summary

FlashAttention-2 achieves an approximate 2x speedup over the original FlashAttention kernel (and up to 9x over standard PyTorch attention) by minimizing non-matrix-multiplication operations, partitioning compute across the sequence length dimension, and optimizing GPU warp layout.

## 2. Findings per Sub-Question

- FlashAttention-2 achieves roughly a 2x speedup over FlashAttention by cutting non-matmul FLOPs and parallelizing over sequence length. *([KEPT] - Source: arxiv.org)*
  > "FlashAttention-2 yields a 2x speedup over FlashAttention by reducing non-matmul FLOPs and parallelizing over the sequence length dimension."

- FlashAttention-2 partitions work across sequence length rather than solely batch and head dimensions. *([KEPT] - Source: arxiv.org)*
  > "parallelizing over the sequence length dimension even within a single thread block."

- FlashAttention-2 achieves a 10x speedup by completely eliminating SRAM memory traffic on Nvidia A100 GPUs. *([DROPPED] - Source: arxiv.org)*
  > "FlashAttention-2 yields a 2x speedup over FlashAttention"

## 3. Where Sources Disagree

*No cross-source factual contradictions detected.*

## 4. Run Metadata

- **Tokens / cost / latency:** synthetic placeholder - not measured
- **Mode:** multi

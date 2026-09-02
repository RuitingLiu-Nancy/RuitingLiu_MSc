# Utility cross-encoder (256 tokens)

Primary utility cross-encoder used for the reported confirmation analysis.

- Configured base checkpoint: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Canonical upstream page: <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2>
- Task: scalar utility prediction for query-comment pairs
- Objective: Smooth L1 pointwise regression
- Training set size: 15,000 pairs from 300 Development queries
- Maximum sequence length: 256 tokens
- Epochs: 3
- Batch size: 32
- Learning rate: 3e-5
- Warm-up ratio: 0.1
- Random seed: 13

The directory contains the learned parameters, tokenizer and runtime
configuration needed to load this fitted checkpoint. The training corpus is
prepared with the preprocessing and feature-building code in this repository.
Use is subject to the upstream base model's licence and terms.

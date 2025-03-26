A pytorch implementation of speculative sampling supporting both single sequence and batched generation.

The implementation uses a smaller "drafter" model to propose tokens and a larger "verifier" model to accept or reject them, resulting in significant speedups compared to standard autoregressive decoding.

Some details:
- Using greedy decoding to keep the code simple
- In batched mode, each sequence performs independent speculative sampling
- The forward pass of the model is managed using Hugging Face's transformers library
- KV caching is intentionally not used to maintain simplicity

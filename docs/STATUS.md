# ChatV1 status

## Current state

ChatV1 is an original research implementation, not a ChatGPT-scale model. The current transformer is intentionally small and the bootstrap dataset is synthetic.

Implemented:
- causal Transformer model
- deterministic bootstrap tokenizer
- device selection
- resumable checkpoint training
- inference CLI
- dataset splitting and batch sampling
- validation loss/perplexity evaluation
- reusable checkpoint loader
- request validation contracts
- automated unit tests for core components
- zero-budget constraint

## Next engineering tracks

1. Replace the character tokenizer with a production-grade byte/subword tokenizer.
2. Build a licensed public-data ingestion pipeline with filtering, deduplication and train/validation manifests.
3. Add real evaluation suites for instruction following, coding, reasoning and safety.
4. Add a provider/model-agnostic inference service and streaming API.
5. Add retrieval interfaces and source attribution before fine-tuning.
6. Add multimodal adapters after the text stack is reliable.
7. Add image-to-video as a separate capability service rather than coupling it to the language model.

## Verification policy

A feature is not marked verified unless an automated test or an explicit runnable acceptance check has actually executed successfully. GitHub Actions may be used for reproducible verification, but a missing workflow run is not treated as a pass.

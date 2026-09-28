# ChatV1 roadmap

## Definition of done

The system is not considered complete because files exist. Each stage must produce a runnable result and pass automated checks.

### Stage 1 — Tiny language model
- [x] Transformer implementation
- [x] causal attention
- [x] tokenizer
- [x] MPS/CUDA/CPU device selection
- [x] training script
- [x] smoke tests

### Stage 2 — Useful language model
- dataset pipeline
- subword tokenizer
- validation split
- checkpoint/resume
- evaluation metrics
- instruction format
- streaming generation

### Stage 3 — Production inference
- API
- conversation state
- structured errors
- request limits
- observability
- deterministic test suite

### Stage 4 — Agent capabilities
- tool registry
- tool permission model
- memory
- retrieval
- file understanding
- web/search integration

### Stage 5 — Multimodal
- image encoder
- image understanding
- multimodal prompt routing
- audio support

### Stage 6 — Video
- image-to-video adapter
- generation queue
- FFmpeg post-processing
- 9:16 Shorts output
- 10–13 second generation pipeline

### Stage 7 — Product
- web UI
- authentication
- job history
- model configuration
- usage/error dashboards

## Quality gate

Every stage requires:
1. automated tests,
2. a runnable demo,
3. failure-path tests,
4. documented limitations,
5. reproducible setup.

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


## Non-negotiable engineering requirements

### Tests are required for every subsystem

No feature is considered complete without:
- unit tests for normal behavior
- edge-case tests
- invalid-input/failure-path tests
- integration tests where components interact
- regression tests for every bug discovered
- a runnable demo or acceptance test for user-visible behavior

### Design for future needs

New components must expose clean interfaces so they can later support:
- local or cloud inference
- model replacement/upgrades
- streaming
- multimodal inputs
- tool calling
- memory/retrieval
- job queues and retries
- authentication/authorization
- observability
- persistence and migrations
- rate limiting
- cancellation/resume
- versioned model/configuration APIs

Do not over-engineer unused infrastructure, but avoid hard-coding assumptions that would block these future capabilities.

### Interruption/resume protocol

Work must be checkpointed in Git frequently. A network interruption must not erase completed work.

Before stopping a milestone:
1. commit the completed changes,
2. keep the roadmap/status accurate,
3. record known failures or next actions,
4. never claim a feature is verified unless its tests/demo actually ran.

If a session disconnects, the next session resumes from the latest committed repository state and continues from the recorded status. Active execution itself is not guaranteed to continue while the user is offline.

# ChatV1 — Personal AI Research System

An original, modular AI system built from first principles and open-source components.

## Goal

Build a private AI platform that can progressively support:

- text generation
- conversation context
- tokenizer + transformer research model
- vision understanding
- tool calling
- persistent memory
- safety and input validation
- image generation
- image-to-video generation
- a web UI
- reproducible tests and evaluation

## Engineering rules

1. No proprietary OpenAI source code, weights, hidden prompts, private data, or hidden chain-of-thought is copied.
2. Prefer open-source models and documented APIs.
3. Every subsystem gets tests and explicit failure handling.
4. Secrets never enter Git.
5. Changes are small, reviewable, and reproducible.
6. Hardware constraints are treated as first-class requirements.

## Development stages

1. Repository + CI baseline
2. Python/PyTorch research foundation
3. Tiny tokenizer + transformer
4. Training/evaluation harness
5. Chat inference service
6. Vision and multimodal routing
7. Tools and memory
8. Image/video generation pipeline
9. Web application
10. End-to-end evaluation

## Current hardware target

Primary development machine: Apple Silicon MacBook Air, 8 GB unified memory.

Heavy model inference/training is expected to run on an available GPU environment rather than assuming the Mac can host large models.

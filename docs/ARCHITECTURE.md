# ChatV1 architecture principles

## Product experience

The frontend and backend are part of one user-friendly product. Internal complexity must not be exposed to users unless it is useful for debugging or advanced settings.

User flow:
1. Start a conversation.
2. Attach text/images/files when supported.
3. The system determines which internal capability is needed.
4. Execute the minimum required pipeline.
5. Return a clear result with useful progress/error states.
6. Preserve conversation/history when persistence is enabled.

## Backend orchestration

A request should flow through explicit stages:

request -> validation -> context assembly -> capability routing -> retrieval/tools/model inference -> post-processing -> response

Every stage must have:
- typed contracts
- timeouts
- structured errors
- logging hooks
- tests
- cancellation where practical

## RAG

RAG is an internal capability, not a user requirement. The system should decide when retrieval is useful.

Pipeline:
document ingestion -> parsing -> chunking -> metadata -> embeddings/index -> retrieval -> context filtering -> model response

Requirements:
- source attribution
- retrieval evaluation
- stale-document handling
- permission-aware retrieval
- no silent fabrication when retrieval fails

## Fine-tuning

Fine-tuning is not a default answer to knowledge problems.

Use RAG for changing/private knowledge where appropriate. Use fine-tuning when behavior, format, domain style, or task performance requires learned parameter changes.

Fine-tuning pipeline:
dataset validation -> train/validation split -> baseline evaluation -> training -> checkpoint -> evaluation -> regression checks -> versioned model

Never replace a better retrieval solution with fine-tuning merely to avoid building retrieval.

## Model routing

Keep model interfaces provider/model agnostic so an open model can be replaced without rewriting the application.

## Frontend

The UI should be responsive, accessible, keyboard-friendly, mobile-aware, and provide:
- loading/streaming states
- retry actions
- meaningful errors
- upload progress
- generation progress
- cancellation where supported
- conversation history
- clear model/tool activity when useful
- no confusing internal jargon by default

## Zero-budget constraint

Use free/open-source components and free compute where available. Never silently introduce a paid dependency. If a requested capability exceeds free resources, document the limitation and provide the best feasible free path.

## Testing

Every layer requires unit, edge-case, failure-path, and integration tests. User-visible capabilities also require acceptance tests. Every discovered bug gets a regression test.

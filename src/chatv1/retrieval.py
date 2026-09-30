from dataclasses import dataclass
import re

from .rag import Chunk, RetrievedChunk


@dataclass(frozen=True)
class LexicalRetriever:
    chunks: tuple[Chunk, ...]

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = set(re.findall(r"\\w+", query.lower()))
        scored: list[RetrievedChunk] = []
        for chunk in self.chunks:
            words = set(re.findall(r"\\w+", chunk.text.lower()))
            score = len(terms & words)
            if score:
                scored.append(RetrievedChunk(chunk, float(score)))
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:top_k]

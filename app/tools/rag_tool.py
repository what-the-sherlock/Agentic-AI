from dataclasses import dataclass
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.llm.gemini_client import generate_text

@dataclass
class RAGAnswer:
    answer: str
    chunks: List[str]
    scores: List[float]
    confidence: float

class SimpleIndex:
    def __init__(self, index, embeddings: np.ndarray, chunks: List[str]):
        self.index = index
        self.embeddings = embeddings
        self.chunks = chunks

class RAGService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def _simple_chunk(self, text: str, max_chars: int = 500) -> List[str]:
        text = text.strip()
        if not text:
            return []
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for w in words:
            if current_len + len(w) + 1 > max_chars:
                chunks.append(" ".join(current))
                current = [w]
                current_len = len(w)
            else:
                current.append(w)
                current_len += len(w) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

    def build_index_from_text(self, text: str) -> SimpleIndex:
        chunks = self._simple_chunk(text)
        if not chunks:
            chunks = [text] if text else []

        if not chunks or chunks == [""]:
            embeddings = np.zeros((1, 384), dtype="float32")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            return SimpleIndex(index=index, embeddings=embeddings, chunks=[])

        embeddings = self.model.encode(chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))
        return SimpleIndex(index=index, embeddings=embeddings, chunks=chunks)

    def answer_question(self, index: SimpleIndex, question: str, top_k: int = 4) -> RAGAnswer:
        question = (question or "").strip()
        if not question or not index.chunks:
            return RAGAnswer("No context or question provided.", [], [], 0.0)

        q_emb = self.model.encode([question]).astype("float32")
        scores, idxs = index.index.search(q_emb, min(top_k, len(index.chunks)))

        idxs = idxs[0]
        scores_list = scores[0].tolist()
        used_chunks = [index.chunks[i] for i in idxs if 0 <= i < len(index.chunks)]

        if not used_chunks:
             return RAGAnswer("The document provided is empty or unreadable.", [], [], 0.0)

        min_dist = min(scores_list) if scores_list else 10.0
        confidence = max(0.0, 1.0 - (min_dist / 1.5)) 

        context = "\n\n---\n\n".join(used_chunks)
        is_action_items = "action item" in question.lower()
        
        qa_prompt = f"""
You are answering a question based ONLY on the provided document context.

Context:
\"\"\"{context}\"\"\"

Question:
\"\"\"{question}\"\"\"

Rules:
- If the context is insufficient, say: "The document does not contain enough information."
- Answer in 3–6 short sentences.
"""
        if is_action_items:
            qa_prompt += "\n- Return action items as a bullet list starting with '- '."

        answer_text = generate_text(prompt=qa_prompt, temperature=0.2)

        return RAGAnswer(
            answer=answer_text,
            chunks=used_chunks,
            scores=scores_list,
            confidence=confidence,
        )
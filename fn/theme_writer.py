import faiss
import numpy as np
from typing import List, Dict, Any
import requests
import re

from sentence_transformers import SentenceTransformer, CrossEncoder

from .config import (
    XAI_API_KEY,
    XAI_API_URL,
    GROK_MODEL,
    REQUEST_TIMEOUT
)


class ThemeWriter:

    def __init__(
        self,
        embed_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "BAAI/bge-reranker-large"
    ):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.reranker = CrossEncoder(reranker_model_name)

        self.dim = self.embed_model.get_sentence_embedding_dimension()
        self._reset_index()

    def _reset_index(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []

    def embed_fn(self, text: str):
        vec = self.embed_model.encode(text, normalize_embeddings=True)
        return vec.astype("float32")

    def chunk_text(self, text: str, chunk_size=300, overlap=50):
        if not text:
            return []

        words = text.split()
        chunks = []
        i = 0

        while i < len(words):
            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))
            i += chunk_size - overlap

        return chunks

    def build_index(self, papers: List[Dict[str, Any]]):
        vectors = []
        self.doc_enum_map = {}

        for i, paper in enumerate(papers):
            self.doc_enum_map[paper["id"]] = i + 1

        chunk_global_id = 0

        for paper in papers:
            doc_id = paper["id"]
            doc_enum = self.doc_enum_map[doc_id]
            content = paper.get("content", "")

            for chunk in self.chunk_text(content):
                emb = self.embed_fn(chunk)

                self.chunks.append({
                    "chunk_id": chunk_global_id,
                    "doc_enum": doc_enum,
                    "text": chunk
                })

                vectors.append(emb)
                chunk_global_id += 1

        if not vectors:
            return

        vectors = np.vstack(vectors).astype("float32")
        self.index.add(vectors)

    def search(self, query: str, top_k=20):
        if len(self.chunks) == 0:
            return []

        q_emb = self.embed_fn(query).reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)

        results = []
        for i in I[0]:
            if i == -1:
                continue
            results.append(self.chunks[i])

        return results

    def rerank(self, query: str, chunks: List[Dict], top_k=5):
        if not chunks:
            return []

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]

    def call_llm(self, prompt: str) -> str:
        res = requests.post(
            XAI_API_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=REQUEST_TIMEOUT
        )

        try:
            return res.json()["choices"][0]["message"]["content"]
        except Exception:
            return "LLM_ERROR"

    def answer_question(self, question: str, retrieved_chunks: List[Dict]):
        if not retrieved_chunks:
            return "No data available.", {}

        context = ""
        local_map = {}

        for i, ch in enumerate(retrieved_chunks):
            local_id = i + 1
            context += f"[{local_id}] {ch['text']}\n"
            local_map[local_id] = ch

        prompt = f"""
Answer using ONLY context.
Cite every claim with [1], [2].

Context:
{context}

Question:
{question}
"""

        answer = self.call_llm(prompt)
        return answer, local_map

    def normalize_citation(self, text: str, local_map: Dict[int, Dict]):
        def repl(match):
            local_id = int(match.group(1))
            chunk = local_map.get(local_id)

            if not chunk:
                return "[UNK]"

            return f"[{chunk['doc_enum']}.{chunk['chunk_id']}]"

        return re.sub(r"\[(\d+)\]", repl, text)

    def generate_theme(self, papers: List[Dict], questions: Dict[str, str]):
        self._reset_index()
        self.build_index(papers)

        if len(self.chunks) == 0:
            return "No sufficient content to generate theme."

        answers = []

        for query, question in questions.items():
            candidates = self.search(query, top_k=20)
            top_chunks = self.rerank(question, candidates, top_k=5)
            ans, local_map = self.answer_question(question, top_chunks)
            ans = self.normalize_citation(ans, local_map)
            answers.append(ans)

        return "\n".join(answers)
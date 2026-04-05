import faiss
import numpy as np
from typing import List, Dict, Any
import requests
import re

from .config import (
    XAI_API_KEY,
    XAI_API_URL,
    GROK_MODEL,
    REQUEST_TIMEOUT
)


class ThemeWriter:

    def __init__(self, embed_fn, dim: int = 768):
        self.embed_fn = embed_fn
        self.dim = dim
        self._reset_index()

    def _reset_index(self):
        self.index = faiss.IndexFlatL2(self.dim)
        self.chunks = []

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

        # 🔥 map doc_id -> doc_enum (1..N)
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

        vectors = np.array(vectors).astype("float32")
        self.index.add(vectors)

    def search(self, query: str, top_k=5):
        if len(self.chunks) == 0:
            return []

        q_emb = np.array([self.embed_fn(query)]).astype("float32")
        D, I = self.index.search(q_emb, top_k)

        results = []
        for i in I[0]:
            if i == -1:
                continue
            results.append(self.chunks[i])

        return results

    def call_llm(self, prompt: str) -> str:
        res = requests.post(
            XAI_API_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROK_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
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

        # DEBUG
        print("\n==== CONTEXT ====")
        print(context)

        prompt = f"""
Answer the question using ONLY the context below.
Every claim MUST have citation.
Use inline citations like [1], [2].

Context:
{context}

Question:
{question}
"""

        answer = self.call_llm(prompt)

        print("\n==== RAW ANSWER ====")
        print(answer)

        return answer, local_map

    def normalize_citation(self, text: str, local_map: Dict[int, Dict]):
        def repl(match):
            local_id = int(match.group(1))
            chunk = local_map.get(local_id)

            if not chunk:
                return "[UNK]"

            doc_enum = chunk["doc_enum"]
            chunk_id = chunk["chunk_id"]

            return f"[{doc_enum}.{chunk_id}]"

        return re.sub(r"\[(\d+)\]", repl, text)

    def generate_theme(self, papers: List[Dict], questions: List[str]):
        self._reset_index()
        self.build_index(papers)

        if len(self.chunks) == 0:
            return "No sufficient content to generate theme."

        answers = []

        for q in questions:
            chunks = self.search(q, top_k=5)
            ans, local_map = self.answer_question(q, chunks)
            ans = self.normalize_citation(ans, local_map)
            answers.append(ans)

        return "\n".join(answers)
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.types import Document, RetrievedDoc

_OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}


def _is_openai_model(model_name: str) -> bool:
    return model_name in _OPENAI_EMBEDDING_MODELS


def _openai_embed(texts: list[str], model_name: str, batch_size: int = 256) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        batch_emb = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_emb)

    arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.clip(norms, 1e-10, None)
    return np.ascontiguousarray(arr)


def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    from nltk.tokenize import regexp_tokenize
    return regexp_tokenize(
        text,
        pattern=r"\d{4}-\d{2}-\d{2}|[a-z]+(?:-[a-z]+)*|\d+(?::\d+)?",
        gaps=False,
    )


class DenseRetriever:
    def __init__(
            self,
            documents: list[Document],
            model_name: str,
            index_path: str | Path,
            id_map_path: str | Path,
    ):
        self.documents = documents
        self.doc_lookup = {doc.doc_id: doc for doc in documents}
        self.model_name = model_name
        self.use_openai = _is_openai_model(model_name)

        if not self.use_openai:
            self.model = SentenceTransformer(model_name)

        self.index = faiss.read_index(str(index_path))

        with open(id_map_path, "rb") as f:
            self.doc_ids: list[str] = pickle.load(f)

    def _embed_query(self, query: str) -> np.ndarray:
        if self.use_openai:
            return _openai_embed([query], self.model_name)
        else:
            emb = self.model.encode(
                [query],
                prompt_name="query",
                normalize_embeddings=True,
            )
            return np.ascontiguousarray(np.asarray(emb, dtype=np.float32))

    def retrieve(
            self,
            query_view: str,
            view_type: str,
            top_k: int,
    ) -> list[RetrievedDoc]:
        q_emb = self._embed_query(query_view)

        scores, indices = self.index.search(q_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        outputs: list[RetrievedDoc] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            doc_id = self.doc_ids[idx]
            doc = self.doc_lookup[doc_id]
            outputs.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=float(score),
                    rank=rank,
                    source="dense",
                    view_type=view_type,
                    query_view=query_view,
                    metadata=doc.metadata,
                )
            )
        return outputs

    @staticmethod
    def build_and_save(
            documents: list[Document],
            model_name: str,
            index_path: str | Path,
            id_map_path: str | Path,
            graph_path: str | Path | None = None,
            knn_k: int = 5,
            batch_size: int = 32,
    ) -> None:
        if not documents:
            raise ValueError("Cannot build dense index from empty documents.")

        texts = [doc.text for doc in documents]
        use_openai = _is_openai_model(model_name)

        print(f"  [DenseRetriever] encoding {len(texts)} documents "
              f"({'OpenAI API' if use_openai else 'SentenceTransformer'})...")

        if use_openai:
            embeddings = _openai_embed(texts, model_name, batch_size=min(batch_size, 256))
        else:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            embeddings = np.ascontiguousarray(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        index_path = Path(index_path)
        id_map_path = Path(id_map_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

        doc_ids = [doc.doc_id for doc in documents]
        with id_map_path.open("wb") as f:
            pickle.dump(doc_ids, f)

        if graph_path is not None:
            graph_path = Path(graph_path)
            graph_path.parent.mkdir(parents=True, exist_ok=True)

            k = min(knn_k + 1, len(documents))
            scores_mat, indices_mat = index.search(embeddings, k)

            doc_graph: dict[str, list[tuple[str, float]]] = {}
            for i, doc_id in enumerate(doc_ids):
                neighbors: list[tuple[str, float]] = []
                for j, score in zip(indices_mat[i], scores_mat[i]):
                    if j < 0 or j == i:
                        continue
                    neighbors.append((doc_ids[j], float(score)))
                doc_graph[doc_id] = neighbors[:knn_k]

            with graph_path.open("wb") as f:
                pickle.dump(doc_graph, f)
            print(f"  [DenseRetriever] saved doc graph -> {graph_path}  (k={knn_k})")


class SparseRetriever:
    def __init__(
            self,
            documents: list[Document],
            bm25_path: str | Path,
            id_map_path: str | Path,
    ):
        self.documents = documents
        self.doc_lookup = {doc.doc_id: doc for doc in documents}

        with open(bm25_path, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        with open(id_map_path, "rb") as f:
            self.doc_ids: list[str] = pickle.load(f)

    def retrieve(
            self,
            query_view: str,
            view_type: str,
            top_k: int,
    ) -> list[RetrievedDoc]:
        query_tokens = simple_tokenize(query_view)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        outputs: list[RetrievedDoc] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            doc_id = self.doc_ids[idx]
            doc = self.doc_lookup[doc_id]
            outputs.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=float(score),
                    rank=rank,
                    source="sparse",
                    view_type=view_type,
                    query_view=query_view,
                    metadata=doc.metadata,
                )
            )
        return outputs

    @staticmethod
    def build_and_save(
            documents: list[Document],
            bm25_path: str | Path,
            id_map_path: str | Path,
    ) -> None:
        if not documents:
            raise ValueError("Cannot build sparse index from empty documents.")

        corpus_tokens = [simple_tokenize(doc.text) for doc in documents]
        bm25 = BM25Okapi(corpus_tokens)

        bm25_path = Path(bm25_path)
        id_map_path = Path(id_map_path)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)

        with bm25_path.open("wb") as f:
            pickle.dump(bm25, f)
        with id_map_path.open("wb") as f:
            pickle.dump([doc.doc_id for doc in documents], f)

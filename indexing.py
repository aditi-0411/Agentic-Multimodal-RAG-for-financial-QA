"""
indexing.py — Smart chunking and per-modality index construction.

Builds three separate knowledge bases (text / table / metadata), each with
its own FAISS dense index and BM25 sparse index, so each specialist agent
retrieves only from its own modality.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from src.config import embed_texts


@dataclass
class KnowledgeBase:
    """Bundles all three modality-specific indexes and their source chunks."""
    text_chunks: List[Dict[str, Any]] = field(default_factory=list)
    table_chunks: List[Dict[str, Any]] = field(default_factory=list)
    metadata_chunks: List[Dict[str, Any]] = field(default_factory=list)

    text_index: Any = None
    table_index: Any = None
    metadata_index: Any = None

    bm25_text: Any = None
    bm25_table: Any = None
    bm25_metadata: Any = None


def build_faiss_index(chunks, embedding_model):
    texts = [c["content"] for c in chunks]
    embs = embed_texts(embedding_model, texts)
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(np.array(embs))
    return idx, texts


def chunk_dataset(processed_data, chunk_size=500, chunk_overlap=100):
    """Splits each record's text/table/metadata into per-modality chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    text_chunks, table_chunks, metadata_chunks = [], [], []

    for item in processed_data:
        base_meta = item["metadata"]

        for c in splitter.split_text(item["text"]):
            text_chunks.append({"content": c, "type": "text", "metadata": base_meta})

        if item["table"] and len(item["table"].strip()) > 10:
            table_chunks.append({"content": item["table"], "type": "table", "metadata": base_meta})

        meta_str = (
            f"Company: {base_meta.get('company', 'N/A')} | "
            f"Year: {base_meta.get('year', 'N/A')} | "
            f"Sector: {base_meta.get('sector', 'N/A')}"
        )
        metadata_chunks.append({"content": meta_str, "type": "metadata", "metadata": base_meta})

    print(f"Text chunks:     {len(text_chunks)}")
    print(f"Table chunks:    {len(table_chunks)}")
    print(f"Metadata chunks: {len(metadata_chunks)}")

    return text_chunks, table_chunks, metadata_chunks


def build_knowledge_base(processed_data, embedding_model, chunk_size=500, chunk_overlap=100) -> KnowledgeBase:
    text_chunks, table_chunks, metadata_chunks = chunk_dataset(processed_data, chunk_size, chunk_overlap)

    text_index, text_texts = build_faiss_index(text_chunks, embedding_model)
    table_index, table_texts = build_faiss_index(table_chunks, embedding_model)
    metadata_index, metadata_texts = build_faiss_index(metadata_chunks, embedding_model)

    bm25_text = BM25Okapi([t.split() for t in text_texts])
    bm25_table = BM25Okapi([t.split() for t in table_texts])
    bm25_metadata = BM25Okapi([t.split() for t in metadata_texts])

    print("All three indexes built")

    return KnowledgeBase(
        text_chunks=text_chunks, table_chunks=table_chunks, metadata_chunks=metadata_chunks,
        text_index=text_index, table_index=table_index, metadata_index=metadata_index,
        bm25_text=bm25_text, bm25_table=bm25_table, bm25_metadata=bm25_metadata,
    )

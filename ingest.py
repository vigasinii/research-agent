"""Ingest markdown docs into Qdrant."""
import os
import sys
from pathlib import Path
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION = "research_docs"
DOCS_DIR = Path(__file__).parent / "docs"
CHUNK_SIZE = 400   # tokens (approximated as words for simplicity)
CHUNK_OVERLAP = 50
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384


def _log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] [INGEST] {msg}")


def word_chunk(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


def main() -> None:
    _log(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    _log("Connecting to Qdrant at localhost:6333")
    client = QdrantClient(host="localhost", port=6333)

    # Recreate collection for a clean ingest
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        _log(f"Dropping existing collection '{COLLECTION}'")
        client.delete_collection(COLLECTION)

    _log(f"Creating collection '{COLLECTION}' (dim={VECTOR_DIM}, cosine)")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        _log(f"No .md files found in {DOCS_DIR}")
        sys.exit(1)

    points: list[PointStruct] = []
    point_id = 0

    for doc_path in docs:
        _log(f"Processing {doc_path.name}")
        text = doc_path.read_text(encoding="utf-8")
        chunks = word_chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)
        _log(f"  → {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "source": doc_path.name,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                )
            )
            point_id += 1

    _log(f"Upserting {len(points)} total points into Qdrant")
    client.upsert(collection_name=COLLECTION, points=points)
    _log(f"Ingest complete. Collection '{COLLECTION}' has {point_id} vectors.")


if __name__ == "__main__":
    main()

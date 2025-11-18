import json
from pathlib import Path
from nomic import embed
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# -----------------------------
# CONFIG
# -----------------------------
CHUNKS_DIR = Path("docs/processed/chunks")  # contains *_chunks.jsonl
COLLECTION_NAME = "isaac_chunks"
BATCH_SIZE = 50
VECTOR_DIM = 512  # Nomic-Embed-Text v1.5
DEVICE = "cuda"  # default GPU
import torch
if not torch.cuda.is_available():
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# -----------------------------
# Qdrant client setup
# -----------------------------
qdrant_client = QdrantClient("localhost", port=6333)

# Recreate collection (safe to run multiple times)
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
)

# -----------------------------
# Helper to read chunks
# -----------------------------
def read_chunks(jsonl_file: Path):
    with jsonl_file.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunk_id = chunk.get("id") or chunk.get("chunk_id")
            text = chunk["content"]
            yield str(chunk_id), text

# -----------------------------
# Ingest function
# -----------------------------
points = []
total_chunks = 0
for jsonl_file in CHUNKS_DIR.glob("*_chunks.jsonl"):
    print(f"Processing file: {jsonl_file.name}")
    for chunk_id, text in read_chunks(jsonl_file):
        total_chunks += 1
        try:
            resp = embed.text(
                texts=[text],
                model="nomic-embed-text-v1.5",
                task_type="search_document",
                dimensionality=VECTOR_DIM,
                inference_mode="local",
                device=DEVICE
            )
        except Exception as e:
            print(f"⚠ Embedding failed for chunk {chunk_id}: {e}")
            continue

        embeddings = resp.get("embeddings")
        if not embeddings or len(embeddings[0]) != VECTOR_DIM:
            print(f"⚠ Invalid embedding for chunk {chunk_id}, skipping")
            continue

        vector = embeddings[0]
        points.append({
            "id": chunk_id,
            "vector": vector,
            "payload": {"text": text}
        })

        # Batch insert
        if len(points) >= BATCH_SIZE:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"Upserted {len(points)} points")
            points = []

# Insert remaining points
if points:
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted remaining {len(points)} points")

print(f"Finished ingestion. Total chunks processed: {total_chunks}")

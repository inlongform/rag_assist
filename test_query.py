from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from nomic import embed

DEVICE = "cuda"  # default GPU
import torch
if not torch.cuda.is_available():
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Embed the query text
query_text = "How do I spawn a robot in Isaac Lab?"
resp = embed.text(
    texts=[query_text],
    model="nomic-embed-text-v1.5",
    task_type="search_document",
    dimensionality=512,
    inference_mode="local",
    device=DEVICE
)
query_vector = resp["embeddings"][0]

# Perform search using query_points
results = client.query_points(
    collection_name="isaac_chunks",
    query=query_vector,
    limit=5,
    with_payload=True
)

# Print results
for r in results.points:
    print("id:", r.id, "score:", r.score)
    print("payload:", r.payload)
    print()
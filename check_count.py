from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")

info = client.get_collection("isaac_chunks")
print(info)

count = client.count("isaac_chunks")
print("Total points:", count.count)
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "gst_instruments"

# Delete if exists
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
    print("🗑️ Collection deleted")


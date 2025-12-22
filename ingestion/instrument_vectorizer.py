import json
import hashlib
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from embeddings import embed_text, EMBEDDING_DIM
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INSTRUMENTS_DIR = Path("storage/instruments")
COLLECTION_NAME = "notification_instruments_cloud"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

RECREATE_COLLECTION = True  # Set True only when rebuilding

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def stable_id(group_id: str) -> str:
    return hashlib.md5(group_id.encode("utf-8")).hexdigest()

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# --------------------------------------------------
def ensure_collection(client: QdrantClient):
    collections = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME in collections and RECREATE_COLLECTION:
        client.delete_collection(COLLECTION_NAME)
        collections.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            ),
            optimizers_config={"indexing_threshold": 1}
        )

# --------------------------------------------------
def run_vectorization():
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    ensure_collection(client)

    files = sorted(INSTRUMENTS_DIR.glob("*.json"))
    print(f"\n📦 Vectorizing {len(files)} instruments\n")

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        text = data.get("composite_text", "").strip()
        if not text:
            continue

        point_id = stable_id(data["group_id"])
        vector = embed_text(text)

        payload = {
            "group_id": data["group_id"],
            "tax_type": data["tax_type"],
            "notification_no": data["notification_no"],
            "latest_effective_date": data.get("latest_effective_date"),
            "file_paths": data.get("file_paths", []),
            "content_hash": content_hash(text),
        }

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )

    print("✅ Vectorization complete")

# --------------------------------------------------
if __name__ == "__main__":
    run_vectorization()

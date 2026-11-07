from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.config import settings
from embedding import EMBEDDING_DIMENSIONS, embeddings

COLLECTION_NAME = "test"


def get_client() -> QdrantClient:
    """Build a Qdrant client."""
    return QdrantClient(
        url=settings.QDRANT_ENDPOINT,
        api_key=settings.QDRANT_API_KEY,
    )


def init_vector_store(
    collection_name: str = COLLECTION_NAME,
    client: QdrantClient | None = None,
) -> QdrantVectorStore:
    """Connect to Qdrant, create the collection if missing, return the store."""
    client = client or get_client()

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

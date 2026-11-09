from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Modifier

from config.config import settings
from embedding import EMBEDDING_DIMENSIONS, embeddings

COLLECTION_NAME = "entredocs-hybrid"
SPARSE_VECTOR_NAME = "sparse"
DENSE_VECTOR_NAME = "dense"


def get_client() -> QdrantClient:
    """Build a Qdrant client."""
    return QdrantClient(
        url=settings.QDRANT_ENDPOINT,
        api_key=settings.QDRANT_API_KEY,
    )


def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name="Qdrant/bm25")


def init_vector_store(
    collection_name: str = COLLECTION_NAME,
    client: QdrantClient | None = None,
) -> QdrantVectorStore:
    """Create hybrid search .."""
    client = client or get_client()

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
            },
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=get_sparse_embeddings(),
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )

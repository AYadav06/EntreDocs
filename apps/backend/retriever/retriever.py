from retriever.Reranker import hybridSearch_with_rerank
from generator.llm import answer
from qdrant_client import models

from embedding import embeddings
from vectorStore.qdrant import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    get_client,
    get_sparse_embeddings,
    SPARSE_VECTOR_NAME,
)


def search(
    query: str,
    k: int = 4,
    collection_name: str = COLLECTION_NAME,
    prefetch_k: int = 20,
) -> list[tuple]:
    """Hybrid search: two independent ranking streams (dense + sparse) are
    fetched in parallel and fused with Reciprocal Rank Fusion (RRF).
    """
    client = get_client()
    sparse_emb = get_sparse_embeddings()

    dense_vec = embeddings.embed_query(query)
    spar_vec = models.SparseVector(
        indices=sparse_emb.embed_query(query).indices,
        values=sparse_emb.embed_query(query).values,
    )

    return client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                using=DENSE_VECTOR_NAME, query=dense_vec, limit=prefetch_k
            ),
            models.Prefetch(
                using=SPARSE_VECTOR_NAME, query=spar_vec, limit=prefetch_k
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    ).points

def main() -> None:
    query = "Sliding Window Attention."
    results = hybridSearch_with_rerank(query)  #retrieval
    llm_result = answer(query, [p for p, _ in results])  #generation
    for i, (point, rerank_score) in enumerate(results, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        content = (payload.get("page_content") or "").strip().replace("\n", " ")
        print(f"[{i}] rerank_score={rerank_score:.4f}  page={metadata.get('page')}")
        print(f"    {content[:200]}...\n")
    print(llm_result)
if __name__ == "__main__":
    main()

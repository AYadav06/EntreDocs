from embedding import embeddings
from qdrant_client import models

from vectorStore.qdrant import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, get_client, get_sparse_embeddings
from sentence_transformers import CrossEncoder


reranker= CrossEncoder("BAAI/bge-reranker-base")


def hybridSearch_with_rerank(query,k=4,rerank_k=20):

    client = get_client()
    dense_vec= embeddings.embed_query(query)

    spar_vec = models.SparseVector(
        indices = get_sparse_embeddings().embed_query(query).indices,
        values = get_sparse_embeddings().embed_query(query).values,
    )

    points = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(using=DENSE_VECTOR_NAME, query=dense_vec, limit=rerank_k),
            models.Prefetch(using=SPARSE_VECTOR_NAME, query=spar_vec, limit=rerank_k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=rerank_k,
        with_payload=True,
    ).points
    
    docs= [p.payload.get("page_content","") for p in points]
    scores= reranker.predict([(query,d) for d in docs])

    ranked= sorted(zip(points,scores),key=lambda x:x[1],reverse=True)
    return ranked[:k]
from embedding import embeddings
from vectorStore.qdrant import COLLECTION_NAME, get_client


def search(query: str, k: int = 4, collection_name: str = COLLECTION_NAME):
    """Embed the query and return the top-k most similar chunks from Qdrant."""
    client = get_client()
    return client.query_points(
        collection_name=collection_name,
        query=embeddings.embed_query(query),
        limit=k,
        with_payload=True,
    )


def main() -> None:
    query = "Sliding Window Attention."
    result = search(query)

    print(f"Query: {query}\n")
    for i, point in enumerate(result.points, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        content = (payload.get("page_content") or "").strip().replace("\n", " ")
        print(f"[{i}] score={point.score:.4f}  page={metadata.get('page')}")
        print(f"    {content[:200]}...\n")


if __name__ == "__main__":
    main()

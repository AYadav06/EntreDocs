def build_prompt(query: str, results: list) -> str:
    """Build the generation prompt from the query and retrieved search results."""

    context = "\n\n".join(
        f"[Page {p.payload.get('metadata', {}).get('page')}] {p.payload.get('page_content') or ''}"
        for p in results
    )

    prompt = f"""You are a helpful assistant answering questions from the following retrieved documents.
Answer based ONLY on the retrieved context. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:
"""
    return prompt

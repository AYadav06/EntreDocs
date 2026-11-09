# EntreDocs Backend

Enterprise RAG backend for question answering over internal PDF documents.
It combines **hybrid retrieval** (dense + sparse), a **cross-encoder reranker**,
and **OpenAI generation**, with a **HyDE** query-expansion mode for
paraphrase-heavy questions.

> **Status:** ingestion, hybrid search, reranking, generation and HyDE search
> are implemented. ACL filtering and RAGAS evaluation are planned (see
> [Roadmap](#roadmap)).

---

## Current flow

```
                 ┌─────────────────────────────────────────────┐
                 │                  INGESTION                  │
                 │                                             │
 PDFs in docs ──▶ pypdf (page text + metadata)                 │
                 │        │                                    │
                 │        ▼                                    │
                 │ RecursiveCharacterTextSplitter               │
                 │   chunk_size=1000, overlap=150               │
                 │        │                                    │
                 │        ▼                                    │
                 │ QdrantVectorStore.add_documents()            │
                 │   dense  : text-embedding-3-large (3072d)    │
                 │   sparse : Qdrant/bm25 (IDF)                 │
                 └──────────────────┬──────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Qdrant collection  │
                         │  "entredocs-hybrid" │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┴──────────────────────────┐
                 │                QUERY TIME                   │
                 │                                             │
 Question ──┬──▶ dense embed ──┐                              │
            │                  ├──▶ RRF fusion ──▶ rerank ──▶ │
            └──▶ BM25 sparse ───┘   (top 20)     bge-base     │
                 │                                             │
                 └── (optional) HyDE: LLM answer ──▶ dense     │
                                     only                      │
                                    │                          │
                                    ▼                          │
                 GPT-4o-mini answers from retrieved context    │
                 └─────────────────────────────────────────────┘
```

### Ingestion

`document_loader.py` loads every `*.pdf` from `documents/`, extracts page text
with `pypdf`, and splits it into overlapping chunks. Each chunk keeps
`source` and `page` metadata. Chunks are written to Qdrant with both a dense
vector (`text-embedding-3-large`, 3072 dimensions) and a sparse BM25 vector.

### Hybrid retrieval

`retriever/retriever.py` runs two independent searches against the same
collection:

| Branch | Vector                   | Strength                          |
| ------ | ------------------------ | --------------------------------- |
| Dense  | `text-embedding-3-large` | semantic similarity / paraphrases |
| Sparse | `Qdrant/bm25` (IDF)      | exact terms, names, acronyms      |

Each branch returns `prefetch_k` candidates; Qdrant fuses them with
**Reciprocal Rank Fusion (RRF)** and returns the top `k`.

### Reranking

`retriever/Reranker.py` takes the fused candidates and scores each
`(query, chunk)` pair with the **`BAAI/bge-reranker-base`** cross-encoder.
Results are sorted by rerank score and truncated to `k`. This is the highest
quality path and the one used by generation.

### Generation

`generator/llm.py` builds a context-only prompt (`generator/prompt.py`) and
calls **`gpt-4o-mini`** at `temperature=0`. The model is instructed to answer
only from the retrieved context and to say so when the answer is missing.

### HyDE (optional query expansion)

`query/hyde.py` implements **Hypothetical Document Embeddings**:

1. The LLM writes a short, factual hypothetical answer to the question.
2. That answer is embedded instead of the raw question.
3. The vector is used for a **dense-only** search.

Dense-only is intentional: HyDE exists to bridge vocabulary mismatch, while
BM25 depends on the exact vocabulary. Embedding a generated paragraph for the
sparse branch injects invented terms and dilutes the real query keywords, so
it hurts more than it helps.

---

## Project structure

```
apps/backend/
├── config/config.py          # pydantic-settings, loads .env
├── document_loader.py        # PDF → chunks → Qdrant
├── embedding.py              # OpenAI dense embeddings + dimensions
├── generator/
│   ├── llm.py                # GPT-4o-mini + answer()
│   └── prompt.py             # grounded answer prompt
├── retriever/
│   ├── Reranker.py           # hybrid search + bge cross-encoder
│   └── retriever.py          # hybrid search, answer, CLI demo
├── query/
│   └── hyde.py               # HyDE dense search
├── vectorStore/
│   └── qdrant.py             # collection schema + client helpers
├── documents/                # input PDFs (git-ignored)
└── .env                      # secrets (git-ignored)
```

---

## Setup

Requires Python ≥ 3.13. From `apps/backend/`:

```bash
uv sync
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Create `apps/backend/.env`:

```dotenv
OPENAI_API_KEY=sk-...
GROQ_API_KEY=...
QDRANT_ENDPOINT=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=...
```

Notes:

- `QDRANT_ENDPOINT` can be a local instance (`http://localhost:6333`) or Qdrant
  Cloud. Use a cloud endpoint or run Qdrant locally with Docker.
- The collection is created automatically on first ingest, including the
  named `dense`/`sparse` vectors.
- The first reranker run downloads `BAAI/bge-reranker-base` (~1 GB).

All commands below assume the working directory is `apps/backend/`, because
`config.py` reads `.env` relative to the current directory.

---

## Usage

### 1. Ingest documents

Put PDFs in `documents/`, then:

```bash
python document_loader.py
```

### 2. Ask a question (hybrid + rerank + generate)

```bash
python retriever/retriever.py
```

Edit the `query` variable in `main()` or import it:

```python
from retriever.retriever import search
from retriever.Reranker import hybridSearch_with_rerank
from generator.llm import answer

query = "How does sliding window attention work in Mistral 7B?"
results = hybridSearch_with_rerank(query, k=4)          # retrieve + rerank
print(answer(query, [p for p, _ in results]))            # generate
```

### 3. HyDE search

```bash
python query/hyde.py
```

Or programmatically:

```python
from query.hyde import hyde_search

for point in hyde_search("How does sliding window attention work?", top_k=3):
    print(point.score, point.payload["metadata"].get("page"))
```

---

## Configuration reference

| Setting              | Value                         |
| -------------------- | ----------------------------- |
| Collection           | `entredocs-hybrid`            |
| Dense vector         | `dense`, cosine, 3072 dims    |
| Sparse vector        | `sparse`, BM25 + IDF modifier |
| Chunk size / overlap | 1000 / 150 characters         |
| Dense model          | `text-embedding-3-large`      |
| Sparse model         | `Qdrant/bm25` (FastEmbed)     |
| Reranker             | `BAAI/bge-reranker-base`      |
| Generator            | `gpt-4o-mini`, temperature 0  |
| Fusion               | Reciprocal Rank Fusion (RRF)  |

---

## RAGAS evaluation (planned)

RAGAS scores the pipeline on the **four retrieval/answer quality axes**:

| Metric                | Question it answers                                          |
| --------------------- | ------------------------------------------------------------ |
| **Faithfulness**      | Is every claim in the answer supported by retrieved context? |
| **Answer relevancy**  | Does the answer actually address the question?               |
| **Context precision** | Are the retrieved chunks relevant / well ranked?             |
| **Context recall**    | Did retrieval find everything needed to answer?              |

To evaluate, you need a small labelled dataset (10–50 questions is enough):
`user_input`, `response`, `retrieved_contexts`, and a `reference` answer.
Collect the first three from the live pipeline and write references by hand.

Minimal example (not yet wired into the repo):

```python
# pip install ragas datasets
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from generator.llm import answer
from retriever.Reranker import hybridSearch_with_rerank

questions = ["How does sliding window attention work in Mistral 7B?"]
references = ["..."]

samples = []
for query, reference in zip(questions, references):
    points = [p for p, _ in hybridSearch_with_rerank(query, k=4)]
    samples.append(
        {
            "user_input": query,
            "response": answer(query, points),
            "retrieved_contexts": [p.payload["page_content"] for p in points],
            "reference": reference,
        }
    )

result = evaluate(
    Dataset.from_dict(
        {key: [s[key] for s in samples] for key in samples[0]}
    ),
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)
print(result)
```

Practical tips:

- Evaluate **hybrid+rerank** and **HyDE** as separate variants so you can see
  which retrieval strategy actually helps.
- `gpt-4o-mini` is fine as the RAGAS judge for a first pass; use a stronger
  model when the numbers matter.
- Pin `ragas` and `datasets` versions — metric APIs move between releases.
- Ragas makes many extra LLM calls; start with ~20 questions.

---

## Roadmap

- [ ] ACL / document-level access filtering (enterprise requirement)
- [ ] RAGAS evaluation script + committed question set
- [ ] FastAPI service layer
- [ ] Incremental / idempotent ingestion
- [ ] Hybrid HyDE variant (dense on hypothetical answer, sparse on the original
      query, fused with RRF) if keyword-heavy questions need it

"""Demonstrates metadata filtering and Maximal Marginal Relevance (MMR) reranking.

1. Metadata Filtering: Restricts search to specific tenants, categories, or tags.
2. MMR Reranking: Avoids retrieving near-duplicate chunks, selecting both relevant
   and diverse contexts to optimize the LLM context window.

Run with: python examples/metadata_filtering_and_mmr.py
"""

from ragforge.chunking import RecursiveCharacterChunker
from ragforge.pipeline import RagPipeline
from ragforge.reranking import MaxMarginalRelevanceReranker

DOCS = [
    {
        "id": "billing_faq_1",
        "text": "Subscription renewals occur on the 1st of each month. Invoices are emailed.",
        "metadata": {"department": "finance", "access_level": "public"},
    },
    {
        "id": "billing_faq_2",
        "text": "Monthly subscription charges run on day 1. Customers receive email receipts.",
        "metadata": {"department": "finance", "access_level": "public"},
    },
    {
        "id": "billing_enterprise",
        "text": "Enterprise billing includes custom net-30 terms and dedicated account management.",
        "metadata": {"department": "finance", "access_level": "enterprise"},
    },
    {
        "id": "engineering_auth",
        "text": "API authentication requires Bearer tokens generated from developer console.",
        "metadata": {"department": "engineering", "access_level": "developer"},
    },
]


def main() -> dict:
    # 1. Pipeline with MMR reranker to maximize diversity and eliminate redundancy
    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=30),
        reranker=MaxMarginalRelevanceReranker(lambda_mult=0.6),
    )
    pipeline.ingest_batch(DOCS)

    print("=== 1. Standard Query (Finance & Public Filter) ===")
    finance_public = pipeline.answer(
        "How do subscription renewals and billing work?",
        k=2,
        filter_fn=lambda chunk: (
            chunk.metadata.get("department") == "finance"
            and chunk.metadata.get("access_level") == "public"
        ),
    )
    print(f"Answer: {finance_public.answer}")
    print("Retrieved chunks:")
    for ctx in finance_public.contexts:
        print(f"  [{ctx.doc_id}] ({ctx.metadata['access_level']}) :: {ctx.text}")

    print("\n=== 2. MMR vs Redundant Contexts ===")
    # Query without filter, but demonstrating MMR picking diverse docs across departments
    diverse_answer = pipeline.answer(
        "billing and authentication",
        k=2,
    )
    print("Retrieved diverse contexts:")
    for ctx in diverse_answer.contexts:
        print(f"  [{ctx.doc_id}] Dept: {ctx.metadata.get('department')} :: {ctx.text}")

    return {
        "finance_count": len(finance_public.contexts),
        "diverse_count": len(diverse_answer.contexts),
    }


if __name__ == "__main__":
    main()

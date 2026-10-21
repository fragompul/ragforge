"""Defend a RAG pipeline against indirect prompt injection and PII leaks.

A poisoned document in the index tries to hijack the model; a customer record
carries personal data. With ``Guardrails`` both are neutralized *before* they
reach the prompt -- and the findings are reported for auditing.

Run with: python examples/guardrails_demo.py
"""

from ragforge import Guardrails, RagPipeline
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.guardrails import spotlight_prompt_formatter
from ragforge.sample import SAMPLE_DOCUMENTS


def main() -> dict[str, object]:
    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=0),
        bm25_analyzer="english",
        guardrails=Guardrails(action="sanitize"),
        prompt_formatter=spotlight_prompt_formatter,
    )
    pipeline.ingest_batch(SAMPLE_DOCUMENTS)

    shipping = pipeline.answer("What is the standard shipping time?", k=3)
    poisoned = next(c for c in shipping.contexts if c.doc_id == "poisoned-faq")
    billing = pipeline.answer("Who was charged twice and refunded?", k=3)
    record = next(c for c in billing.contexts if c.doc_id == "customer-record")

    print("Poisoned FAQ as the model sees it:\n ", poisoned.text, "\n")
    for finding in poisoned.guardrails["findings"]:
        print(f"  ! {finding['rule']:<22} {finding['severity']:<6} {finding['excerpt']}")
    print("\nCustomer record as the model sees it:\n ", record.text)
    return {
        "injection_risk": poisoned.guardrails["risk"],
        "poisoned_text": poisoned.text,
        "record_text": record.text,
        "prompt_is_spotlighted": "untrusted" in shipping.prompt,
    }


if __name__ == "__main__":
    main()

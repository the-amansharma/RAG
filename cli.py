import json
from pathlib import Path
from qdrant_client import QdrantClient
from ingestion.embeddings import embed_text
from retrieval.llm_explainer import explain

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "notification_instruments"
INSTRUMENTS_DIR = Path("storage/instruments")

TOP_K = 3
MIN_SCORE = 0.4
AMBIGUITY_GAP = 0.03

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def load_composite_text(group_id: str) -> str:
    filename = group_id.replace("::", "__").replace("/", "_") + ".json"
    path = INSTRUMENTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Instrument file not found: {filename}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return data["composite_text"]


GST_KEYWORDS = {
    "gst", "cgst", "sgst", "igst",
    "goods and services tax",
    "works contract",
    "construction",
    "supply of services",
}

def looks_like_gst_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in GST_KEYWORDS)


def is_ambiguous(results) -> bool:
    if len(results) < 2:
        return False
    return (results[0].score - results[1].score) < AMBIGUITY_GAP


# --------------------------------------------------
# CLI LOOP
# --------------------------------------------------
def run_cli():
    client = QdrantClient(url=QDRANT_URL)

    print("\n📘 Legal Assistant")
    print("Type your question. Type 'exit' to quit.\n")

    while True:
        query = input("❓ Ask your question: ").strip()

        if not query or query.lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break

        print("🔍 Searching relevant notification...")

        query_vector = embed_text(query)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
            with_payload=True
        ).points

        # ---------------------------
        # DEBUG: SCORE DISTRIBUTION
        # ---------------------------
        # print("\n📊 SCORE DISTRIBUTION:")
        # for i, r in enumerate(results, start=1):
        #     print(
        #         f"  #{i} score={r.score:.4f} "
        #         f"group_id={r.payload.get('group_id')}"
        #     )
        # print()

        if not results:
            print("❌ No relevant GST notification found.\n")
            continue

        top = results[0]

        # ---------------------------
        # OUTCOME CLASSIFICATION
        # ---------------------------
        # if top.score < MIN_SCORE:
        #     print(
        #         f"❌ No sufficiently relevant GST notification found "
        #         f"(score={top.score:.3f})\n"
        #     )
        #     continue

        # if not looks_like_gst_query(query):
        #     print(
        #         "⚠️ Query appears to relate to a non-GST statute "
        #         "(e.g., Customs / Excise / Income Tax).\n"
        #         "GST explanation blocked.\n"
        #     )
        #     continue

        # ---------------------------
        # AMBIGUITY WARNING
        # ---------------------------
        if is_ambiguous(results):
            print(
                "⚠️ Multiple GST notifications may be relevant. "
                "Answer may depend on context.\n"
            )

        # ---------------------------
        # LOAD INSTRUMENT
        # ---------------------------
        instrument = top.payload
        composite_text = load_composite_text(instrument["group_id"])

        print(
            f"✅ Matched: {instrument['notification_no']} "
            f"({instrument['tax_type']}) "
            f"[score={top.score:.3f}]\n"
        )
        #Legal text
        print("\n")
        print("Legal text")
        print("\n\n")
        print(composite_text)

        # ---------------------------
        # LLM EXPLANATION (SAFE ZONE)
        # ---------------------------
        # print("✍️ Generating explanation...\n")

        # try:
        #     answer = explain(
        #         question=query,
        #         instrument=instrument,
        #         composite_text=composite_text
        #     )
        # except Exception as e:
        #     print(f"❌ Explanation failed: {e}\n")
        #     continue

        # print("📄 ANSWER:\n")
        # print(answer)
        # print("\n" + "-" * 70 + "\n")


# --------------------------------------------------
if __name__ == "__main__":
    run_cli()

# cli.py
import json
from pathlib import Path

# ---------------- CONFIG ----------------
INSTRUMENTS_DIR = Path("storage/instruments")
MIN_SCORE = 0.4
AMBIGUITY_GAP = 0.03

GST_KEYWORDS = {
    "gst", "cgst", "sgst", "igst",
    "goods and services tax",
    "works contract",
    "construction",
    "supply of services",
}

# ---------------- HELPERS ----------------
def load_composite_text(group_id: str) -> str:
    filename = group_id.replace("::", "__").replace("/", "_") + ".json"
    path = INSTRUMENTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Instrument file not found: {filename}")
    return json.loads(path.read_text(encoding="utf-8"))["composite_text"]


def looks_like_gst_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in GST_KEYWORDS)


def is_ambiguous(results) -> bool:
    if len(results) < 2:
        return False
    return (results[0].score - results[1].score) < AMBIGUITY_GAP

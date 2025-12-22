import json
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
REGISTRY_PATH = Path("storage/document_registry.json")
EXTRACTED_TEXT_DIR = Path("storage/extracted_text")
OUTPUT_DIR = Path("storage/instruments")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def load_extracted_pages(file_path: Path) -> list:
    """
    Load extracted text pages for a PDF
    """
    safe_name = (
        file_path
        .relative_to(Path("data/notifications"))
        .as_posix()
        .replace("/", "__")
        .replace(".pdf", ".json")
    )
    json_path = EXTRACTED_TEXT_DIR / safe_name

    if not json_path.exists():
        return []

    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("pages", [])


def build_document_text(pages: list) -> str:
    """
    Preserve page order and boundaries
    """
    blocks = []
    for p in pages:
        text = p.get("text", "").strip()
        if text:
            blocks.append(f"[PAGE {p.get('page_no')}]\n{text}")
    return "\n\n".join(blocks)


# --------------------------------------------------
# MAIN GROUPER
# --------------------------------------------------
def run_instrument_grouper():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError("document_registry.json not found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    # -----------------------------
    # Group by group_id
    # -----------------------------
    instruments = {}

    for entry in registry:
        group_id = entry["group_id"]
        instruments.setdefault(group_id, []).append(entry)

    print(f"📦 Total instruments found: {len(instruments)}\n")

    # -----------------------------
    # Process each instrument
    # -----------------------------
    for idx, (group_id, docs) in enumerate(instruments.items(), start=1):
        print(f"[{idx}/{len(instruments)}] Building: {group_id}", end="\r", flush=True)

        # Sort documents chronologically
        docs_sorted = sorted(
            docs,
            key=lambda d: (
                parse_date(d.get("issued_on")) or datetime.max,
                d["file_path"]
            )
        )

        composite_blocks = []
        file_paths = []

        for d in docs_sorted:
            pages = load_extracted_pages(Path(d["file_path"]))
            text = build_document_text(pages)

            header = (
                f"[{d['document_nature'].upper()} | "
                f"Issued: {d.get('issued_on') or 'UNKNOWN'} | "
                f"Source: {Path(d['file_path']).name}]\n"
            )

            composite_blocks.append(header + text)
            file_paths.append(d["file_path"])

        composite_text = "\n\n".join(composite_blocks)

        latest_date = None
        for d in docs_sorted:
            dt = parse_date(d.get("issued_on"))
            if dt and (not latest_date or dt > latest_date):
                latest_date = dt

        instrument_payload = {
            "group_id": group_id,
            "tax_type": docs_sorted[0]["tax_type"],
            "notification_no": docs_sorted[0]["notification_no"],
            "documents": [
                {
                    "document_nature": d["document_nature"],
                    "issued_on": d.get("issued_on"),
                    "identity_source": d.get("identity_source"),
                    "file_path": d["file_path"]
                }
                for d in docs_sorted
            ],
            "composite_text": composite_text,
            "latest_effective_date": (
                latest_date.date().isoformat() if latest_date else None
            ),
            "file_paths": file_paths
        }

        out_name = group_id.replace("::", "__").replace("/", "_") + ".json"
        out_path = OUTPUT_DIR / out_name

        out_path.write_text(
            json.dumps(instrument_payload, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print("\n\n✅ INSTRUMENT GROUPING COMPLETE")
    print(f"📂 Output directory: {OUTPUT_DIR.resolve()}")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    run_instrument_grouper()

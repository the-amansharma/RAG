import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pdfplumber

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_PDF_DIR = Path("data/notifications")
EXTRACTED_TEXT_DIR = Path("storage/extracted_text")
OUTPUT_REGISTRY = Path("storage/document_registry.json")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def extract_page_one_text_from_pdf(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return ""
            return pdf.pages[0].extract_text() or ""
    except Exception:
        return ""


def extract_page_one_text_from_extracted_json(pdf_path: Path) -> str:
    safe_name = (
        pdf_path.relative_to(BASE_PDF_DIR)
        .as_posix()
        .replace("/", "__")
        .replace(".pdf", ".json")
    )
    json_path = EXTRACTED_TEXT_DIR / safe_name

    if not json_path.exists():
        return ""

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        pages = data.get("pages", [])
        if not pages:
            return ""
        return pages[0].get("text", "") or ""
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    return (
        text.replace("\n", " ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u00a0", " ")
        .strip()
    )


def normalize_date(day, month, year):
    try:
        return datetime.strptime(
            f"{day} {month} {year}", "%d %B %Y"
        ).date().isoformat()
    except Exception:
        return None


def make_ungrouped_id(pdf_path: Path) -> str:
    h = hashlib.md5(str(pdf_path).encode("utf-8")).hexdigest()[:8]
    return f"UNGROUPED::{h}"

# --------------------------------------------------
# REGEX PATTERNS
# --------------------------------------------------
NOTIFICATION_PATTERNS = [
    r'Notification\s+No\.?\s*(\d{1,3})\s*[/-]\s*(\d{4})',
    r'No\.?\s*(\d{1,3})\s*[/-]\s*(\d{4})',
    r'(\d{1,3})\s*[/-]\s*(\d{4})\s*[-–—].*?\(Rate\)',
]

DATE_PATTERNS = [
    r'New\s+Delhi,\s+the\s+(\d{1,2})(st|nd|rd|th)?\s*([A-Za-z]+),\s*(\d{4})',
    r'Dated\s+the\s+(\d{1,2})(st|nd|rd|th)?\s*([A-Za-z]+),\s*(\d{4})',
]

# --------------------------------------------------
# CORE EXTRACTORS
# --------------------------------------------------
def extract_notification_no(text: str):
    for p in NOTIFICATION_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    return None


def extract_issued_date(text: str):
    for p in DATE_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return normalize_date(m.group(1), m.group(3), m.group(4))
    return None


def extract_document_nature(text: str):
    t = text.lower()
    if "corrigendum" in t:
        return "corrigendum"
    if re.search(r'\bamendment\b|\bsubstitution\b|\bomission\b|\binsertion\b', t):
        return "amendment"
    return "original"


def extract_tax_type(text: str, pdf_path: Path):
    t = text.lower()

    if "central tax" in t or "cgst" in t:
        return "Central Tax"
    if "integrated tax" in t or "igst" in t:
        return "Integrated Tax"
    if "union territory tax" in t or "utgst" in t:
        return "Union Territory Tax"
    if "cess" in t:
        return "Compensation Cess"

    p = str(pdf_path).lower()
    if "central tax" in p or "cgst" in p:
        return "Central Tax"
    if "integrated tax" in p or "igst" in p:
        return "Integrated Tax"
    if "union territory tax" in p or "utgst" in p:
        return "Union Territory Tax"
    if "cess" in p:
        return "Compensation Cess"

    return None

# --------------------------------------------------
# IDENTITY EXTRACTION (FINAL LOGIC)
# --------------------------------------------------
def extract_document_identity(pdf_path: Path, mode: str):
    if mode == "pdf":
        raw_text = extract_page_one_text_from_pdf(pdf_path)
    else:
        raw_text = extract_page_one_text_from_extracted_json(pdf_path)

    text = normalize_text(raw_text)
    if not text:
        return None

    tax_type = extract_tax_type(text, pdf_path)
    notification_no = extract_notification_no(text)
    issued_on = extract_issued_date(text)
    document_nature = extract_document_nature(text)

    # If even tax_type is missing, we cannot identify this document at all
    if not tax_type:
        return None

    # ✅ FINAL, CORRECT GROUPABILITY RULE
    is_groupable = bool(tax_type and notification_no)

    if is_groupable:
        group_id = f"{tax_type}::{notification_no}"
    else:
        group_id = make_ungrouped_id(pdf_path)

    return {
        "group_id": group_id,
        "tax_type": tax_type,
        "notification_no": notification_no,
        "issued_on": issued_on,
        "document_nature": document_nature,
        "source_type": "notification",
        "identity_source": mode,
        "file_path": str(pdf_path),
        "is_groupable": is_groupable
    }

# --------------------------------------------------
# MAIN RUNNER (WITH RESCAN LOOP)
# --------------------------------------------------
def run_identity_extraction():
    all_files = sorted(BASE_PDF_DIR.rglob("*.pdf"))
    registry = []
    skipped = all_files.copy()
    mode = "pdf"

    print(f"\n📄 Total PDF files: {len(all_files)}\n")

    while skipped:
        print(f"\n🔍 Processing mode: {mode.upper()}")
        remaining = []

        for idx, pdf_file in enumerate(skipped, start=1):
            print(
                f"[{idx}/{len(skipped)}] {pdf_file.relative_to(BASE_PDF_DIR)}",
                end="\r",
                flush=True
            )

            identity = extract_document_identity(pdf_file, mode)

            if identity:
                registry.append(identity)
            else:
                remaining.append(pdf_file)

        print(f"\n✔ Identified in this pass: {len(skipped) - len(remaining)}")
        print(f"❌ Still skipped: {len(remaining)}")

        skipped = remaining
        if not skipped:
            break

        choice = input(
            "\n👉 Re-scan skipped files using extracted-text PAGE-1? (y/n): "
        ).strip().lower()

        if choice != "y":
            print("\n⏹ User stopped reprocessing.")
            break

        mode = "extracted_text"

    OUTPUT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REGISTRY.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n✅ IDENTITY EXTRACTION COMPLETE")
    print(f"   ✔ Identified : {len(registry)}")
    print(f"   ❌ Skipped   : {len(skipped)}")
    print(f"   📄 Registry : {OUTPUT_REGISTRY.resolve()}")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    run_identity_extraction()

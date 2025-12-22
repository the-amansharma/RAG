import json
import pdfplumber
from pathlib import Path
import hashlib

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_PDF_DIR = Path("data/notifications")
OUTPUT_DIR = Path("storage/extracted_text")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# --------------------------------------------------
# CORE EXTRACTION
# --------------------------------------------------
def extract_pdf_pages(pdf_path: Path) -> dict:
    pages = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({
                    "page_no": i,
                    "text": text.strip()
                })
    except Exception as e:
        print(f"\n❌ Failed to extract: {pdf_path}\n   Reason: {e}")
        return None

    return {
        "source_file": pdf_path.name,
        "file_path": str(pdf_path),
        "file_hash": file_hash(pdf_path),
        "total_pages": len(pages),
        "pages": pages
    }

# --------------------------------------------------
# MAIN RUNNER
# --------------------------------------------------
def run_batch_extraction():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(BASE_PDF_DIR.rglob("*.pdf"))
    total_files = len(pdf_files)

    if total_files == 0:
        print("⚠️ No PDF files found.")
        return

    print(f"📄 Total PDF files found: {total_files}\n")

    extracted_count = 0
    failed_count = 0

    for idx, pdf_file in enumerate(pdf_files, start=1):
        relative_path = pdf_file.relative_to(BASE_PDF_DIR)

        print(
            f"[{idx}/{total_files}] Processing: {relative_path}",
            end="\r",
            flush=True
        )

        data = extract_pdf_pages(pdf_file)
        if not data:
            failed_count += 1
            continue

        # SAFE OUTPUT PATH (no collisions)
        safe_name = relative_path.as_posix().replace("/", "__").replace(".pdf", ".json")
        output_file = OUTPUT_DIR / safe_name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        extracted_count += 1

    print("\n\n✅ EXTRACTION COMPLETE")
    print(f"   ✔ Successfully extracted : {extracted_count}")
    print(f"   ❌ Failed extractions    : {failed_count}")
    print(f"   📂 Output directory     : {OUTPUT_DIR.resolve()}")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    run_batch_extraction()

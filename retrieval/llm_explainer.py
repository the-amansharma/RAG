import requests

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "mistral"   # change only if needed

# --------------------------------------------------
# STRICT PROMPT
# --------------------------------------------------
PROMPT_TEMPLATE = """
You are a GST legal explainer.

STRICT RULES (DO NOT BREAK):
- Explain ONLY the provided legal text.
- Do NOT add, infer, or assume any law, section, rate, or condition.
- Do NOT use outside knowledge.
- Do NOT generalize beyond this notification.
- If the legal text does not specify something, say exactly:
  "Not specified in this notification."
- Rephrase in clear, simple language.
- Do NOT mention anything outside the text.

Context:
Tax Type: {tax_type}
Notification No: {notification_no}
As on date: {as_on_date}

Legal Text:
----------------
{legal_text}
----------------

User Question:
{question}

Answer format:
- Clear explanation in paragraphs
- End with a citation line exactly like:
  Source: Notification No. {notification_no}, {tax_type}
"""

# --------------------------------------------------
# EXPLAINER
# --------------------------------------------------
def explain(question: str, instrument: dict, composite_text: str, as_on_date: str | None = None) -> str:
    prompt = PROMPT_TEMPLATE.format(
        tax_type=instrument["tax_type"],
        notification_no=instrument["notification_no"],
        as_on_date=as_on_date or "Latest applicable",
        legal_text=composite_text,
        question=question
    )

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    if resp.status_code != 200:
        raise RuntimeError(f"LLM error: {resp.status_code} → {resp.text}")

    return resp.json()["response"].strip()

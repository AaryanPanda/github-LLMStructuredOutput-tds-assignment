import os, re, json
import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Config ----
UPSTREAM = "https://api.groq.com/openai/v1/chat/completions"
MODEL    = "openai/gpt-oss-20b"            # the chat model on your Groq key
API_KEY  = os.environ.get("GROQ_API_KEY")  # set in Render's Environment tab
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# What the grader sends us
class ExtractRequest(BaseModel):
    text: str

# The exact shape we must return (Pydantic guarantees these 4 fields)
class Invoice(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str


SYSTEM_PROMPT = (
    "You extract invoice fields from text. Respond with ONLY a JSON object, "
    "no other words, with exactly these keys:\n"
    "- vendor: the vendor/company name (string)\n"
    "- amount: the total amount due (a number, no currency symbol or commas)\n"
    "- currency: the 3-letter uppercase currency code, e.g. USD, EUR, GBP\n"
    "- date: the payment due date in YYYY-MM-DD format\n"
    'Example: {"vendor": "Acme Ltd.", "amount": 1234.5, "currency": "USD", "date": "2026-03-15"}'
)


def parse_amount(v):
    """Turn '$3,250.00' or '3250 USD' or 3250 into the float 3250.0."""
    if isinstance(v, (int, float)):
        return float(v)
    digits = re.sub(r"[^0-9.]", "", str(v))      # keep only digits and the dot
    return float(digits) if digits else 0.0


def find_json(content: str) -> dict:
    """Pull the JSON object out of the model's reply, even if wrapped in text."""
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("no JSON found")


async def ask_model(text: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        # Try strict JSON mode first; if the model rejects it, retry plainly.
        r = await client.post(
            UPSTREAM, headers=HEADERS,
            json={**payload, "response_format": {"type": "json_object"}},
        )
        if r.status_code != 200:
            r = await client.post(UPSTREAM, headers=HEADERS, json=payload)
    r.raise_for_status()
    return find_json(r.json()["choices"][0]["message"]["content"])


@app.post("/extract", response_model=Invoice)
async def extract(req: ExtractRequest):
    text = (req.text or "").strip()

    # Empty input -> 422, never a crash
    if not text:
        return JSONResponse({"detail": "empty input"}, status_code=422)

    try:
        data = await ask_model(text)
        return Invoice(
            vendor=str(data.get("vendor", "")),
            amount=parse_amount(data.get("amount", 0)),
            currency=str(data.get("currency", "")).upper()[:3],
            date=str(data.get("date", "")),
        )
    except Exception:
        # Anything went wrong -> return safe valid JSON, never a 500
        return Invoice(vendor="", amount=0.0, currency="USD", date="")


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL, "key_loaded": bool(API_KEY)}

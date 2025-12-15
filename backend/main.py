# main.py
import os
import re
import json
import logging
import tempfile
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Any

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from openpyxl import Workbook
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not found in environment. Running in mock mode.")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cc-analyzer")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod (set your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------------------
# Helpers: Extract raw PDF text (no hard preprocessing)
# -----------------------------------------
def extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for page in doc:
        chunks.append(page.get_text("text") or "")
    return "\n".join(chunks)

# -----------------------------------------
# LLM Structured schema (per-file)
# -----------------------------------------
# Each file must produce metadata + transactions + summary + recommendations
response_schemas = [
    ResponseSchema(
        name="metadata",
        description=("Object containing file-level metadata: "
                     "card_last4 (string or empty), statement_month (YYYY-MM if available), statement_year (YYYY if available), bank (string if identifiable)")
    ),
    ResponseSchema(
        name="transactions",
        description=("Array of objects. Each must include keys: "
                     "date (MM/DD or MM/DD/YY or MM/DD/YYYY), merchant (string, non-empty), "
                     "amount (number, positive for charges), category (string).")
    ),
    ResponseSchema(
        name="summary",
        description="Object mapping category -> percentage of total spending (float, no % sign)."
    ),
    ResponseSchema(
        name="recommendations",
        description="Array of 2–5 short personalized recommendations based on the file's transactions."
    ),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# Prompt: tells the model to extract file metadata + transactions (prompt-only approach)
prompt_template = """
You are a financial statement extractor.

You will receive raw text copied from a single credit card statement PDF. It may contain headers, footers, page numbers, icons, and multi-line transactions.

Your job for THIS FILE:
1) Return metadata (card_last4 if visible, statement_month in YYYY-MM if visible, statement_year in YYYY if visible, bank name if visible).
2) Identify transactions ONLY. A valid transaction MUST:
   - Start when a DATE appears (MM/DD, MM/DD/YY, or MM/DD/YYYY).
   - Contain a merchant (human readable, contains letters).
   - Contain an amount (like 23.49, $23.49, 2,275.00, (15.99)). Parentheses indicate negative (credit/refund) — output positive charge amounts only (i.e., skip negative refunds in "amount" but you can keep them as separate records if you want, mark category "Refund" or ignore).
   - If a transaction is multi-line, MERGE lines until the first amount; that amount belongs to that transaction.
   - Mention card name 
3) Assign each credit card transaction to ONE category from the list below.
Allowed categories (choose EXACTLY one): 
[Groceries, Dining, Coffee & Tea, Subscriptions, Shopping, Bills & Utilities, Transportation, Travel, Entertainment, Health & Wellness, Education, Other]

Rules:
- Return ONLY the category name. No sentences.
- If unsure, choose "Other".
- Do NOT create any new categories.
- Do NOT return explanations or extra text.
4) Ignore totals, page headers/footers, rewards text, and other non-transaction noise.

Return STRICT JSON that matches the schema below. Do NOT include extra keys or prose.

Schema instructions:
{format_instructions}

RAW FILE TEXT BELOW:
{statement}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # deterministic

chain = prompt | llm

# -----------------------------------------
# Normalization helpers (backend canonicalization)
# -----------------------------------------
DATE_RX = re.compile(r"^\s*(\d{1,2}/\d{1,2}(?:/\d{2,4})?)")
AMOUNT_RX = re.compile(r"\(?-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})\)?")

ICON_CHARS_PATTERN = r"[•◆▪▸►▶⧫★☆⚫⚪●]+"

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def amount_to_float(tok: str) -> float:
    if tok is None:
        return 0.0
    s = str(tok).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
    try:
        v = float(s)
    except Exception:
        v = 0.0
    return -v if neg else v

def normalize_transaction(tx: Dict[str, Any]) -> Dict[str, Any]:
    # canonicalize amount, merchant, category, date
    out = {}
    out["date"] = norm_spaces(tx.get("date", ""))
    out["merchant"] = norm_spaces(re.sub(ICON_CHARS_PATTERN, "", str(tx.get("merchant", ""))))
    # coerce numeric amount
    try:
        out["amount"] = float(tx.get("amount", 0.0))
    except Exception:
        out["amount"] = amount_to_float(tx.get("amount", 0.0))
    out["category"] = norm_spaces(tx.get("category", "Other"))
    # optional preserve raw payload
    if "raw" in tx:
        out["raw"] = tx["raw"]
    return out

# -----------------------------------------
# Suspicious / duplicate detection (simple rules)
# -----------------------------------------
def detect_suspicious_and_duplicates(all_txns: List[Dict[str, Any]]):
    suspicious = []
    duplicates = []
    # rule a: identical merchant + amount appearing >1 time across files within 7 days -> duplicate suspect
    key_counts = Counter((t["merchant"].lower(), round(float(t["amount"]), 2)) for t in all_txns)
    for (merchant, amt), cnt in key_counts.items():
        if cnt > 1:
            duplicates.append({"merchant": merchant, "amount": amt, "count": cnt})

    # rule b: small ATM fees or service fees (keywords)
    fee_keywords = ["atm fee", "overdraft", "service fee", "late fee", "cash advance", "atm"]
    for t in all_txns:
        desc = (t["merchant"] or "").lower()
        if any(k in desc for k in fee_keywords) and abs(float(t["amount"])) > 0:
            suspicious.append({"merchant": t["merchant"], "amount": t["amount"], "reason": "Fee or service charge"})

    # rule c: recurring subscriptions detection (same merchant monthly)
    # build merchant -> months seen
    months_by_merchant = defaultdict(set)
    for t in all_txns:
        mth = t.get("statement_month") or t.get("statement_year") or t.get("date", "")[:7]  # best-effort
        months_by_merchant[t["merchant"]].add(mth)
    for merchant, mset in months_by_merchant.items():
        if len([m for m in mset if m]) >= 2:
            suspicious.append({"merchant": merchant, "amount": None, "reason": "Recurring / possible subscription", "months": sorted(list(mset))})

    return {"duplicates": duplicates, "suspicious": suspicious}

# -----------------------------------------
# Combine + compute analytics
# -----------------------------------------
def compute_aggregates(all_txns: List[Dict[str, Any]]):
    monthly = defaultdict(float)
    card_spend = defaultdict(float)
    category_spend = defaultdict(float)
    total_spent = 0.0

    for t in all_txns:
        amt = float(t.get("amount", 0.0))
        # only positive charges contribute to spending totals
        if amt <= 0:
            continue
        total_spent += amt
        # month inference: prefer statement_month, else parse date
        month = t.get("statement_month")
        if not month:
            # try to create YYYY-MM from date strings
            try:
                dt = datetime.strptime(t.get("date", ""), "%m/%d/%Y")
                month = dt.strftime("%Y-%m")
            except Exception:
                # try mm/dd/yy
                try:
                    dt = datetime.strptime(t.get("date", ""), "%m/%d/%y")
                    month = dt.strftime("%Y-%m")
                except Exception:
                    month = "unknown"
        monthly[month] += amt

        # card spending
        card = t.get("card_last4") or "unknown"
        card_spend[card] += amt

        # category
        cat = t.get("category") or "Other"
        category_spend[cat] += amt

    # compute percent shares
    summary = {k: round(v / (total_spent or 1.0) * 100, 2) for k, v in category_spend.items()}
    return {
        "monthly_spending": dict(monthly),
        "card_spend": dict(card_spend),
        "category_spend": dict(category_spend),
        "total_spent": round(total_spent, 2),
        "category_summary_percent": summary
    }

# -----------------------------------------
# Export helpers
# -----------------------------------------
def export_transactions_excel(all_txns: List[Dict[str, Any]], out_path: str):
    df = pd.DataFrame(all_txns)
    df = df[["statement_month", "card_last4", "date", "merchant", "amount", "category"]].fillna("")
    writer = pd.ExcelWriter(out_path, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name="transactions")
    writer.save()

def export_transactions_csv(all_txns: List[Dict[str, Any]], out_path: str):
    df = pd.DataFrame(all_txns)
    df = df[["statement_month", "card_last4", "date", "merchant", "amount", "category"]].fillna("")
    df.to_csv(out_path, index=False)

# -----------------------------------------
# Main: multi-file analyze endpoint
# -----------------------------------------
@app.post("/analyze")
async def analyze_statement(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    all_files_results = []
    all_txns = []

    # iterate each uploaded PDF
    for f in files:
        try:
            content = await f.read()
            raw_text = extract_pdf_text(content)
            log.info("Processing file %s length=%d chars", f.filename, len(raw_text))

            # Prepare and call LLM (prompt-only)
            llm_input = {
                "statement": raw_text[:200_000],  # limit for safety
                "format_instructions": format_instructions
            }

            # If no API key, fallback to a simple mock (for dev)
            if not OPENAI_API_KEY:
                # Simple mock: return zero transactions (developer can fill)
                parsed = {"metadata": {"card_last4": "", "statement_month": ""}, "transactions": [], "summary": {}, "recommendations": []}
            else:
                res = chain.invoke(llm_input)
                try:
                    parsed = parser.parse(res.content or "")
                except Exception as pe:
                    log.warning("Parser error for %s: %s", f.filename, pe)
                    # attempt minimal JSON load fallback
                    try:
                        parsed = json.loads(res.content or "{}")
                    except Exception:
                        parsed = {"metadata": {}, "transactions": [], "summary": {}, "recommendations": []}

            # Normalize parsed contents
            metadata = parsed.get("metadata", {}) or {}
            txns_raw = parsed.get("transactions", []) or []
            txns_normalized = []
            for tx in txns_raw:
                nt = normalize_transaction(tx)
                # attach file-level metadata
                nt["source_file"] = f.filename
                nt["card_last4"] = metadata.get("card_last4") or ""
                nt["statement_year"] = metadata.get("statement_year") or ""
                nt["statement_month"] = metadata.get("statement_month") or ""
                txns_normalized.append(nt)
                all_txns.append(nt)

            file_result = {
                "file": f.filename,
                "metadata": metadata,
                "transactions": txns_normalized,
                "summary": parsed.get("summary", {}),
                "recommendations": parsed.get("recommendations", [])
            }
            all_files_results.append(file_result)

        except Exception as e:
            log.exception("Failed processing file %s: %s", f.filename if f else "unknown", e)
            file_result = {"file": getattr(f, "filename", "unknown"), "error": str(e)}
            all_files_results.append(file_result)
            continue

    # After processing all files -> compute combined analytics
    aggregates = compute_aggregates(all_txns)
    flags = detect_suspicious_and_duplicates(all_txns)

    # Build global recommendations (simple merge of file recs plus LLM-level suggestions)
    global_recommendations = []
    for fr in all_files_results:
        for r in fr.get("recommendations", [])[:3]:
            if r not in global_recommendations:
                global_recommendations.append(r)
    # simple additional suggestions based on flags
    if flags["duplicates"]:
        global_recommendations.append("We found possible duplicate charges across statements — review duplicates.")
    if flags["suspicious"]:
        global_recommendations.append("Some transactions look like fees or recurring subscriptions — consider reviewing recurring charges.")

    response = {
        "files": all_files_results,
        "combined": {
            "all_transactions": all_txns,
            "aggregates": aggregates,
            "flags": flags,
            "global_recommendations": global_recommendations
        }
    }

    return JSONResponse(content=response, media_type="application/json; charset=utf-8")


# -----------------------------------------
# Exports: Excel & CSV for combined transactions
# -----------------------------------------
@app.post("/export/excel")
async def export_excel(files: List[UploadFile] = File(...)):
    # For convenience: reuse /analyze to get combined result, then export
    analyze_resp = await analyze_statement(files)
    data = analyze_resp.body.decode("utf-8")
    parsed = json.loads(data)
    all_txns = parsed.get("combined", {}).get("all_transactions", [])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    tmp.close()
    export_transactions_excel(all_txns, tmp.name)
    return FileResponse(tmp.name, filename="transactions.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.post("/export/csv")
async def export_csv(files: List[UploadFile] = File(...)):
    analyze_resp = await analyze_statement(files)
    data = analyze_resp.body.decode("utf-8")
    parsed = json.loads(data)
    all_txns = parsed.get("combined", {}).get("all_transactions", [])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    export_transactions_csv(all_txns, tmp.name)
    return FileResponse(tmp.name, filename="transactions.csv", media_type="text/csv")

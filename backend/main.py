from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import re,sys
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------- PDF text helpers --------
def extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_transactions(raw_text: str) -> str:
    """
    Keep only lines that look like transactions:
    MM/DD description amount
    """
    pattern = r"(\d{2}/\d{2})\s+(.+?)\s+(-?\$?\d+\.\d{2})"
    matches = re.findall(pattern, raw_text)

    transactions = []
    for m in matches:
        date, merchant, amount = m
        # normalize amount (remove $ sign)
        amt = amount.replace("$", "")
        transactions.append(f"{date} {merchant.strip()} {amt}")
    return "\n".join(transactions)  


# -------- LLM setup --------
response_schemas = [
    ResponseSchema(
        name="transactions",
        description="List of transactions. Fields: date (MM/DD), optional post_date, merchant, amount(number), category."
    ),
    ResponseSchema(
        name="summary",
        description="Dict of top spending categories with numeric % share."
    ),
    ResponseSchema(
        name="recommendations",
        description="List of 2–3 money-saving recommendations."
    ),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
You are a financial assistant.
Input: list of raw transaction lines from a credit card statement.

Tasks:
1. Parse each into:
   - date (MM/DD)
   - optional post_date (if a second MM/DD appears after the first)
   - merchant (clean, no trailing numbers/codes like ZIP or last-4)
   - amount (number, positive=charge, negative=refund or payment received; handle parentheses as negative)
   - category (dynamic, inferred from merchant patterns)
2. Summarize all categories with percentage share.
   - IMPORTANT: output numbers only (floats), not strings, not with "%" symbol.

3. Suggest 2–3 money-saving recommendations.

Return valid JSON strictly following this schema:
{format_instructions}

Transaction lines:
{transactions}
""")
chain = LLMChain(llm=llm, prompt=prompt)

# -------- API Endpoint --------
@app.post("/analyze")
async def analyze_statement(file: UploadFile = File(...)):
    print("✅ Received request", file=sys.stderr)
    try:
        if not file.filename.lower().endswith(".pdf"):
            return {"error": "Only PDF files are supported."}

        contents = await file.read()

        # Extract raw text
        raw_text = extract_pdf_text(contents)
        # Ask LLM to parse lines
        result = chain.invoke({
            "transactions": (raw_text),
            "format_instructions": format_instructions
        })
        parsed = parser.parse(result["text"])
        # --- Clean Summary for Pie chart --- 
        summary ={}
        for k,v in (parsed.get("summary") or {}).items():
            try:
                summary[k]=float(str(v).replace("%","").strip())
            except Exception:
                continue
        
        # --- If LLM skipped summary ---
        if not summary and parsed.get("transactions"):
            cat_total={}
            for t in parsed.get("transactions"):
                cat=t.get("category","other")
                amt_str=str(t.get("amount","0")).replace("%","").replace(",","").strip()
                try:
                    amt = float(amt_str) if amt_str else 0
                except:
                    amt = 0
                cat_total[cat]=cat_total.get(cat,0)+amt
            total=sum(cat_total.values()) or 1 
            summary={cat:round((amt/total) * 100,2) for cat,amt in cat_total.items()}
        # --- Compute total spent ---
        total_spent = 0.0
        for t in (parsed.get("transactions") or []):
            try:
                amt = str(t.get("amount", "0")).replace("$", "").replace(",", "").strip()
                if amt.startswith("(") and amt.endswith(")"):
                    amt = "-" + amt[1:-1]
                total_spent += float(amt)
            except:
                continue




        return {
            "file_name": file.filename,
            "transactions": parsed.get("transactions", []),
            "recommendations": parsed.get("recommendations", []),
            "total_spent": round(total_spent, 2),
            "summary": summary,

        }

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

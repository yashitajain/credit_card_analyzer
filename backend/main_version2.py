import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# --- FastAPI App ---
app = FastAPI()

# CORS so frontend (Next.js) can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PDF Extraction ---
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o-mini")

# Schema for structured output
response_schemas = [
    ResponseSchema(
        name="transactions",
        description=(
            "List of transactions. Each must include: "
            "date (MM/DD or MM/DD/YY), merchant(string), amount(number), "
            "category (Dining, Groceries, Travel, Subscriptions, Shopping, Utilities, Other, etc.)."
        ),
    ),
    ResponseSchema(
        name="summary",
        description="Top spending categories with approximate percentage share."
    ),
    ResponseSchema(
        name="recommendations",
        description="List of recommendations to save money."
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template("""
You are a financial assistant.

Extract structured transactions from this credit card statement text.  
Each transaction must include:
- date (MM/DD or MM/DD/YY)
- merchant (clean name, merge multi-line descriptions if needed)
- amount (number, positive for charges, negative for refunds)
- category (infer dynamically: Dining, Coffee, Groceries, Travel, Shopping, Entertainment, Utilities, Subscriptions, Auto, Other)

Then:
- Summarize top 3 spending categories with % share.
- Suggest 2–3 money-saving recommendations.

Return valid JSON strictly following this schema:
{format_instructions}

Statement Text:
{statement}
""")

chain = prompt | llm

# --- API Endpoint ---
@app.post("/analyze")
async def analyze_statement(file: UploadFile = File(...)):
    try:
        # Step 1: Extract PDF text
        contents = await file.read()
        raw_text = extract_pdf_text(contents)
        raw_text = raw_text.encode("utf-8", "ignore").decode("utf-8")

        # Step 2: Run through LLM
        result = chain.invoke({
            "statement": raw_text[:12000],  # truncate if needed
            "format_instructions": format_instructions,
        })

        # Step 3: Parse result
        parsed = parser.parse(result.content)

        # Step 4: Return JSON response
        return JSONResponse(
            content={
                "file_name": file.filename,
                "transactions": parsed.get("transactions", []),
                "summary": parsed.get("summary", {}),
                "recommendations": parsed.get("recommendations", []),
                "preview_text": raw_text[:300],
            },
            media_type="application/json; charset=utf-8",
        )

    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
                "file_name": file.filename if file else "unknown",
                "status": "error",
            }
        )

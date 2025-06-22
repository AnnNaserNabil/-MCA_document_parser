import getpass
import os
 
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import json, re

# Load environment variables
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# Load raw PDF text using MuPDF
def load_pdf_text(path):
    doc = fitz.open(path)
    all_text = ""
    for page in doc:
        text = page.get_text()
        if text:
            all_text += text + "\n"
    return all_text.strip()

# Extract structured JSON from raw text
def ask_llm_for_adt1_fields(pdf_text):
    messages = [
        SystemMessage(content="You're an assistant that extracts key data from Indian Form ADT-1 audit appointment forms."),
        HumanMessage(content=f"""
Given the content below, extract the following fields and return the result as valid JSON:

- company_name
- cin
- email_of_company
- audit_account_period
- registered_office
- appointment_date
- number_of_years_to_audit
- auditor_name
- auditor_address
- auditor_email
- auditor_frn_or_membership
- appointment_type

Text:
{pdf_text}
""")
    ]
    response = llm(messages)
    return response.content

# Clean and save JSON to file
def clean_and_save(json_text, filename="output.json"):
    cleaned = re.sub(r"```json|```", "", json_text).strip()
    try:
        data = json.loads(cleaned)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print("✅ Data saved to", filename)
        return data
    except json.JSONDecodeError as e:
        print("❌ Failed to decode JSON:", e)
        print("Raw response was:\n", cleaned)
        return None

# Generate user-friendly summary
def generate_summary_from_json(data, filename="summary.txt"):
    messages = [
        SystemMessage(content="You're an AI assistant summarizing company filings in human-friendly language."),
        HumanMessage(content=f"""
Based on the following structured data from a Form ADT-1 filing, generate a clear, professional, and non-technical 3–5 line summary. Keep it concise and human-readable.

Data:
{json.dumps(data, indent=2)}
""")
    ]
    response = llm(messages)
    summary = response.content.strip()
    with open(filename, "w") as f:
        f.write(summary)
    print(f"📝 Summary saved to {filename}")

# 🧠 New Layer: Extract Deep Insights from Full PDF Text
def generate_additional_insights(raw_text, filename="insights.txt"):
    messages = [
        SystemMessage(content="You're an analyst reviewing a government form (Form ADT-1)."),
        HumanMessage(content=f"""
Read the full PDF content below and list **any important additional observations** that are not already included in the structured fields:
- appointment context (casual vacancy, reappointment, C&AG, etc.)
- resolution or consent details
- whether board or AGM approved it
- any irregularities, notes, or regulatory flags
- names/dates of attachments, if mentioned

Be concise. List them in bullet points or short paragraphs.

Text:
{raw_text}
""")
    ]
    response = llm(messages)
    insights = response.content.strip()
    with open(filename, "w") as f:
        f.write(insights)
    print(f"🔍 Additional insights saved to {filename}")

# 🚀 Master Execution Flow
if __name__ == "__main__":
    pdf_path = "ADT1.pdf"

    # Step 1: Extract raw text
    text = load_pdf_text(pdf_path)

    # Step 2: Generate structured JSON
    raw_json = ask_llm_for_adt1_fields(text)
    data = clean_and_save(raw_json)

    # Step 3: Human-readable summary
    if data:
        generate_summary_from_json(data)

    # Step 4: Deep insights from full raw PDF
    generate_additional_insights(text)

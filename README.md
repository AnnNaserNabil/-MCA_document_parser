# 📄 Filesure ADT-1 PDF Extractor

This project is a take-home assignment for Filesure. It demonstrates how to use Python and AI (Google Gemini via LangChain) to extract and summarize information from **Indian Form ADT-1** filings — which report the appointment of statutory auditors to the Ministry of Corporate Affairs (MCA).

---

## ✅ What This Tool Does

Given an ADT-1 PDF (digitally signed MCA filing):

1. **Extracts structured data** such as:
   - Company name, CIN, auditor name & address, date of appointment, etc.
2. **Generates a human-readable summary**
3. **Derives additional deep insights** from the raw PDF text
4. Outputs:
   - `output.json`: structured key fields
   - `summary.txt`: plain English summary
   - `insights.txt`: LLM-generated contextual analysis

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [PyMuPDF (`fitz`)](https://github.com/pymupdf/PyMuPDF) – For extracting raw PDF text
- [LangChain](https://www.langchain.com/) – For managing LLM prompts
- [Google Gemini Pro (via `langchain_google_genai`)](https://ai.google.dev/)
- `.env` for secure API key storage

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://https://github.com/AnnNaserNabil/-MCA_document_parser
cd -MCA_document_parser
```

2. Install dependencies

```bash
pip install -r requirements.txt

```

3. Create .env file with your Gemini API key

```
GOOGLE_API_KEY=your-gemini-api-key
```

4. Place your ADT-1 PDF in the root folder
Name it ADT1.pdf or change the script to use a different filename.


🚀 Run the Pipeline

```bash

python extractor.py

```
### This will create:

output.json → structured audit appointment info

summary.txt → simple AI-generated summary

insights.txt → deeper insights from the full text







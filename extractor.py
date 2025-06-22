"""
ADT-1 Form Parser

This script extracts and processes information from Indian Form ADT-1 (Appointment of Auditor) PDFs.
It uses Google's Gemini AI via LangChain to parse and analyze the document structure.
"""

# Standard library imports
import getpass  # For securely getting the API key from user input
import os       # For file operations and environment variables
import json     # For JSON data handling
import re       # For regular expression operations
from typing import Optional, Dict, Any  # For type hints

# Check if Google API key is already set in environment variables
# If not, prompt the user to enter it securely
if "GOOGLE_API_KEY" not in os.environ:
    # Securely prompt user for API key without echoing to screen
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Third-party library imports (after environment setup)
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini AI interface
from langchain.schema import SystemMessage, HumanMessage     # For chat message formatting
from dotenv import load_dotenv  # For loading .env files
import fitz  # PyMuPDF library for PDF processing

# Load environment variables from .env file if present in the project directory
# This allows for storing configuration like API keys outside the code
load_dotenv()

# Initialize the Google Generative AI chat model with specific parameters:
# - model: "gemini-2.0-flash" is the model version being used
# - temperature: 0.2 for more focused and deterministic responses
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

def load_pdf_text(path: str) -> str:
    """
    Extract text content from a PDF file using PyMuPDF (fitz).
    
    Args:
        path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content from all pages of the PDF, or empty string on error
    """
    try:
        # Open the PDF document using PyMuPDF
        doc = fitz.open(path)
        
        # Initialize an empty string to store all extracted text
        all_text = ""
        
        # Iterate through each page in the document
        for page in doc:
            # Extract text from the current page
            text = page.get_text()
            
            # If text was extracted, add it to our result with a newline
            if text:
                all_text += text + "\n"
                
        # Close the document to free resources
        doc.close()
        
        # Return the accumulated text with leading/trailing whitespace removed
        return all_text.strip()
        
    except Exception as e:
        # Print error message and return empty string if something goes wrong
        print(f"❌ Error loading PDF: {e}")
        return ""

def ask_llm_for_adt1_fields(pdf_text: str) -> str:
    """
    Extract structured data from ADT-1 form text using Google's Gemini AI.
    
    This function sends the extracted PDF text to Google's Gemini AI model with specific
    instructions to extract key fields commonly found in ADT-1 forms.
    
    Args:
        pdf_text (str): Raw text extracted from the PDF document
        
    Returns:
        str: JSON-formatted string containing the extracted fields, or an empty JSON object on error
    """
    # Prepare the messages for the LLM conversation
    messages = [
        # System message sets the context for the AI
        SystemMessage(content="You're an assistant that extracts key data from Indian Form ADT-1 audit appointment forms."),
        # Human message provides the actual instruction and data
        HumanMessage(content=f"""
Given the content below, extract the following fields and return the result as valid JSON:

- company_name: Full legal name of the company
- cin: Corporate Identification Number
- email_of_company: Company's contact email
- audit_account_period: Period covered by the audit
- registered_office: Company's registered office address
- appointment_date: Date of auditor appointment
- number_of_years_to_audit: Duration of the appointment
- auditor_name: Name of the appointed auditor/firm
- auditor_address: Business address of the auditor
- auditor_email: Contact email of the auditor
- auditor_frn_or_membership: FRN or membership number of the auditor
- appointment_type: Type of appointment (e.g., first, reappointment, casual vacancy)

Text:
{pdf_text}
""")
    ]
    
    try:
        # Send the prepared messages to the LLM and get the response
        response = llm(messages)
        # Return the content of the LLM's response
        return response.content
    except Exception as e:
        # Log any errors and return an empty JSON object
        print(f"❌ Error querying LLM: {e}")
        return "{}"

def clean_and_save(json_text: str, filename: str = "output.json") -> Optional[Dict[str, Any]]:
    """
    Clean and save JSON data to a file.
    
    This function takes a potentially dirty JSON string (may contain markdown formatting),
    cleans it, validates it, and saves it to a file. It handles various error cases
    and provides helpful error messages.
    
    Args:
        json_text (str): Raw JSON string that may contain markdown code blocks
        filename (str): Output filename where the cleaned JSON will be saved
                       (default: "output.json")
        
    Returns:
        Optional[Dict]: Parsed JSON data as a Python dictionary if successful,
                      None if there was an error
    """
    # Remove markdown code block markers (```json and ```) and trim whitespace
    cleaned = re.sub(r"```(?:json)?", "", json_text).strip()
    
    try:
        # Parse the cleaned JSON string into a Python dictionary
        data = json.loads(cleaned)
        
        # Write the parsed data to the specified file with pretty-printing
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
            
        print("✅ Data saved to", filename)
        return data
        
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors specifically
        print("❌ Failed to decode JSON:", e)
        print("Raw response was:\n", cleaned)
        return None
        
    except Exception as e:
        # Handle any other errors (e.g., file writing errors)
        print(f"❌ Error saving file {filename}: {e}")
        return None

def generate_summary_from_json(data: Dict[str, Any], filename: str = "summary.txt") -> None:
    """
    Generate a human-readable summary from structured ADT-1 data.
    
    This function takes the structured data extracted from the ADT-1 form and
    uses an LLM to generate a concise, natural language summary that highlights
    the key points in an easily understandable format.
    
    Args:
        data (Dict): Parsed ADT-1 form data as a dictionary
        filename (str): Output filename where the summary will be saved
                      (default: "summary.txt")
    """
    # Prepare the messages for the LLM conversation
    messages = [
        # System message sets the tone and style for the summary
        SystemMessage(content="You're an AI assistant summarizing company filings in human-friendly language."),
        # Human message provides the data and instructions
        HumanMessage(content=f"""
Based on the following structured data from a Form ADT-1 filing, generate a clear, 
professional, and non-technical 3–5 line summary. Keep it concise and human-readable.
Focus on the key points: company name, auditor details, and appointment specifics.

Data:
{json.dumps(data, indent=2)}
""")
    ]
    
    try:
        # Get the summary from the LLM
        response = llm(messages)
        summary = response.content.strip()
        
        # Save the generated summary to the specified file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary)
            
        print(f"📝 Summary saved to {filename}")
        
    except Exception as e:
        # Handle any errors during the summary generation
        print(f"❌ Error generating summary: {e}")

def generate_additional_insights(raw_text: str, filename: str = "insights.txt") -> None:
    """
    Extract additional insights from the raw PDF text using AI analysis.
    
    This function analyzes the complete text of the ADT-1 form to identify and extract
    any important information that might not be captured in the structured fields.
    It looks for contextual details, approval information, and potential red flags.
    
    Args:
        raw_text (str): The complete text content extracted from the PDF
        filename (str): Output filename where insights will be saved
                      (default: "insights.txt")
    """
    # Prepare the messages for the LLM conversation
    messages = [
        # System message sets the analytical context
        SystemMessage(content="You're an analyst reviewing a government form (Form ADT-1)."),
        # Human message provides detailed instructions and the text to analyze
        HumanMessage(content=f"""
Carefully review the full PDF content below and identify any significant information 
that wasn't captured in the standard fields. Focus on:

1. Appointment context (e.g., casual vacancy, reappointment, C&AG appointment)
2. Specific resolution or consent details
3. Approval details (board meeting, AGM, etc.)
4. Any irregularities, special notes, or regulatory flags
5. References to attachments or supporting documents

Format your response as clear, concise bullet points. Only include notable findings.

Text:
{raw_text[:30000]}  # Limit to first 30k chars to avoid token limits
""")
    ]
    
    try:
        # Get insights from the LLM
        response = llm(messages)
        insights = response.content.strip()
        
        # Save the insights to the specified file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(insights)
            
        print(f"🔍 Additional insights saved to {filename}")
        
    except Exception as e:
        # Handle any errors during insight generation
        print(f"❌ Error generating insights: {e}")

def main() -> None:
    """
    Main execution function for the ADT-1 form parser.
    
    This function orchestrates the entire PDF processing pipeline:
    1. Loads and validates the input PDF
    2. Extracts text content from the PDF
    3. Processes the text to extract structured data
    4. Generates a human-readable summary
    5. Extracts additional insights from the full text
    
    The function handles errors gracefully and provides user-friendly status updates.
    """
    # Define the path to the input PDF file
    pdf_path = "ADT1.pdf"
    
    # Check if the input file exists before proceeding
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found - {pdf_path}")
        print("Please place your ADT-1 PDF in the same directory as this script.")
        print("The file should be named 'ADT1.pdf' or update the script with the correct filename.")
        return

    try:
        # Step 1: Extract raw text from the PDF
        print("🔍 Extracting text from PDF...")
        text = load_pdf_text(pdf_path)
        
        # Verify that text was successfully extracted
        if not text:
            print("❌ No text could be extracted from the PDF. The file might be corrupted or password-protected.")
            return

        # Step 2: Generate structured JSON from the extracted text
        print("🧠 Analyzing document structure and extracting data...")
        raw_json = ask_llm_for_adt1_fields(text)
        data = clean_and_save(raw_json)

        # Proceed only if data was successfully extracted and saved
        if data:
            # Step 3: Generate a human-readable summary
            print("📄 Generating summary...")
            generate_summary_from_json(data)

            # Step 4: Extract additional insights from the full text
            print("💡 Extracting additional insights...")
            generate_additional_insights(text)
            
            # Display completion message with output files
            print("\n✅ Processing complete! The following files have been generated:")
            print(f"   - output.json: Structured data in JSON format")
            print(f"   - summary.txt: Human-readable summary of key information")
            print(f"   - insights.txt: Additional observations and analysis")
        else:
            print("❌ Failed to process the document. Please check the error messages above.")
            print("   - Ensure the PDF contains valid ADT-1 form data")
            print("   - Check that the PDF is not scanned or image-based")
            
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\n🛑 Operation cancelled by user.")
    except Exception as e:
        # Catch any unexpected errors and display a helpful message
        print(f"\n❌ An unexpected error occurred: {e}")
        print("   - Please check your internet connection and API key")
        print("   - Ensure you have the required permissions to read/write files")

# Standard Python idiom to check if this script is being run directly
# (as opposed to being imported as a module)
if __name__ == "__main__":
    # Display a welcome message with basic instructions
    print("\n" + "="*60)
    print("ADT-1 Form Parser".center(60))
    print("="*60)
    print("This tool extracts and analyzes auditor appointment information from ADT-1 forms.")
    print("="*60 + "\n")
    
    # Start the main processing
    main()

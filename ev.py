import streamlit as st
import PyPDF2  # PyPDF2 for PDF text extraction
import requests
from fpdf import FPDF
from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# Function to extract text from PDF using PyPDF2
def extract_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from scanned PDFs using OCR
def extract_text_with_ocr(pdf_file):
    try:
        # Convert the first page of the PDF to an image
        reader = PyPDF2.PdfReader(pdf_file)
        page = reader.pages[0]
        
        # This function assumes you extract an image from the page, so we mock this by saving the page as an image and loading it
        pix = page.extract_text()  # You should use PDF2's method to extract an image of the page
        img = Image.open(io.BytesIO(pix))
        
        # Perform OCR on the image to extract text
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Error extracting text with OCR: {e}")
        return ""

# Function to call GROQ API for answer evaluation using GROQ API Key
def get_groq_response(input_text, pdf_content, prompt):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        endpoint = "https://api.groq.ai/v1/evaluate"  # Replace with actual GROQ endpoint if different
        
        # Construct the request payload (this depends on GROQ API specifications)
        payload = {
            "input_text": input_text,
            "pdf_content": pdf_content,
            "prompt": prompt
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Sending the request to the GROQ API
        response = requests.post(endpoint, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()  # Assuming the result is in JSON format
            return result.get('evaluation_result', "Error: No evaluation result")
        else:
            st.error(f"Error during GROQ API call: {response.status_code} - {response.text}")
            return "Error during evaluation"
    except Exception as e:
        st.error(f"Error during GROQ API call: {e}")
        return "Error during evaluation"

# Function to generate a report PDF
def generate_report(evaluation_result):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Question-Answer Evaluation Report", ln=True, align='C')
    
    for question, answer, result in evaluation_result:
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Question: {question}", ln=True)
        pdf.cell(200, 10, txt=f"Answer: {answer}", ln=True)
        pdf.cell(200, 10, txt=f"Result: {result}", ln=True)
    
    return pdf.output(dest='S')

# Streamlit app layout
st.title("Question-Answer Evaluator")

# File upload
question_pdf = st.file_uploader("Upload Question PDF", type="pdf")
answer_pdfs = st.file_uploader("Upload Answer PDFs", type="pdf", accept_multiple_files=True)

if question_pdf and answer_pdfs:
    # Extract text from PDFs
    st.write("Extracting text from PDFs...")
    question_text = extract_pdf_text(question_pdf)
    
    # OCR extraction for scanned PDFs
    ocr_enabled = st.checkbox("Enable OCR for scanned PDFs", value=False)
    if ocr_enabled:
        st.write("Using OCR for text extraction...")
        answer_texts = [extract_text_with_ocr(file) for file in answer_pdfs]
    else:
        answer_texts = [extract_pdf_text(file) for file in answer_pdfs]

    # Display the extracted texts (for debugging purposes)
    st.subheader("Extracted Questions")
    st.text(question_text)

    st.subheader("Extracted Answers")
    for i, answer_text in enumerate(answer_texts):
        st.text(f"Answer {i+1}: {answer_text}")

    # Show a progress bar for evaluation
    progress_bar = st.progress(0)

    evaluation_results = []
    for i, answer_text in enumerate(answer_texts):
        st.write(f"Evaluating Answer {i+1}...") 
        progress_bar.progress((i + 1) / len(answer_texts))

        # Call the GROQ API to evaluate the answer (replacing placeholder logic)
        evaluation_result = get_groq_response(question_text, answer_text, "Evaluate this answer based on the question.")
        
        if "error" in evaluation_result:
            st.error(f"Error in evaluation for Answer {i+1}: {evaluation_result}")
        else:
            evaluation_results.append((question_text, answer_text, evaluation_result))
            st.success(f"Answer {i+1} evaluation complete!")

    # Generate and show the report after all evaluations
    st.write("All evaluations complete! Generating report...")
    report_pdf = generate_report(evaluation_results)
    
    # Provide download link for the report
    st.download_button(
        label="Download Evaluation Report",
        data=report_pdf,
        file_name="evaluation_report.pdf",
        mime="application/pdf"
    )

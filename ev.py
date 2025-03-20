import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import requests
import json
from io import BytesIO
from fpdf import FPDF
from dotenv import load_dotenv
import os
import pytesseract
from PIL import Image
import io
import google.generativeai as genai  # Importing google.generativeai for Gemini integration

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI API with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF using PyMuPDF
def extract_pdf_text(pdf_file):
    try:
        doc = fitz.open(pdf_file)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from scanned PDFs using OCR
def extract_text_with_ocr(pdf_file):
    try:
        # Convert the first page of the PDF to an image
        doc = fitz.open(pdf_file)
        page = doc[0]  # Assuming we want the first page only
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))

        # Perform OCR on the image to extract text
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Error extracting text with OCR: {e}")
        return ""

# Function to call Gemini API for answer evaluation using google.generativeai
def get_gemini_response(input_text, pdf_content, prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Use the gemini model from Google Generative AI
        response = model.generate_content([input_text, pdf_content, prompt])
        return response.text
    except Exception as e:
        st.error(f"Error during Gemini API call: {e}")
        return ""

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

        evaluation_result = get_gemini_response(question_text, answer_text, "Evaluate this answer based on the question.")
        
        if "error" in evaluation_result:
            st.error(f"Error in evaluation for Answer {i+1}: {evaluation_result['error']}")
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

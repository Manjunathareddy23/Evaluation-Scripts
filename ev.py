import streamlit as st
import pdfplumber  # Better PDF text extraction library
from fpdf import FPDF
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

# Function to extract text from PDF using pdfplumber
def extract_pdf_text(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
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
    
    # Extract text from answer PDFs
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

        # Simple evaluation based on text similarity or placeholder response
        evaluation_result = f"Evaluation for Answer {i+1}: [Placeholder Evaluation Logic]"

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

import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import requests
from io import BytesIO
import json
from fpdf import FPDF

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to call Gemini API for answer evaluation
def evaluate_answers(question_text, answer_text):
    api_url = "https://api.gemini.example/evaluate"  # Replace with Gemini API URL
    payload = {
        "questions": question_text,
        "answers": answer_text
    }
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",  # Replace with your API key
        "Content-Type": "application/json"
    }
    response = requests.post(api_url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        return response.json()  # Returns the evaluation result
    else:
        return {"error": "Error in API response"}

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
answer_pdf = st.file_uploader("Upload Answer PDF", type="pdf")

if question_pdf and answer_pdf:
    # Extract text from PDFs
    st.write("Extracting text from PDFs...")
    question_text = extract_pdf_text(question_pdf)
    answer_text = extract_pdf_text(answer_pdf)
    
    # Display the extracted texts (for debugging purposes)
    st.subheader("Extracted Questions")
    st.text(question_text)
    st.subheader("Extracted Answers")
    st.text(answer_text)

    # Call the Gemini API to evaluate answers
    st.write("Evaluating answers...")
    evaluation_result = evaluate_answers(question_text, answer_text)
    
    if "error" in evaluation_result:
        st.error(evaluation_result["error"])
    else:
        # Generate and show the report
        st.write("Evaluation complete! Generating report...")
        report_pdf = generate_report(evaluation_result)
        
        # Provide download link for the report
        st.download_button(
            label="Download Report",
            data=report_pdf,
            file_name="evaluation_report.pdf",
            mime="application/pdf"
        )


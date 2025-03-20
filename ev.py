import os
import requests
import streamlit as st
import PyPDF2
from fpdf import FPDF
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the Hugging Face API key from the environment variable
HF_API_KEY = os.getenv("HF_API_KEY")

# Function to extract text from PDF using PyPDF2
def extract_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle None values
        return text.strip()  # Strip any unnecessary leading/trailing whitespace
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


# Function to call Hugging Face API for question-answer evaluation
def get_huggingface_response(question, answer):
    try:
        # Hugging Face API URL (using a pre-trained model for question-answering)
        api_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
        
        # Access the Hugging Face API key from environment variables
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}  # Use the loaded environment variable

        # Construct the payload
        payload = {
            "inputs": {
                "question": question,
                "context": answer,
            }
        }

        # Make request to Hugging Face API
        response = requests.post(api_url, headers=headers, json=payload)
        result = response.json()

        if response.status_code == 200:
            # Extract and return the result (answer found by the model)
            return result.get('answer', "No answer found")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return "Error: Unable to process the request."

    except Exception as e:
        st.error(f"Error in Hugging Face API call: {e}")
        return "Error during evaluation"


# Function to generate a report PDF
def generate_report(evaluation_results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Question-Answer Evaluation Report", ln=True, align='C')

    for question, answer, result in evaluation_results:
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Question: {question}", ln=True)
        
        # Use multi_cell to handle long text (answers), but make sure the text is not too long for a single line
        pdf.multi_cell(0, 10, txt=f"Answer: {answer}", align='L')
        
        pdf.ln(5)  # Add some space between the answer and the result
        pdf.cell(200, 10, txt=f"Result: {result}", ln=True)

    return pdf.output(dest='S')


# Streamlit app layout
st.title("Question-Answer Evaluator")

# File upload for question and answer PDFs
question_pdf = st.file_uploader("Upload Question PDF", type="pdf")
answer_pdfs = st.file_uploader("Upload Answer PDFs", type="pdf", accept_multiple_files=True)

if question_pdf and answer_pdfs:
    # Extract text from question PDF
    st.write("Extracting text from Question PDF...")
    question_text = extract_pdf_text(question_pdf)

    # Extract text from answer PDFs
    answer_texts = []
    for file in answer_pdfs:
        st.write(f"Extracting text from Answer PDF {file.name}...")
        answer_texts.append(extract_pdf_text(file))

    # Display extracted text (for debugging purposes)
    st.subheader("Extracted Questions")
    st.text_area("Extracted Question Text", question_text, height=150)

    st.subheader("Extracted Answers")
    for i, answer_text in enumerate(answer_texts):
        st.text_area(f"Extracted Answer {i+1}", answer_text, height=150)

    # Show a progress bar for evaluation
    progress_bar = st.progress(0)

    evaluation_results = []
    for i, answer_text in enumerate(answer_texts):
        st.write(f"Evaluating Answer {i+1}...") 
        progress_bar.progress((i + 1) / len(answer_texts))

        # Call Hugging Face API to evaluate the answer based on the question
        evaluation_result = get_huggingface_response(question_text, answer_text)
        
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

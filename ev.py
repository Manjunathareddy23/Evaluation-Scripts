import requests
import streamlit as st
import PyPDF2
from fpdf import FPDF

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

# Function to call Hugging Face API for question-answer evaluation
def get_huggingface_response(question, answer):
    try:
        # Hugging Face API URL (using a pre-trained model for question-answering)
        api_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
        
        # Access the Hugging Face API key from secrets
        headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}  # Use your API key stored in secrets

        # Construct the payload for the Hugging Face model
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
            return result['answer']
        else:
            return "Error: Unable to process the request."

    except Exception as e:
        st.error(f"Error in Hugging Face API call: {e}")
        return "Error during evaluation"

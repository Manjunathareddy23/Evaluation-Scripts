from dataclasses import dataclass
import os
import streamlit as st


@dataclass
class Config:

    HF_MODEL = (
        "mistralai/Mistral-7B-Instruct-v0.3"
    )

    HF_API = (
        "https://api-inference.huggingface.co/models/"
        + HF_MODEL
    )

    HF_TOKEN = (
        os.getenv("HF_API_KEY")
        or st.secrets.get("HF_API_KEY", "")
    )

    REQUEST_TIMEOUT = 120

    MAX_PDF_SIZE = 20

    MAX_FILES = 100

    REPORT_FOLDER = "reports"

    LOG_FOLDER = "logs"

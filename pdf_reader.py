"""
pdf_reader.py

Production-ready PDF text extraction.
Supports both text-based PDFs and scanned PDFs.
"""

from __future__ import annotations

import fitz
import streamlit as st


class PDFReader:

    def __init__(self):
        pass

    @st.cache_data(show_spinner=False)
    def extract_text(_self, uploaded_file) -> str:
        """
        Extract text from a PDF.

        Parameters
        ----------
        uploaded_file : UploadedFile

        Returns
        -------
        str
        """

        try:

            pdf_bytes = uploaded_file.read()

            document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )

            pages = []

            for page in document:

                pages.append(
                    page.get_text("text")
                )

            document.close()

            return "\n".join(pages).strip()

        except Exception as e:

            raise RuntimeError(
                f"Unable to read PDF : {e}"
            )

    @st.cache_data(show_spinner=False)
    def page_count(_self, uploaded_file):

        pdf_bytes = uploaded_file.read()

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        total = len(doc)

        doc.close()

        return total

    @st.cache_data(show_spinner=False)
    def has_text(_self, uploaded_file):

        text = _self.extract_text(uploaded_file)

        return len(text.strip()) > 20

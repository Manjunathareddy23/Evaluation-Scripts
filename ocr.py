"""
ocr.py

OCR Engine
"""

from __future__ import annotations

import fitz
import numpy as np
import easyocr
from PIL import Image
import streamlit as st


class OCREngine:

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    @st.cache_data(show_spinner=False)
    def extract_text(_self, uploaded_pdf):

        pdf_bytes = uploaded_pdf.read()

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        final_text = []

        for page in document:

            pix = page.get_pixmap(
                dpi=300
            )

            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            image = np.array(image)

            result = _self.reader.readtext(
                image,
                detail=0,
                paragraph=True
            )

            final_text.append(
                "\n".join(result)
            )

        document.close()

        return "\n".join(final_text)

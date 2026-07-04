"""
evaluator.py

Production-ready Answer Evaluation Engine
"""

from __future__ import annotations

import json
import requests

from config import Config


class AnswerEvaluator:

    def __init__(self):

        self.url = Config.HF_API

        self.headers = {
            "Authorization": f"Bearer {Config.HF_TOKEN}"
        }

    def build_prompt(
        self,
        question: str,
        answer: str
    ):

        return f"""
You are an experienced university professor.

Evaluate the student's answer.

QUESTION:
{question}

STUDENT ANSWER:
{answer}

Return ONLY valid JSON.

Format:

{{
    "marks": 0,
    "percentage": 0,
    "grade": "",
    "strengths": [],
    "weaknesses": [],
    "missing_points": [],
    "feedback": "",
    "improvement": ""
}}
"""

    def evaluate(
        self,
        question,
        answer
    ):

        prompt = self.build_prompt(
            question,
            answer
        )

        payload = {

            "inputs": prompt,

            "parameters": {

                "max_new_tokens": 600,

                "temperature": 0.2,

                "return_full_text": False

            }

        }

        response = requests.post(

            self.url,

            headers=self.headers,

            json=payload,

            timeout=Config.REQUEST_TIMEOUT

        )

        response.raise_for_status()

        result = response.json()

        text = result[0]["generated_text"]

        return json.loads(text)

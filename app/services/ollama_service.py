from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import httpx

from app.config import settings


SYSTEM_PROMPT = """
You are an enterprise email intelligence engine.

Analyze emails intelligently.

Possible categories:
- spam
- replyable
- ignorable

Rules:

1. spam
   - scams
   - phishing
   - lottery
   - fake offers
   - promotions

2. replyable
   - questions
   - requests
   - meetings
   - support requests
   - action-required emails

3. ignorable
   - newsletters
   - notifications
   - FYI updates
   - automated emails

Rules:
- Return ONLY valid JSON
- No markdown
- No explanations
- Summary under 20 words
- Reply under 40 words
"""


class OllamaClient:

    def __init__(self):

        self.model = settings.ollama_model
        self.url = settings.ollama_url
        self.timeout = settings.ollama_timeout_sec
        self.retries = settings.ollama_retries

    # =====================================================
    # INTERNAL HTTP CALL
    # =====================================================

    async def _post(self, payload: Dict[str, Any]):

        last_exception = None

        for _ in range(self.retries + 1):

            try:

                async with httpx.AsyncClient(
                    timeout=self.timeout
                ) as client:

                    response = await client.post(
                        self.url,
                        json=payload
                    )

                    response.raise_for_status()

                    return response.json()

            except Exception as e:

                last_exception = e

                await asyncio.sleep(1)

        raise last_exception

    # =====================================================
    # GENERATE
    # =====================================================

    async def _generate(self, prompt: str):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 300,
            },
        }

        data = await self._post(payload)

        return data.get("response", "")

    # =====================================================
    # CLEAN JSON
    # =====================================================

    def _clean_json(self, text: str):

        text = text.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")

        start = text.find("{")
        end = text.rfind("}") + 1

        if start >= 0 and end > start:
            text = text[start:end]

        try:
            return json.loads(text)

        except Exception:

            return {
                "category": "ignorable",
                "summary": "Unable to analyze email."
            }

    # =====================================================
    # MAIN EMAIL ANALYSIS
    # =====================================================

    async def analyze_email_complete(
        self,
        subject: str,
        body: str,
    ):

        prompt = f"""
{SYSTEM_PROMPT}

Analyze this email.

Return ONLY valid JSON.

Formats:

SPAM:
{{
  "category": "spam"
}}

IGNORABLE:
{{
  "category": "ignorable",
  "summary": "Short summary"
}}

REPLYABLE:
{{
  "category": "replyable",
  "summary": "Short summary",
  "reply_draft": "Professional reply"
}}

Email Subject:
{subject}

Email Body:
{body}
"""

        response = await self._generate(prompt)

        return self._clean_json(response)

    # =====================================================
    # TPO SUMMARY
    # =====================================================

    async def generate_tpo_summary(
        self,
        subject: str,
        body: str,
    ):

        prompt = f"""
You are a placement and internship email summarizer.

Generate a detailed student-friendly summary.

Focus on:
- company name
- role
- eligibility
- CGPA criteria
- deadline
- interview date
- salary/package
- important instructions

Rules:
- Return plain text only
- No JSON
- No markdown
- Keep under 120 words
- Make it easy to understand

Email Subject:
{subject}

Email Body:
{body}
"""

        response = await self._generate(prompt)

        return response.strip()
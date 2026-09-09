"""Gemini (Vertex AI) client for assigning change categories.

Wraps the google-genai client configured for Vertex AI and requests structured
output: a JSON array whose values are drawn from the allowed category names.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from google import genai
from google.genai import types

from .prompt import ALLOWED_CATEGORIES, CONFIDENCE_LEVELS, ImageRef


@dataclass
class CategorizeResult:
    """The model's category assignment plus token usage for one request."""

    categories: list[str]
    confidence: str
    summary: str
    raw_text: str
    prompt_tokens: int
    candidates_tokens: int
    total_tokens: int


class GeminiCategorizer:
    """Calls Gemini via Vertex AI to assign land-cover-change categories."""

    def __init__(
        self,
        project: str = "earthsystem-dev-c3po",
        location: str = "global",
        model: str = "gemini-2.5-pro",
    ) -> None:
        """Create the categorizer.

        Args:
            project: the Google Cloud project for Vertex AI.
            location: the Vertex AI location (e.g. "global").
            model: the Gemini model name.
        """
        # The SDK otherwise falls back to the gcloud default project for quota,
        # which may not have the API enabled.
        os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", project)
        self.model = model
        self._client = genai.Client(
            vertexai=True, project=project, location=location
        )
        self._schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "summary": types.Schema(type=types.Type.STRING),
                "categories": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.STRING,
                        enum=ALLOWED_CATEGORIES,
                    ),
                ),
                "confidence": types.Schema(
                    type=types.Type.STRING,
                    enum=CONFIDENCE_LEVELS,
                ),
            },
            required=["summary", "categories", "confidence"],
            property_ordering=["summary", "categories", "confidence"],
        )

    def categorize(self, prompt: str, images: list[ImageRef]) -> CategorizeResult:
        """Run the model on a prompt and ordered images.

        Args:
            prompt: the text prompt (references the images by their labels).
            images: the labeled images, in the same order referenced by prompt.

        Returns:
            the parsed categories and token usage.
        """
        parts: list[types.Part] = [types.Part.from_text(text=prompt)]
        for img in images:
            parts.append(types.Part.from_text(text=f"{img.label}:"))
            parts.append(
                types.Part.from_bytes(data=img.png_bytes, mime_type="image/png")
            )

        response = self._client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=self._schema,
            ),
        )

        raw_text = response.text or "{}"
        categories, confidence, summary = _parse_result(raw_text)

        usage = response.usage_metadata
        return CategorizeResult(
            categories=categories,
            confidence=confidence,
            summary=summary,
            raw_text=raw_text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            candidates_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
        )


def _parse_result(raw_text: str) -> tuple[list[str], str, str]:
    """Parse the JSON object into (categories, confidence, summary)."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return [], "", ""
    if not isinstance(parsed, dict):
        return [], "", ""

    categories = _parse_categories(parsed.get("categories"))

    confidence = parsed.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        confidence = ""

    summary = parsed.get("summary")
    if not isinstance(summary, str):
        summary = ""

    return categories, confidence, summary


def _parse_categories(value: object) -> list[str]:
    """Keep only valid, known, de-duplicated category names from a list."""
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item in ALLOWED_CATEGORIES and item not in seen:
            seen.add(item)
            result.append(item)
    return result

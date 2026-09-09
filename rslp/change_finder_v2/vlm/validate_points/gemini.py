"""Gemini (Vertex AI) client for validating whether a point has a real change.

Wraps the google-genai client configured for Vertex AI and requests structured output:
a binary positive/negative prediction with reasoning and a confidence level. Copied and
adapted from the categorization client so this package is self-contained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from google import genai
from google.genai import types

from .prompt import ALLOWED_PREDICTIONS, CONFIDENCE_LEVELS, ImageRef


@dataclass
class ValidateResult:
    """The model's validation result plus token usage for one request."""

    prediction: str | None
    confidence: str | None
    reasoning: str
    raw_text: str
    prompt_tokens: int
    candidates_tokens: int
    total_tokens: int


class GeminiValidator:
    """Calls Gemini via Vertex AI to validate whether a point has a real change."""

    def __init__(
        self,
        project: str = "earthsystem-dev-c3po",
        location: str = "global",
        model: str = "gemini-2.5-pro",
    ) -> None:
        """Create the validator.

        Args:
            project: the Google Cloud project for Vertex AI.
            location: the Vertex AI location (e.g. "global").
            model: the Gemini model name.
        """
        import os

        # The SDK otherwise falls back to the gcloud default project for quota,
        # which may not have the API enabled.
        os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", project)
        self.model = model
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "reasoning": types.Schema(type=types.Type.STRING),
                "prediction": types.Schema(
                    type=types.Type.STRING,
                    enum=ALLOWED_PREDICTIONS,
                ),
                "confidence": types.Schema(
                    type=types.Type.STRING,
                    enum=CONFIDENCE_LEVELS,
                ),
            },
            required=["reasoning", "prediction", "confidence"],
            property_ordering=["reasoning", "prediction", "confidence"],
        )

    def validate(self, prompt: str, images: list[ImageRef]) -> ValidateResult:
        """Run the model on a prompt and ordered images.

        Args:
            prompt: the text prompt (references the images by their captions).
            images: the labeled images, in the same order referenced by the prompt.

        Returns:
            the parsed prediction and token usage.
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
        prediction, confidence, reasoning = _parse_result(raw_text)

        usage = response.usage_metadata
        return ValidateResult(
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            raw_text=raw_text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            candidates_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
        )


def _parse_result(raw_text: str) -> tuple[str | None, str | None, str]:
    """Parse the JSON object into (prediction, confidence, reasoning)."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, None, ""
    if not isinstance(parsed, dict):
        return None, None, ""

    prediction = parsed.get("prediction")
    if prediction not in ALLOWED_PREDICTIONS:
        prediction = None

    confidence = parsed.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        confidence = None

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""

    return prediction, confidence, reasoning

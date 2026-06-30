"""Gemini (Vertex AI) client for assigning fine-grained change categories.

Wraps the google-genai client configured for Vertex AI and requests structured output:
one pre-class category and/or one post-class category, OR a single same-class category,
plus reasoning and a confidence level. Each category field is nullable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from google import genai
from google.genai import types

from .prompt import (
    ALLOWED_POST,
    ALLOWED_PRE,
    ALLOWED_SAME,
    CONFIDENCE_LEVELS,
    ImageRef,
)


@dataclass
class CategorizeResult:
    """The model's category assignment plus token usage for one request."""

    pre_change_category: str | None
    post_change_category: str | None
    same_change_category: str | None
    flagged_for_review: bool
    confidence: str | None
    reasoning: str
    raw_text: str
    prompt_tokens: int
    candidates_tokens: int
    total_tokens: int


class GeminiCategorizer:
    """Calls Gemini via Vertex AI to assign fine-grained change categories."""

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
                "pre_change_category": types.Schema(
                    type=types.Type.STRING,
                    enum=ALLOWED_PRE,
                    nullable=True,
                ),
                "post_change_category": types.Schema(
                    type=types.Type.STRING,
                    enum=ALLOWED_POST,
                    nullable=True,
                ),
                "same_change_category": types.Schema(
                    type=types.Type.STRING,
                    enum=ALLOWED_SAME,
                    nullable=True,
                ),
                "flagged_for_review": types.Schema(type=types.Type.BOOLEAN),
                "confidence": types.Schema(
                    type=types.Type.STRING,
                    enum=CONFIDENCE_LEVELS,
                ),
            },
            required=[
                "reasoning",
                "pre_change_category",
                "post_change_category",
                "same_change_category",
                "flagged_for_review",
                "confidence",
            ],
            property_ordering=[
                "reasoning",
                "pre_change_category",
                "post_change_category",
                "same_change_category",
                "flagged_for_review",
                "confidence",
            ],
        )

    def categorize(self, prompt: str, images: list[ImageRef]) -> CategorizeResult:
        """Run the model on a prompt and ordered images.

        Args:
            prompt: the text prompt (references the images by their captions).
            images: the labeled images, in the same order referenced by the prompt.

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
        pre, post, same, flagged, confidence, reasoning = _parse_result(raw_text)

        usage = response.usage_metadata
        return CategorizeResult(
            pre_change_category=pre,
            post_change_category=post,
            same_change_category=same,
            flagged_for_review=flagged,
            confidence=confidence,
            reasoning=reasoning,
            raw_text=raw_text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            candidates_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
        )


def _parse_result(
    raw_text: str,
) -> tuple[str | None, str | None, str | None, bool, str | None, str]:
    """Parse the JSON object into (pre, post, same, flagged, confidence, reasoning)."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None, None, None, False, None, ""
    if not isinstance(parsed, dict):
        return None, None, None, False, None, ""

    pre = _validate(parsed.get("pre_change_category"), ALLOWED_PRE)
    post = _validate(parsed.get("post_change_category"), ALLOWED_POST)
    same = _validate(parsed.get("same_change_category"), ALLOWED_SAME)

    # A same-class category is mutually exclusive with pre/post categories.
    if same is not None and (pre is not None or post is not None):
        same = None

    flagged = bool(parsed.get("flagged_for_review"))

    confidence = parsed.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        confidence = None

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""

    return pre, post, same, flagged, confidence, reasoning


def _validate(value: object, allowed: list[str]) -> str | None:
    """Return the value if it is an allowed category name, else None."""
    if isinstance(value, str) and value in allowed:
        return value
    return None

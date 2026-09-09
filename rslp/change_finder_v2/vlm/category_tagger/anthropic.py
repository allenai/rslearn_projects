"""Anthropic (Claude) client for assigning fine-grained change categories.

Mirrors :class:`~rslp.change_finder_v2.vlm.category_tagger.gemini.GeminiCategorizer`,
exposing the same ``categorize(prompt, images) -> CategorizeResult`` interface so the
orchestrator can swap one for the other.

Anthropic has no native response-schema, so structured output is forced via a single
tool whose ``input_schema`` mirrors the Gemini schema; ``tool_choice`` requires the
model to call it. The returned tool input is validated by the same logic as the Gemini
client (reused via :func:`gemini._parse_result`).
"""

from __future__ import annotations

import base64
import json

import anthropic

from .gemini import CategorizeResult, _parse_result
from .prompt import (
    ALLOWED_POST,
    ALLOWED_PRE,
    ALLOWED_SAME,
    CONFIDENCE_LEVELS,
    ImageRef,
)

# Name of the forced tool used to extract structured output.
_TOOL_NAME = "assign_categories"


class AnthropicCategorizer:
    """Calls Claude via the Anthropic API to assign fine-grained change categories."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1024,
    ) -> None:
        """Create the categorizer.

        Args:
            api_key: the Anthropic API key. If None, the client reads it from the
                ``ANTHROPIC_API_KEY`` environment variable.
            model: the Claude model name.
            max_tokens: max output tokens per request.
        """
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)
        self._tool = {
            "name": _TOOL_NAME,
            "description": (
                "Record the fine-grained land-cover-change categorization for the "
                "marked point."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "pre_change_category": {
                        "type": ["string", "null"],
                        "enum": [*ALLOWED_PRE, None],
                    },
                    "post_change_category": {
                        "type": ["string", "null"],
                        "enum": [*ALLOWED_POST, None],
                    },
                    "same_change_category": {
                        "type": ["string", "null"],
                        "enum": [*ALLOWED_SAME, None],
                    },
                    "flagged_for_review": {"type": "boolean"},
                    "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
                },
                "required": [
                    "reasoning",
                    "pre_change_category",
                    "post_change_category",
                    "same_change_category",
                    "flagged_for_review",
                    "confidence",
                ],
            },
        }

    def categorize(self, prompt: str, images: list[ImageRef]) -> CategorizeResult:
        """Run the model on a prompt and ordered images.

        Args:
            prompt: the text prompt (references the images by their captions).
            images: the labeled images, in the same order referenced by the prompt.

        Returns:
            the parsed categories and token usage.
        """
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "text", "text": f"{img.label}:"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(img.png_bytes).decode("ascii"),
                    },
                }
            )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=[self._tool],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": content}],
        )

        tool_input = _extract_tool_input(response)
        raw_text = json.dumps(tool_input)
        pre, post, same, flagged, confidence, reasoning = _parse_result(raw_text)

        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return CategorizeResult(
            pre_change_category=pre,
            post_change_category=post,
            same_change_category=same,
            flagged_for_review=flagged,
            confidence=confidence,
            reasoning=reasoning,
            raw_text=raw_text,
            prompt_tokens=input_tokens,
            candidates_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )


def _extract_tool_input(response: anthropic.types.Message) -> dict:
    """Return the input dict of the forced tool-use block, or {} if absent."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
            return dict(block.input)
    return {}

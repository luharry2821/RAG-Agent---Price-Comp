"""Claude wrapper for the two generative steps of the pipeline.

The agent works fully offline — retrieval, matching and price comparison are
deterministic Python. Claude is used for the parts where language matters:

1. turning a messy question into a :class:`QuerySpec`;
2. writing the final grounded answer over the retrieved listings.

If no credentials are configured (or the SDK isn't installed) the agent falls
back to a heuristic parser and a template writer, so nothing hard-fails.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_MODEL = os.environ.get("SNEAKERRAG_MODEL", "claude-opus-5")
DEFAULT_EFFORT = os.environ.get("SNEAKERRAG_EFFORT", "low")


class LLMUnavailable(RuntimeError):
    pass


class NullLLM:
    """Stand-in used when Claude isn't configured."""

    name = "none"
    available = False

    def complete(self, system: str, user: str, **kw: Any) -> str:
        raise LLMUnavailable("no Claude credentials configured")


class ClaudeLLM:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install
            raise LLMUnavailable("pip install anthropic to enable Claude answers") from exc

        self._anthropic = anthropic
        # Zero-arg client: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or
        # an `ant auth login` profile, in that order.
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.name = model
        self.available = True

    def complete(self, system: str, user: str, *, max_tokens: int = 4000,
                 effort: str | None = None) -> str:
        """One grounded completion. Streams so long contexts can't time out."""
        anthropic = self._anthropic
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                # Sampling params (temperature/top_p) are rejected by this model
                # family; response shape is controlled by the system prompt.
                output_config={"effort": effort or self.effort},
                messages=[{"role": "user", "content": user}],
            ) as stream:
                message = stream.get_final_message()
        except anthropic.NotFoundError as exc:
            raise LLMUnavailable(f"model {self.model} unavailable: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailable(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailable(f"Claude API error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"could not reach the Claude API: {exc}") from exc

        if getattr(message, "stop_reason", None) == "refusal":
            raise LLMUnavailable("request was declined by the model")
        return "".join(b.text for b in message.content if getattr(b, "type", "") == "text").strip()


def get_llm(model: str = "", *, enabled: bool = True) -> Any:
    """Return a Claude client, or :class:`NullLLM` when unavailable."""
    if not enabled or os.environ.get("SNEAKERRAG_NO_LLM"):
        return NullLLM()
    try:
        return ClaudeLLM(model or DEFAULT_MODEL)
    except LLMUnavailable:
        return NullLLM()
    except Exception:
        return NullLLM()


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_json_response(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction from a model response."""
    if not text:
        return {}
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(body)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}

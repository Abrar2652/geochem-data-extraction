"""
llm_clients.py - Unified interface to multiple LLM providers.

Supports:
  - Anthropic Claude  (claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)
  - OpenAI GPT        (gpt-5.2, gpt-5-mini, gpt-4o, gpt-4-turbo)
  - Google Gemini     (gemini-3-flash-preview, gemini-2.5-flash, gemini-2.0-flash)

All clients implement the same interface:
  client.complete(system, user, max_tokens) -> str (raw LLM text response)
  client.complete_json(system, user, max_tokens) -> dict | list
"""

from __future__ import annotations
import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """Abstract base for all LLM clients."""

    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        images: list[dict] | None = None,
    ) -> str:
        """Return the raw text response from the LLM.

        Args:
            system: System prompt.
            user: User message text.
            max_tokens: Maximum response tokens.
            images: Optional list of image dicts for vision input.
                Each dict: {"image_bytes": bytes, "media_type": str}.
                When None, behaves as a text-only call.
        """

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 8192,
        images: list[dict] | None = None,
    ) -> dict | list:
        """Complete and parse the response as JSON.

        Retries up to max_retries times on JSON parse failures.
        Strips markdown code fences before parsing.
        """
        for attempt in range(self.max_retries):
            raw = self.complete(system, user, max_tokens, images=images)
            try:
                return _parse_json(raw)
            except json.JSONDecodeError as exc:
                if attempt == self.max_retries - 1:
                    raise ValueError(
                        f"LLM ({self.model}) returned invalid JSON after "
                        f"{self.max_retries} attempts.\nLast response:\n{raw[:500]}"
                    ) from exc
                time.sleep(1)  # brief pause before retry
        return {}

    @property
    def provider(self) -> str:
        return self.__class__.__name__.replace("Client", "").lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic Claude
# ──────────────────────────────────────────────────────────────────────────────

class ClaudeClient(LLMClient):
    """Anthropic Claude client using the messages API."""

    AVAILABLE_MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        self._client = _anthropic.Anthropic(api_key=api_key) if api_key else _anthropic.Anthropic()

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        images: list[dict] | None = None,
    ) -> str:
        # Build user message content
        if images:
            content_parts: list[dict] = []
            for img in images:
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": base64.b64encode(img["image_bytes"]).decode("ascii"),
                    },
                })
            content_parts.append({"type": "text", "text": user})
            messages = [{"role": "user", "content": content_parts}]
        else:
            messages = [{"role": "user", "content": user}]

        for attempt in range(self.max_retries):
            try:
                # Use streaming for large max_tokens to avoid SDK timeout
                if max_tokens > 16384:
                    with self._client.messages.stream(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=messages,
                        temperature=self.temperature,
                    ) as stream:
                        return stream.get_final_text()
                else:
                    response = self._client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=messages,
                        temperature=self.temperature,
                    )
                    return response.content[0].text
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
        return ""

    def complete_with_tool(
        self,
        system: str,
        user: str,
        tool_schema: dict,
        max_tokens: int = 4096,
    ) -> dict:
        """Use Anthropic tool use for guaranteed JSON schema output."""
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    tools=[tool_schema],
                    tool_choice={"type": "any"},
                    temperature=self.temperature,
                )
                for block in response.content:
                    if block.type == "tool_use":
                        return dict(block.input)
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI GPT
# ──────────────────────────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """OpenAI client using the chat completions API."""

    AVAILABLE_MODELS = [
        "gpt-5.2",#"gpt-4o",
        "gpt-5-mini",
        "gpt-4-turbo",
        "o1",
        "o3-mini",
    ]

    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self._client = _openai.OpenAI(api_key=api_key) if api_key else _openai.OpenAI()

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        images: list[dict] | None = None,
    ) -> str:
        # Build user message content
        if images:
            content_parts: list[dict] = []
            for img in images:
                b64 = base64.b64encode(img["image_bytes"]).decode("ascii")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['media_type']};base64,{b64}",
                        "detail": "high",
                    },
                })
            content_parts.append({"type": "text", "text": user})
            user_message = {"role": "user", "content": content_parts}
        else:
            user_message = {"role": "user", "content": user}

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    # max_output_tokens=max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        user_message,
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def complete_json_structured(
        self,
        system: str,
        user: str,
        response_format: Any,
        max_tokens: int = 4096,
    ) -> dict:
        """Use OpenAI structured outputs (beta) with a Pydantic model."""
        for attempt in range(self.max_retries):
            try:
                response = self._client.beta.chat.completions.parse(
                    model=self.model,
                    max_output_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format=response_format,
                )
                parsed = response.choices[0].message.parsed
                return parsed.model_dump() if hasattr(parsed, "model_dump") else {}
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Google Gemini
# ──────────────────────────────────────────────────────────────────────────────

class GeminiClient(LLMClient):
    """Google Gemini client using the google-genai SDK."""

    AVAILABLE_MODELS = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    def __init__(self, model: str = "gemini-3-flash-preview", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        try:
            import google.genai as genai
            import google.genai.types as gentypes
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

        self._genai = genai
        self._types = gentypes
        if api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = genai.Client()  # uses GOOGLE_API_KEY env var

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        images: list[dict] | None = None,
    ) -> str:
        # Build contents with optional images
        if images:
            parts = []
            for img in images:
                parts.append(self._types.Part.from_bytes(
                    data=img["image_bytes"],
                    mime_type=img["media_type"],
                ))
            parts.append(user)
            contents = parts
        else:
            contents = user

        for attempt in range(self.max_retries):
            try:
                config = self._types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.temperature,
                    # max_output_tokens=max_tokens,
                )
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def complete_json_native(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict | list:
        """Use Gemini's native JSON mode (response_mime_type)."""
        for attempt in range(self.max_retries):
            try:
                config = self._types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.temperature,
                    # max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                )
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=config,
                )
                raw = response.text or ""
                return _parse_json(raw)
            except json.JSONDecodeError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Local HuggingFace (open-weight 7-32B models, no vision)
# ──────────────────────────────────────────────────────────────────────────────

class LocalHFClient(LLMClient):
    """Local HuggingFace model client (text-only, no vision).

    Mirrors `iswc2026/scripts/pipeline_adapter.py::LocalHFClient` so the
    same short-name aliases work in geochem_benchmark.run() without changes
    on the caller side. When ``images`` are passed (vision LLM fallback),
    we log a warning and return an empty string so the pipeline records the
    vision step as failed and falls through to text-only paths.
    """

    LOCAL_MODEL_MAP = {
        "qwen25-7b":      "Qwen/Qwen2.5-7B-Instruct",
        "qwen25-32b":     "Qwen/Qwen2.5-32B-Instruct",
        "mistral-7b-v02": "mistralai/Mistral-7B-Instruct-v0.2",
        "mistral-7b-v03": "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral-7b":     "mistralai/Mistral-7B-Instruct-v0.3",
        "llama31-8b":     "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3-8b":      "meta-llama/Meta-Llama-3.1-8B-Instruct",
    }

    _instances: dict = {}  # repo -> (tokenizer, model)

    AVAILABLE_MODELS = list(LOCAL_MODEL_MAP.keys())

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        repo = self.LOCAL_MODEL_MAP.get(model, model)
        if repo in LocalHFClient._instances:
            self._tokenizer, self._model = LocalHFClient._instances[repo]
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            logger.info(f"LocalHFClient: loading {repo} (one-time)")
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True,
            )
            LocalHFClient._instances[repo] = (tok, mdl)
            self._tokenizer, self._model = tok, mdl

    def complete(self, system: str, user: str, max_tokens: int = 4096,
                 images: list[dict] | None = None) -> str:
        if images:
            logger.warning(
                "LocalHFClient(%s) ignoring %d images (no vision capability)",
                self.model, len(images),
            )
            return ""  # signal vision unsupported; pipeline can fall through
        import torch
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        text = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=8192,
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=min(max_tokens, 4096),
                do_sample=False, temperature=None, top_p=None,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_PROVIDER_MAP: dict[str, type[LLMClient]] = {
    "claude":  ClaudeClient,
    "openai":  OpenAIClient,
    "gemini":  GeminiClient,
    "gpt":     OpenAIClient,
    "local":   LocalHFClient,
    "hf":      LocalHFClient,
}

def create_client(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> LLMClient:
    """Factory function to create an LLM client by provider name.

    Args:
        provider: One of 'claude', 'openai', 'gemini'.
        model: Model identifier. If None, uses each provider's default.
        api_key: API key. Falls back to environment variables if None.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        Configured LLMClient instance.
    """
    provider_lower = provider.lower().strip()
    client_cls = _PROVIDER_MAP.get(provider_lower)
    if client_cls is None:
        available = ", ".join(sorted(_PROVIDER_MAP.keys()))
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

    kwargs: dict = {"temperature": temperature}
    if api_key:
        kwargs["api_key"] = api_key

    # Default models per provider
    defaults = {
        "claude": "claude-sonnet-4-6",
        "openai": "gpt-5.2",
        "gpt":    "gpt-5.2",
        "gemini": "gemini-3-flash-preview",
    }
    resolved_model = model or defaults.get(provider_lower, "")

    return client_cls(model=resolved_model, **kwargs)


def list_available_models() -> dict[str, list[str]]:
    """Return all known models grouped by provider."""
    return {
        "claude": ClaudeClient.AVAILABLE_MODELS,
        "openai": OpenAIClient.AVAILABLE_MODELS,
        "gemini": GeminiClient.AVAILABLE_MODELS,
    }


# ──────────────────────────────────────────────────────────────────────────────
# JSON parsing helper
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | list:
    """Parse JSON from LLM response, stripping markdown code fences if present.

    Handles truncated JSON from output token limits by recovering valid
    samples from the truncation point.
    """
    text = text.strip()

    # Strip ``` fences
    fenced = re.match(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Some models emit leading/trailing prose — try to find the JSON block
    json_start = text.find("{")
    json_array_start = text.find("[")

    if json_start == -1 and json_array_start == -1:
        raise json.JSONDecodeError("No JSON object or array found", text, 0)

    if json_array_start != -1 and (json_start == -1 or json_array_start < json_start):
        # Looks like an array
        text = text[json_array_start:]
        end = text.rfind("]") + 1
        text = text[:end] if end > 0 else text
    else:
        text = text[json_start:]
        end = text.rfind("}") + 1
        text = text[:end] if end > 0 else text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to recover from truncated JSON (output token limit hit).
        # Find the last complete JSON object in a "samples" array.
        return _recover_truncated_json(text)


def _recover_truncated_json(text: str) -> dict:
    """Recover samples from truncated JSON output.

    When LLM output hits the token limit, the JSON is cut mid-way.
    This finds the last complete object in the samples array and
    returns a valid dict with those samples.
    """
    # Find the "samples" array start
    samples_match = re.search(r'"samples"\s*:\s*\[', text)
    if not samples_match:
        raise json.JSONDecodeError("No 'samples' array found in truncated JSON", text, 0)

    array_start = samples_match.end() - 1  # position of '['
    # Find the last complete object by looking for '},\n    {'
    # or '}\n  ]' pattern
    last_complete = text.rfind("},")
    if last_complete == -1:
        last_complete = text.rfind("}")

    if last_complete <= array_start:
        raise json.JSONDecodeError("No complete objects in truncated JSON", text, 0)

    # Build a valid JSON: take everything up to last complete object + close
    recovered = text[:last_complete + 1] + "]}"
    try:
        result = json.loads(recovered)
        n = len(result.get("samples", []))
        logger.info("Recovered %d samples from truncated JSON", n)
        return result
    except json.JSONDecodeError:
        # Last resort: extract individual objects with regex
        pattern = re.compile(r'\{[^{}]*"sample_name"[^{}]*\}')
        objects = []
        for m in pattern.finditer(text):
            try:
                obj = json.loads(m.group())
                objects.append(obj)
            except json.JSONDecodeError:
                continue
        if objects:
            logger.info("Regex-recovered %d samples from truncated JSON", len(objects))
            return {"samples": objects, "extraction_notes": "recovered from truncated output"}
        raise json.JSONDecodeError("Could not recover any samples", text, 0)

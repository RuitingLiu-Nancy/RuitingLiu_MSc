"""Shared LLM client — the single entry point for all model calls.

Extracted verbatim (logic-preserving) from resonance_rag/resonance_oracle.py so
the runnable package has ONE copy of call_chat with no sklearn/pandas import
weight. Every approach in this package imports call_chat from here.

Includes the hardening fixes:
- prompt sanitized for transport (emoji/curly-quotes safe under ASCII locale)
- API key validated as ASCII-safe for the Authorization header
- optional stdlib HTTP path (USE_STD_HTTP_CHAT=1) and ASCII retry fallback
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

PROVIDER_BASE_URLS = {
    "openai": None,
    "deepseek": "https://api.deepseek.com/v1",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REQUESTY_BASE_URL = "https://router.requesty.ai/v1"


class OpenRouterHTTPError(RuntimeError):
    """An auditable OpenRouter HTTP failure without request text or secrets."""

    def __init__(self, status_code: int, message: str, retry_after: str | None = None):
        super().__init__(message)
        self.status_code = int(status_code)
        self.retry_after = retry_after


def requesty_model_identity_matches(requested: str, reported: str) -> bool:
    """Accept Requesty's canonical dated Anthropic response name.

    Request routing remains pinned by the provider-qualified requested ID and
    the response-provider header. Requesty may omit that provider prefix and
    append Anthropic's release date in the response body.
    """
    requested_lower = requested.strip().lower()
    reported_lower = reported.strip().lower()
    if not reported_lower or reported_lower == requested_lower:
        return True
    provider, separator, model_name = requested_lower.partition("/")
    if not separator:
        return False
    if reported_lower == model_name:
        return True
    return bool(
        provider == "anthropic"
        and re.fullmatch(re.escape(model_name) + r"-\d{8}", reported_lower)
    )


def call_openrouter_record(
    prompt: str,
    *,
    model: str,
    provider_tag: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float = 180.0,
) -> dict:
    """Call one fixed OpenRouter endpoint and return text plus usage metadata.

    The caller must name one exact provider endpoint. Fallbacks are disabled,
    all requested generation parameters must be supported, and ZDR/no-training
    routing is required. The API key is read from ``OPENROUTER_API_KEY`` and is
    never returned or logged.
    """
    key_name = "OPENROUTER_API_KEY"
    if key_name not in os.environ:
        raise RuntimeError(f"{key_name} is not set")
    api_key = _validate_api_key_for_header(os.environ[key_name], key_name)
    prompt = _sanitize_prompt_for_transport(prompt)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        "provider": {
            "only": [provider_tag],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        },
    }
    req = urlrequest.Request(
        OPENROUTER_BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "X-OpenRouter-Title": "GraphRAG utility-v2 research judging",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlrequest.urlopen(req, timeout=float(timeout_seconds)) as response:
            body = response.read().decode("utf-8")
            response_headers = {
                "x-request-id": response.headers.get("x-request-id"),
                "x-requesty-request-id": response.headers.get("x-requesty-request-id"),
                "x-requesty-provider": response.headers.get("x-requesty-provider"),
                "x-requesty-cache": response.headers.get("x-requesty-cache"),
                "retry-after": response.headers.get("retry-after"),
            }
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", {}).get("message") or body[:500]
        except Exception:
            detail = body[:500]
        raise OpenRouterHTTPError(
            exc.code,
            f"OpenRouter HTTP {exc.code}: {detail}",
            exc.headers.get("retry-after"),
        ) from exc
    elapsed = time.monotonic() - started
    parsed = json.loads(body)
    choices = parsed.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise RuntimeError("OpenRouter response did not contain a chat message")
    text = str(choices[0]["message"].get("content") or "").strip()
    if not text:
        raise RuntimeError("OpenRouter response contained empty text")
    actual_provider = parsed.get("provider")
    if not actual_provider:
        raise RuntimeError("OpenRouter response omitted provider identity")
    if str(actual_provider).lower() != provider_tag.split("/", 1)[0].lower():
        raise RuntimeError(
            f"OpenRouter provider mismatch: requested {provider_tag}, got {actual_provider}"
        )
    reported_model = parsed.get("model")
    if reported_model != model:
        raise RuntimeError(
            f"OpenRouter model mismatch: requested {model}, got {reported_model}"
        )
    usage = parsed.get("usage") or {}
    return {
        "text": text,
        "response_id": parsed.get("id"),
        "reported_model": reported_model,
        "reported_provider": actual_provider,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "cost": usage.get("cost"),
        },
        "latency_seconds": elapsed,
        "response_headers": response_headers,
    }


def call_requesty_record(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float = 180.0,
) -> dict:
    """Call one provider-qualified Requesty model via chat completions.

    ``model`` must include a provider prefix (for example ``anthropic/`` or
    ``deepinfra/``), which prevents policy/fallback routing from silently
    changing the hosted model. The key is read only from ``REQUESTY_API_KEY``.
    """
    key_name = "REQUESTY_API_KEY"
    if key_name not in os.environ:
        raise RuntimeError(f"{key_name} is not set")
    if "/" not in model or model.startswith("policy/"):
        raise ValueError("Requesty study requires one provider-qualified model, not a routing policy")
    api_key = _validate_api_key_for_header(os.environ[key_name], key_name)
    prompt = _sanitize_prompt_for_transport(prompt)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    req = urlrequest.Request(
        REQUESTY_BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "X-Title": "GraphRAG answer-level research evaluation",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlrequest.urlopen(req, timeout=float(timeout_seconds)) as response:
            body = response.read().decode("utf-8")
            response_headers = {
                "x-request-id": response.headers.get("x-request-id"),
                "retry-after": response.headers.get("retry-after"),
            }
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OpenRouterHTTPError(
            exc.code, f"Requesty HTTP {exc.code}: {body[:500]}", exc.headers.get("retry-after")
        ) from exc
    parsed = json.loads(body)
    choices = parsed.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise RuntimeError("Requesty response did not contain a chat message")
    text = str(choices[0]["message"].get("content") or "").strip()
    if not text:
        raise RuntimeError("Requesty response contained empty text")
    reported_model = str(parsed.get("model") or "")
    if not requesty_model_identity_matches(model, reported_model):
        raise RuntimeError(f"Requesty model mismatch: requested {model}, got {reported_model}")
    requested_provider = model.split("/", 1)[0].lower()
    reported_provider = str(response_headers.get("x-requesty-provider") or "").lower()
    if reported_provider and reported_provider != requested_provider:
        raise RuntimeError(
            f"Requesty provider mismatch: requested {requested_provider}, got {reported_provider}"
        )
    usage = parsed.get("usage") or {}
    return {
        "text": text,
        "response_id": parsed.get("id"),
        "reported_model": reported_model or model,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "cost": usage.get("cost"),
        },
        "latency_seconds": time.monotonic() - started,
        "response_headers": response_headers,
    }


class ClaudeCodeGenerationError(RuntimeError):
    """Preserve the CLI response for an auditable, non-fallback failure."""

    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


def claude_code_runtime_contract() -> dict:
    import configuration as config
    settings = dict(config.params("claude_code_answer_runtime"))
    settings["system_prompt_text"] = config.prompt(settings["system_prompt"])
    return settings


def claude_code_auth_status() -> dict:
    """Only expose non-secret account status, never tokens or identifiers."""
    settings = claude_code_runtime_contract()
    result = subprocess.run(
        [settings["executable"], "auth", "status", "--json"],
        capture_output=True, text=True, timeout=int(settings["auth_timeout_seconds"]),
    )
    parsed = json.loads(result.stdout)
    return {key: parsed.get(key) for key in (
        "loggedIn", "authMethod", "apiProvider", "subscriptionType")}


def call_claude_code_record(prompt: str, model_spec: str, max_tokens: int,
                           temperature: float | None = None) -> dict:
    """Official installed CLI, using the user's logged-in subscription.

    Separate provider identity from the Anthropic SDK and Bedrock. No tools,
    skills, plugins, previous conversation, model fallback or API-key routing.
    CLI sampling temperature is not exposed, so it must be recorded as None.
    """
    settings = claude_code_runtime_contract()
    provider, model = split_provider(model_spec)
    if provider != "claude-code" or model != settings["model"] or temperature is not None:
        raise ValueError("Claude Code model/temperature differs from the registered contract")
    if any(os.environ.get(key) for key in settings["forbidden_provider_env"]):
        raise PermissionError("API-key or alternate-provider environment present; subscription-only call refused")
    auth = claude_code_auth_status()
    if not auth["loggedIn"] or auth["authMethod"] != "claude.ai" or auth["apiProvider"] != "firstParty":
        raise PermissionError("Claude Code subscription login is required; no fallback is allowed")
    if auth["subscriptionType"] not in settings["allowed_subscription_types"]:
        raise PermissionError("subscription type is outside the registered configuration")
    version = subprocess.run([settings["executable"], "--version"], capture_output=True,
                             text=True, check=True, timeout=20).stdout.strip()
    if version != settings["expected_cli_version"]:
        raise ValueError(f"Claude Code version changed: {version}")
    env = dict(os.environ)
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
    env["MAX_THINKING_TOKENS"] = "0"
    args = [settings["executable"], "-p", "--model", model,
            "--safe-mode", "--tools", "", "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands",
            "--no-session-persistence", "--output-format", "stream-json", "--verbose",
            "--max-turns", "1", "--permission-mode", "dontAsk",
            "--system-prompt", settings["system_prompt_text"]]
    with tempfile.TemporaryDirectory(prefix="graphrag-cc-answer-") as cwd:
        try:
            result = subprocess.run(args, input=prompt, text=True, capture_output=True,
                                    cwd=cwd, env=env, timeout=int(settings["timeout_seconds"]))
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCodeGenerationError("Claude Code timed out; do not assume zero usage", {
                "status": "TIMEOUT_USAGE_UNKNOWN", "model": model,
                "stdout": (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else exc.stdout,
                "stderr": (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else exc.stderr,
            }) from exc
    events = []
    for line in result.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    init = next((r for r in events if r.get("type") == "system" and r.get("subtype") == "init"), {})
    final = next((r for r in reversed(events) if r.get("type") == "result"), {})
    record = {"cli_version": version, "auth": auth, "requested_model": model,
              "requested_max_output_tokens": max_tokens, "temperature": None,
              "temperature_control": "not_exposed_by_official_CLI",
              "events": events, "stderr": result.stderr, "returncode": result.returncode,
              "result": final, "billing_basis": "Claude_Max_subscription; list_cost_is_not_invoice"}
    if result.returncode or final.get("is_error") or final.get("subtype") != "success":
        raise ClaudeCodeGenerationError("Claude Code failed or reached an account limit; no provider fallback", record)
    if init.get("model") != model or init.get("tools") or init.get("mcp_servers") or init.get("plugins"):
        raise ClaudeCodeGenerationError("unexpected model or enabled tool/context", record)
    if init.get("apiKeySource") != "none" or set(final.get("modelUsage", {})) != {model}:
        raise ClaudeCodeGenerationError("unexpected model usage or API-key source", record)
    limits = [r.get("rate_limit_info", {}) for r in events if r.get("type") == "rate_limit_event"]
    if any(r.get("isUsingOverage") is True for r in limits):
        raise ClaudeCodeGenerationError("Claude Code reports overage usage; stop subscription-only study", record)
    record["rate_limit_reports"] = limits
    if any(r.get("type") == "assistant" and any(c.get("type") == "tool_use"
           for c in r.get("message", {}).get("content", [])) for r in events):
        raise ClaudeCodeGenerationError("unexpected tool invocation", record)
    answer = str(final.get("result", ""))
    if not answer.strip() or final.get("stop_reason") != "end_turn":
        raise ClaudeCodeGenerationError("empty, refused, or truncated answer; preserve and audit", record)
    record["answer"] = answer
    return record


def split_provider(model_spec: str) -> tuple[str, str]:
    if ":" not in model_spec:
        return "openai", model_spec
    provider, model = model_spec.split(":", 1)
    return provider.strip().lower(), model.strip()


def _ascii_normalize(text: str) -> str:
    """Return an ASCII-only version while preserving as much meaning as possible."""
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", " ": " ",
        "•": "*", "→": "->", "←": "<-", "×": "x",
        "❤️": "<3", "❤": "<3",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def _sanitize_prompt_for_transport(prompt: str) -> str:
    """Make a prompt safe to send even under an ASCII locale."""
    if prompt is None:
        return ""
    if os.environ.get("FORCE_ASCII_API_PROMPT", "").strip().lower() in {"1", "true", "yes"}:
        return _ascii_normalize(prompt)
    try:
        prompt.encode("utf-8")
        return prompt
    except UnicodeEncodeError:
        return _ascii_normalize(prompt)


def _force_ascii(text: str) -> str:
    return _ascii_normalize(text or "")


def _chat_completions_http(*, provider: str, model: str, api_key: str,
                           prompt: str, max_tokens: int, temperature: float) -> str:
    """Stdlib fallback for OpenAI-compatible chat-completions APIs (locale-proof)."""
    base_url = PROVIDER_BASE_URLS.get(provider) or "https://api.openai.com/v1"
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider} HTTP {exc.code}: {body[:1000]}") from exc
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"].strip()


def _bedrock_region() -> str:
    return (
        os.environ.get("BEDROCK_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _call_bedrock_converse(*, model: str, prompt: str,
                           max_tokens: int, temperature: float) -> str:
    """Call Amazon Bedrock through the Converse API.

    Credentials are resolved by boto3. Supported setups include the Bedrock API
    key environment variable (AWS_BEARER_TOKEN_BEDROCK) or standard AWS
    credentials (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/profile/role).
    """
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 package not installed; pip install boto3") from exc

    client = boto3.client("bedrock-runtime", region_name=_bedrock_region())
    resp = client.converse(
        modelId=model,
        messages=[{
            "role": "user",
            "content": [{"text": prompt}],
        }],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )
    parts = resp.get("output", {}).get("message", {}).get("content", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    if not texts:
        raise RuntimeError(f"Bedrock response did not contain text: {json.dumps(resp)[:1000]}")
    return "\n".join(texts).strip()


def _validate_api_key_for_header(api_key: str, key_name: str) -> str:
    """HTTP headers must be latin-1/ASCII-safe (the 'position 7' Bearer bug)."""
    key = (api_key or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(f"{key_name} is empty")
    try:
        key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{key_name} contains non-ASCII characters. Re-export the raw API key only, "
            "for example: export OPENAI_API_KEY='sk-...'. Do not include Chinese text, "
            "smart quotes, or the word 'Bearer'."
        ) from exc
    if key.lower().startswith("bearer "):
        raise RuntimeError(f"{key_name} should contain only the key, not the 'Bearer ' prefix")
    return key


def call_chat(prompt: str, model_spec: str, max_tokens: int = 650,
              temperature: float = 0.35) -> str:
    provider, model = split_provider(model_spec)
    prompt = _sanitize_prompt_for_transport(prompt)

    if provider in {"openai", "deepseek"}:
        key_name = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
        if key_name not in os.environ:
            raise RuntimeError(f"{key_name} is not set")
        api_key = _validate_api_key_for_header(os.environ[key_name], key_name)
        if os.environ.get("USE_STD_HTTP_CHAT", "").strip().lower() in {"1", "true", "yes"}:
            return _chat_completions_http(
                provider=provider, model=model, api_key=api_key,
                prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            )
        from openai import OpenAI
        kwargs: dict[str, Any] = {"api_key": api_key}
        if PROVIDER_BASE_URLS[provider]:
            kwargs["base_url"] = PROVIDER_BASE_URLS[provider]
        client = OpenAI(**kwargs)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except UnicodeEncodeError:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _force_ascii(prompt)}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return resp.choices[0].message.content.strip()

    if provider == "claude":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package not installed; pip install anthropic") from exc
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    if provider == "bedrock":
        return _call_bedrock_converse(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider: {provider}")

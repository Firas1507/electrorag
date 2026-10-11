"""
Local LLM interface for ElectroRAG.

Uses Ollama to run qwen3:8b locally.

This module ONLY talks to Ollama (streaming, error handling).
Prompt construction lives in src/generator.py, which is the single source
of truth for the grounding instructions and [C#] citation rules.
"""

import json
import sys

import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "qwen3:8b"

# Maximum number of generated tokens.
MAX_TOKENS = 1024

# Request timeout in seconds.
TIMEOUT = 180


# ============================================================
# Ollama streaming
# ============================================================

def stream_answer(prompt: str):
    """
    Send a prompt to Ollama and yield only visible answer text.

    Qwen3 may expose reasoning/thinking separately from the
    normal response. We deliberately ignore the thinking field
    and only stream the 'response' field.
    """

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,

        # Explicitly disable Qwen3 thinking.
        "think": False,

        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": 0.0,
        },
    }

    try:
        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        ) as response:

            response.raise_for_status()

            for line in response.iter_lines():

                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        "Ollama returned an invalid JSON streaming response."
                    ) from exc

                if not isinstance(data, dict):
                    raise RuntimeError(
                        "Ollama returned an invalid streaming response."
                    )

                error = data.get("error")
                if error:
                    raise RuntimeError(
                        f"Ollama reported an error for model '{MODEL}': {error}"
                    )

                # Only use the normal visible response.
                piece = data.get("response", "")

                if piece:
                    yield piece

                # Ollama marks the end of generation with done=true.
                if data.get("done", False):
                    break

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. "
            f"Make sure it is running at {OLLAMA_URL}. Details: {exc}"
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama request timed out after {TIMEOUT} seconds. "
            "The model may still be loading or the server may be unresponsive."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        response = exc.response
        status = response.status_code if response is not None else "unknown"
        details = ""
        if response is not None:
            try:
                payload = response.json()
            except ValueError:
                details = response.text.strip()
            else:
                details = (
                    payload.get("error", response.text.strip())
                    if isinstance(payload, dict)
                    else response.text.strip()
                )
        message = f"Ollama returned HTTP {status}"
        if details:
            message += f": {details}"
        raise RuntimeError(message) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}"
        ) from exc


# ============================================================
# Standalone connectivity test
# ============================================================

def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.llm "Your question here"')

    question = " ".join(sys.argv[1:])

    print(f"Model: {MODEL}")
    print(f"Question: {question}")
    print("\nAnswer:")
    print("-" * 50)

    try:
        got_text = False
        for piece in stream_answer(question):
            print(piece, end="", flush=True)
            got_text = True
        print()
        if not got_text:
            print("[ERROR] Ollama returned an empty visible response.")
    except RuntimeError as exc:
        sys.exit(f"\nError: {exc}")


if __name__ == "__main__":
    main()

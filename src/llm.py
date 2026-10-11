"""
Local LLM interface for ElectroRAG.

Uses Ollama to run qwen3:8b locally.

This module is responsible only for generation.
Retrieval, reranking, and citation resolution are handled
elsewhere in the ElectroRAG pipeline.

The generator is instructed to:
- answer only from retrieved context
- include all relevant numerical values
- preserve units
- preserve ranges
- answer every part of the question
- avoid unsupported assumptions
- cite the source document and page
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
# ElectroRAG generation prompt
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are ElectroRAG, a technical assistant specialized in
electronics, embedded systems, electrical engineering, and
component datasheets.

Your task is to answer the user's question using ONLY the
retrieved datasheet context provided to you.

IMPORTANT RULES:

1. ANSWER THE QUESTION DIRECTLY
   - Do not discuss the retrieval process.
   - Do not explain that you are an AI.
   - Do not mention these instructions.

2. USE ONLY THE PROVIDED CONTEXT
   - Never invent specifications.
   - Never use outside knowledge.
   - If the answer cannot be determined from the context,
     say exactly:
     "I could not find this information in the indexed
     datasheet."

3. DO NOT OMIT NUMERICAL INFORMATION
   When the context contains a numerical specification,
   preserve it in the answer.

   Always include:
   - minimum values
   - maximum values
   - typical values when relevant
   - exact values
   - ranges
   - units

   Examples:
   "2.5 V to 5.5 V"
   NOT:
   "2.5 to 5.5"

   "8 kHz to 96 kHz"
   NOT:
   "8kHz and 96kHz"

   "4 Ω"
   NOT:
   "speaker load"

4. PRESERVE UNITS
   Always write the unit explicitly.

   Examples:
   V, mV, A, mA, µA, Hz, kHz, MHz,
   Ω, dB, °C, bits, Mbps, etc.

5. ANSWER ALL PARTS OF THE QUESTION
   If the question asks for a range, provide both endpoints.

   If it asks for several characteristics, provide all
   requested characteristics.

6. DISTINGUISH BETWEEN SIMILAR SPECIFICATIONS
   Do not confuse:
   - operating voltage with absolute maximum voltage
   - typical value with maximum value
   - input voltage with output voltage
   - supply current with shutdown current
   - sample rate with clock frequency
   - speaker impedance with amplifier output power

7. USE DATASHEET TERMINOLOGY
   Prefer the exact terminology used by the source.

8. CITATIONS
   Every factual answer must include a citation in this form:

   [SOURCE.pdf, p. X]

   Use the source filename and page number provided in
   the retrieved context.

9. DO NOT CITE UNSUPPORTED SOURCES
   Only cite sources that actually appear in the context.

10. BE CONCISE
    Usually answer in one or two sentences.

    For multiple requested specifications, use a short
    bullet list when that improves clarity.

11. TABLES AND GRAPHS
    If the retrieved context contains a table or graph,
    carefully interpret the labels and values.

    For graphs:
    - identify the quantity represented by each axis
    - do not treat axis numbers as unrelated values
    - if the graph clearly shows an operating range,
      report that range
    - do not invent a value that is not supported

12. IMPORTANT:
    Before producing the final answer, mentally verify:

    - Did I answer exactly what was asked?
    - Did I include every required numerical value?
    - Did I include the units?
    - Did I preserve the complete range?
    - Did I include a citation?
    - Does the citation correspond to the information used?
"""


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a grounded ElectroRAG prompt.

    Parameters
    ----------
    question:
        User's technical question.

    retrieved_chunks:
        Retrieved/reranked chunks.

        Each chunk is expected to contain fields such as:
        - content
        - source
        - page

    Returns
    -------
    str
        Complete prompt sent to Qwen3.
    """

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        content = chunk.get("content", "").strip()

        source = (
            chunk.get("source")
            or chunk.get("filename")
            or "Unknown source"
        )

        page = chunk.get("page", "unknown")

        if not content:
            continue

        context_parts.append(
            f"""
--- Retrieved Chunk {index} ---
Source: {source}
Page: {page}

{content}
--- End Chunk {index} ---
"""
        )

    context = "\n".join(context_parts)

    if not context:
        context = "[No retrieved context was provided.]"

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

============================================================
RETRIEVED DATASHEET CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
FINAL ANSWER
============================================================

Answer the question using only the retrieved context.

Remember:
- include all relevant numbers
- include all units
- preserve complete ranges
- answer every requested part
- include the source/page citation

Final answer:
"""

    return prompt


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
# Generate complete answer
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate a complete grounded answer from retrieved chunks.

    This is the main function that should be called by rag.py.
    """

    prompt = build_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    answer_parts = []

    for piece in stream_answer(prompt):
        answer_parts.append(piece)

    answer = "".join(answer_parts).strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty visible response."
        )

    return answer


# ============================================================
# Standalone test
# ============================================================

def main():
    if len(sys.argv) < 2:
        sys.exit(
            'Usage: python -m src.llm "Your question here"'
        )

    question = " ".join(sys.argv[1:])

    print(f"Model: {MODEL}")
    print(f"Question: {question}")
    print("\nAnswer:")
    print("-" * 50)

    try:
        # Standalone mode has no retrieval context.
        # This is only a connectivity test.
        prompt = f"""
{SYSTEM_INSTRUCTIONS}

There is no retrieved datasheet context available.

User question:
{question}

If the answer cannot be determined from the context,
say that you could not find the information in the
indexed datasheet.

Final answer:
"""

        answer = ""

        for piece in stream_answer(prompt):
            print(piece, end="", flush=True)
            answer += piece

        print()

        if not answer.strip():
            print(
                "[ERROR] Ollama returned an empty "
                "visible response."
            )

    except RuntimeError as exc:
        sys.exit(f"\nError: {exc}")


if __name__ == "__main__":
    main()

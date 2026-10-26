import json
import os
import re
from typing import Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_udemy import config

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

PROMPT_FILE = os.path.join(BASE_DIR, "PROMPT.txt")


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )


def parse_verifier_json(raw_text: str) -> list[dict[str, str]]:
    """
    Parses LLM response into a list of {sentence, label, reason}.
    Handles markdown backticks and parsing errors gracefully.
    """
    cleaned = raw_text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        # Fallback: search for JSON array block inside the text
        match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    result = []
    for item in data:
        if isinstance(item, dict):
            sentence = str(item.get("sentence", "")).strip()
            label = str(item.get("label", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if sentence:
                result.append({
                    "sentence": sentence,
                    "label": label,
                    "reason": reason,
                })
    return result


def verify(answer: str, chunks: list[str]) -> list[dict[str, str]]:
    """
    Verifies each sentence in answer against retrieved context chunks.
    Returns a list of {sentence, label, reason}.
    If config.USE_VERIFIER is False or inputs are empty, returns [].
    """
    if not getattr(config, "USE_VERIFIER", True):
        return []

    if not answer or not answer.strip():
        return []

    if not os.path.exists(PROMPT_FILE):
        return []

    with open(PROMPT_FILE, mode="r", encoding="utf-8") as f:
        prompt_template = f.read()

    context_str = "\n\n".join(chunks) if chunks else "(No context provided)"
    prompt = prompt_template.format(context=context_str, answer=answer.strip())

    try:
        llm = get_llm()
        raw_response = llm.invoke(prompt).content

        if isinstance(raw_response, list):
            text = "".join(
                b["text"] for b in raw_response if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = str(raw_response)

        return parse_verifier_json(text)
    except Exception as e:
        # Return empty list on connection/parsing errors to prevent breaking caller
        return []

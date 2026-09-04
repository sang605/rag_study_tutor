"""Generate practice questions from your own notes."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import List, Optional

from .llm import LLMError
from .tutor import Tutor, format_context

QUIZ_PROMPT = """From the study notes below, write {n} exam-style practice questions
about "{topic}".

Return ONLY a JSON array, no other text. Each item must look like:
{{"question": "...", "answer": "...", "source": "filename"}}

Study notes:
---------------------
{context}
---------------------"""


@dataclass
class QuizItem:
    question: str
    answer: str
    source: str = ""


def _parse_json_array(text: str) -> Optional[List[dict]]:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _fallback_quiz(hits, n: int) -> List[QuizItem]:
    """No LLM? Make cloze (fill-in-the-blank) questions out of real sentences."""
    items: List[QuizItem] = []
    rng = random.Random(42)
    for hit in hits:
        for sentence in re.split(r"(?<=[.!?])\s+", hit.text):
            sentence = sentence.strip()
            words = [w for w in sentence.split() if len(w) > 5]
            if len(sentence.split()) < 8 or not words:
                continue
            target = rng.choice(words)
            blanked = sentence.replace(target, "______", 1)
            items.append(
                QuizItem(
                    question=f"Fill in the blank: {blanked}",
                    answer=target.strip(".,;:()"),
                    source=hit.source,
                )
            )
            if len(items) >= n:
                return items
    return items


def make_quiz(tutor: Tutor, topic: str, n: int = 5, k: int = 6) -> List[QuizItem]:
    hits = tutor.store.search(topic, k=k)
    if not hits:
        return []

    if not tutor.llm.generative:
        return _fallback_quiz(hits, n)

    prompt = QUIZ_PROMPT.format(n=n, topic=topic, context=format_context(hits))
    try:
        raw = tutor.llm.generate(prompt)
    except LLMError:
        return _fallback_quiz(hits, n)

    data = _parse_json_array(raw)
    if not data:
        return _fallback_quiz(hits, n)

    items = []
    for entry in data[:n]:
        if isinstance(entry, dict) and entry.get("question"):
            items.append(
                QuizItem(
                    question=str(entry.get("question", "")).strip(),
                    answer=str(entry.get("answer", "")).strip(),
                    source=str(entry.get("source", "")).strip(),
                )
            )
    return items or _fallback_quiz(hits, n)

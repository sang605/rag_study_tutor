"""The RAG loop: retrieve -> build a grounded prompt -> answer with sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import Settings, settings as default_settings
from .llm import BaseLLM, LLMError, get_llm
from .retriever import load_index
from .vectorstore import Hit, VectorStore

SYSTEM_PROMPT = (
    "You are a patient study tutor. Answer ONLY from the study notes given to you. "
    "If the notes do not contain the answer, say so plainly instead of inventing one. "
    "Explain in simple language, use short examples, and cite the source file in "
    "square brackets like [notes.pdf] after the facts you use."
)

PROMPT_TEMPLATE = """Study notes:
---------------------
{context}
---------------------

Question: {question}

Answer using only the notes above. Cite sources like [filename]."""


@dataclass
class Answer:
    text: str
    hits: List[Hit] = field(default_factory=list)

    @property
    def sources(self) -> List[str]:
        seen = []
        for hit in self.hits:
            if hit.source not in seen:
                seen.append(hit.source)
        return seen


def format_context(hits: List[Hit]) -> str:
    return "\n\n".join(f"[{hit.source}]\n{hit.text}" for hit in hits)


def extractive_answer(question: str, hits: List[Hit]) -> str:
    """Used when no LLM is configured - shows the passages instead of a written answer."""
    if not hits:
        return "I could not find anything about that in your notes."
    parts = [
        "No language model configured, so here is what your notes say "
        "(add a free API key to .env for written answers).\n"
    ]
    for i, hit in enumerate(hits, start=1):
        parts.append(f"{i}. [{hit.source}] (match {hit.score:.2f})\n{hit.text}\n")
    return "\n".join(parts)


class Tutor:
    def __init__(
        self,
        store: VectorStore | None = None,
        llm: BaseLLM | None = None,
        settings: Settings = default_settings,
    ):
        self.settings = settings
        self.store = store if store is not None else load_index(settings)
        self.llm = llm if llm is not None else get_llm(settings)

    def ask(self, question: str, k: int | None = None) -> Answer:
        k = k or self.settings.top_k
        hits = self.store.search(question, k=k)

        if not hits:
            return Answer("Your index is empty. Run `python main.py index` first.", [])

        if not self.llm.generative:
            return Answer(extractive_answer(question, hits), hits)

        prompt = PROMPT_TEMPLATE.format(
            context=format_context(hits), question=question
        )
        try:
            text = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        except LLMError as exc:
            text = f"({exc})\n\n" + extractive_answer(question, hits)
        return Answer(text, hits)

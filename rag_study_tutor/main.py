"""Command line entry point.

    python main.py index
    python main.py ask "what is a foreign key?"
    python main.py chat
    python main.py quiz "normalization" -n 5
"""

from __future__ import annotations

import argparse
import sys

from ragtutor.config import settings
from ragtutor.llm import get_llm
from ragtutor.quiz import make_quiz
from ragtutor.retriever import build_index, load_index
from ragtutor.tutor import Tutor


def print_answer(answer) -> None:
    print("\n" + answer.text.strip() + "\n")
    if answer.sources:
        print("Sources: " + ", ".join(answer.sources))
    print("-" * 60)


def cmd_index(_args) -> None:
    print(f"Reading notes from {settings.data_dir} ...")
    store = build_index(settings)
    print(f"Done. {len(store)} chunks indexed.")


def cmd_ask(args) -> None:
    tutor = Tutor(settings=settings)
    print(f"(model: {tutor.llm.name})")
    print_answer(tutor.ask(args.question, k=args.k))


def cmd_chat(_args) -> None:
    tutor = Tutor(settings=settings)
    print(f"RAG Study Tutor - model: {tutor.llm.name}. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit", "q"}:
            return
        if not question:
            continue
        print_answer(tutor.ask(question))


def cmd_quiz(args) -> None:
    tutor = Tutor(settings=settings)
    items = make_quiz(tutor, args.topic, n=args.n)
    if not items:
        print("Nothing found on that topic in your notes.")
        return
    print(f"\nPractice quiz on '{args.topic}' ({len(items)} questions)\n")
    for i, item in enumerate(items, start=1):
        print(f"Q{i}. {item.question}")
    print("\n--- answers ---")
    for i, item in enumerate(items, start=1):
        src = f"  [{item.source}]" if item.source else ""
        print(f"A{i}. {item.answer}{src}")


def cmd_status(_args) -> None:
    llm = get_llm(settings)
    print(f"data folder    : {settings.data_dir}")
    print(f"storage folder : {settings.storage_dir}")
    print(f"llm            : {llm.name}")
    try:
        store = load_index(settings)
        print(f"index          : {len(store)} chunks, embedder '{store.embedder.kind}'")
    except FileNotFoundError:
        print("index          : not built yet (run: python main.py index)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RAG Study Tutor")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="read data/ and build the search index").set_defaults(
        func=cmd_index
    )

    p_ask = sub.add_parser("ask", help="ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None, help="how many chunks to retrieve")
    p_ask.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="ask questions in a loop").set_defaults(func=cmd_chat)

    p_quiz = sub.add_parser("quiz", help="generate practice questions")
    p_quiz.add_argument("topic")
    p_quiz.add_argument("-n", type=int, default=5, help="number of questions")
    p_quiz.set_defaults(func=cmd_quiz)

    sub.add_parser("status", help="show current configuration").set_defaults(
        func=cmd_status
    )

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

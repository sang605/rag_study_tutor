# RAG Study Tutor

A study assistant that answers questions **from your own notes** (PDF / TXT / Markdown)
instead of guessing. It also generates practice quizzes from what you're studying.

RAG = **R**etrieval **A**ugmented **G**eneration:

```
your notes -> split into chunks -> turn into vectors -> store
question   -> turn into a vector -> find the closest chunks -> feed them to an LLM -> answer + sources
```

**It runs with zero API keys and zero heavy downloads.** The default embedder is pure
NumPy and the default "LLM" is an extractive answerer that quotes your notes. Add a free
API key (Gemini or Groq) or run Ollama locally and the same code produces real generated
answers. Nothing else changes.

---

## 1. Setup (Windows)

Open a terminal in this folder and run:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux the activate line is `source .venv/bin/activate`.

> `requirements.txt` only has small packages. The optional, bigger ones
> (`sentence-transformers`, `torch`) are listed at the bottom of that file, commented out.

## 2. Add your notes

Drop `.pdf`, `.txt` or `.md` files into the `data/` folder. A sample notes file is
already there so you can test straight away.

## 3. Build the index

```bat
python main.py index
```

This reads `data/`, splits everything into chunks, embeds them and saves the index into
`storage/`. Re-run it whenever you add or change notes.

## 4. Ask questions

```bat
python main.py ask "what is normalization in DBMS?"
python main.py chat
python main.py quiz "transactions" -n 5
```

Or use the web UI:

```bat
streamlit run app.py
```

---

## 5. Turning on a real LLM (all free options)

Copy `.env.example` to `.env` and fill in **one** of these.

| Provider | Cost | What to set |
|---|---|---|
| **Google Gemini** | free tier | `RAG_LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...` from aistudio.google.com |
| **Groq** | free tier | `RAG_LLM_PROVIDER=groq`, `GROQ_API_KEY=...` from console.groq.com |
| **Ollama** | free, runs offline on your laptop | install Ollama, `ollama pull llama3.2`, then `RAG_LLM_PROVIDER=ollama` |
| **none** | default | extractive answers quoted from your notes, no key needed |

`RAG_LLM_PROVIDER=auto` (the default) picks whichever one is configured, and falls back
to extractive mode if none is.

Better embeddings are optional too: `pip install sentence-transformers` and set
`RAG_EMBEDDER=sentence-transformers`. Retrieval quality goes up noticeably; the first run
downloads ~90 MB.

---

## 6. What each file does

```
main.py                 command line entry point (index / ask / chat / quiz)
app.py                  Streamlit web UI
ragtutor/
  config.py             settings, read from .env with sensible defaults
  loaders.py            reads .txt .md .pdf out of data/
  chunking.py           splits long text into overlapping chunks
  embeddings.py         text -> vectors (NumPy hashing embedder, or sentence-transformers)
  vectorstore.py        stores vectors + metadata, cosine similarity search, save/load
  retriever.py          ties loading + chunking + embedding + search together
  llm.py                one interface over gemini / groq / ollama / extractive fallback
  tutor.py              builds the RAG prompt, returns an answer with sources
  quiz.py               generates practice questions from your notes
tests/                  pytest tests you can run with: pytest
data/                   your study material goes here
storage/                generated index (safe to delete, just re-run `python main.py index`)
```

## 7. Ideas to extend it

- Keep chat history so follow-up questions ("explain that simpler") have context.
- Add a re-ranker: retrieve 20 chunks, ask the LLM to pick the best 4.
- Track which questions you got wrong in the quiz and retrieve more of that topic.
- Swap the NumPy store for FAISS or Chroma once the collection gets large.
- Add `.docx` and `.pptx` loaders (python-docx, python-pptx).

## 8. Push it to GitHub

```bat
git init
git add .
git commit -m "RAG study tutor"
git branch -M main
git remote add origin https://github.com/<your-username>/rag_study_tutor.git
git push -u origin main
```

`.gitignore` already keeps `.venv/`, `.env`, `storage/` and your personal notes out of the repo.

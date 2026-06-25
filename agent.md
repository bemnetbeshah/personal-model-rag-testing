# Agent Entry Point

Authoritative project instructions: no project-specific `AGENTS.md` exists yet.

This folder is a personal model RAG testing project using local Ollama models. The project virtual environment lives at `.venv` and uses CPython 3.14.4.

```sh
source .venv/bin/activate
python --version
python main.py "What are the main limitations of the paper?"
python app.py
```

Current conventions:

- Install Python dependencies inside `.venv`.
- Recreate dependencies with `python -m pip install -r requirements.txt` after activating the environment.
- `main.py` loads the research paper, splits it, embeds chunks with Ollama's `nomic-embed-text`, stores them in `InMemoryVectorStore`, exposes that store as the `search_research_paper` tool, and answers with a selected Ollama or Anthropic chat model.
- `app.py` serves a lightweight browser UI at `http://127.0.0.1:8001` by default. Set `RAG_UI_PORT` to choose another port. The UI includes a streaming compare view that sends one question to two selected local Ollama or configured Anthropic models, renders responses as Markdown, and can stream an optional LLM-as-judge verdict from a chosen judge model.
- Set `OLLAMA_CHAT_MODEL` in `.env` to choose the local Ollama chat model; the current default is `gemma4:e4b`.
- Keep this file updated if a project `AGENTS.md` is added later.
- Keep `index.md` files in sync when files or folders are added, removed, or renamed.

# Local Model RAG Compare Tool

This is a personal tool I use to test local model capabilities.

It loads a research paper, chunks it with LangChain, embeds the chunks with an Ollama embedding model, and lets me compare answers from different chat models side by side. The UI can compare local Ollama models against each other or against a configured Anthropic model, and it can optionally use an LLM-as-judge to score the two responses.

## What It Does

- Loads `2606.24855v1.pdf` as the source document.
- Splits the document with LangChain's recursive text splitter.
- Embeds chunks with `nomic-embed-text` through Ollama.
- Stores chunks in LangChain's `InMemoryVectorStore`.
- Streams side-by-side model responses in the browser.
- Renders model outputs as Markdown.
- Optionally streams a judge model verdict.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` for your local setup. At minimum, Ollama should be running with the embedding model available:

```sh
ollama pull nomic-embed-text
```

Set `OLLAMA_CHAT_MODEL` to the local model you want as the default.

## Run

```sh
source .venv/bin/activate
python app.py
```

Open:

```text
http://127.0.0.1:8001
```

## Notes

- `.env` is ignored and should not be committed.
- The vector store is in memory and rebuilt when the app starts.
- Larger local models can take a while to load and generate, especially the first time they are selected.

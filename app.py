import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import socket
from urllib.parse import urlparse

import ollama

from main import (
    ANTHROPIC_MODEL,
    CHAT_MODEL,
    EMBEDDING_MODEL,
    build_retrieval_context,
    build_vector_store,
    load_document,
    ask_agent,
    judge_answers,
    stream_agent,
    stream_judge_answers,
    split_documents,
)


PROJECT_ROOT = Path(__file__).parent
UI_ROOT = PROJECT_ROOT / "ui"
HOST = "127.0.0.1"
PORT = int(os.getenv("RAG_UI_PORT", "8001"))

_vector_store = None
_vector_store_lock = threading.Lock()


class ClientDisconnected(Exception):
    """Raised when the browser closes a streaming response."""


def get_local_ollama_chat_models() -> list[str]:
    """Return local Ollama chat models, excluding embedding and cloud entries."""
    models = []
    for model in ollama.list().get("models", []):
        name = model.get("name") or model.get("model")
        if not name:
            continue
        if name == EMBEDDING_MODEL or name.startswith(f"{EMBEDDING_MODEL}:"):
            continue
        if "cloud" in name:
            continue
        models.append(name)
    return sorted(models)


def get_model_options() -> dict:
    local_models = get_local_ollama_chat_models()
    if CHAT_MODEL not in local_models:
        local_models.insert(0, CHAT_MODEL)

    return {
        "default": {"provider": "ollama", "model": CHAT_MODEL},
        "providers": [
            {
                "id": "ollama",
                "label": "Local Ollama",
                "models": local_models,
                "enabled": True,
            },
            {
                "id": "anthropic",
                "label": "Anthropic",
                "models": [ANTHROPIC_MODEL],
                "enabled": bool(os.getenv("ANTHROPIC_API_KEY")),
            },
        ],
    }


def validate_model_choice(provider: str, model: str) -> None:
    options = get_model_options()
    for provider_config in options["providers"]:
        if provider_config["id"] != provider:
            continue
        if not provider_config["enabled"]:
            raise ValueError(f"{provider_config['label']} is not configured.")
        if model not in provider_config["models"]:
            raise ValueError(f"Unsupported model: {model}")
        return

    raise ValueError(f"Unsupported provider: {provider}")


def answer_with_model(question: str, model_config: dict) -> dict:
    provider = str(model_config.get("provider", "")).strip()
    model = str(model_config.get("model", "")).strip()
    validate_model_choice(provider, model)

    answer = ask_agent(
        question,
        get_vector_store(),
        provider=provider,
        model=model,
    )
    return {
        "answer": answer,
        "provider": provider,
        "chat_model": model,
        "embedding_model": EMBEDDING_MODEL,
    }


def stream_answer_with_model(
    question: str,
    context: str,
    model_config: dict,
    lane: str,
    events: Queue,
) -> None:
    provider = str(model_config.get("provider", "")).strip()
    model = str(model_config.get("model", "")).strip()
    validate_model_choice(provider, model)

    answer_parts = []
    try:
        events.put(
            {
                "type": "start",
                "lane": lane,
                "provider": provider,
                "chat_model": model,
                "embedding_model": EMBEDDING_MODEL,
            }
        )
        for text in stream_agent(
            question,
            context,
            provider=provider,
            model=model,
        ):
            answer_parts.append(text)
            events.put({"type": "delta", "lane": lane, "text": text})

        events.put(
            {
                "type": "done",
                "lane": lane,
                "provider": provider,
                "chat_model": model,
                "answer": "".join(answer_parts),
                "embedding_model": EMBEDDING_MODEL,
            }
        )
    except Exception as error:
        events.put({"type": "error", "lane": lane, "error": str(error)})


def get_vector_store():
    """Build the in-memory vector store once and reuse it for UI requests."""
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                documents = load_document()
                chunks = split_documents(documents)
                _vector_store = build_vector_store(chunks)
    return _vector_store


class RagRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_ROOT), **kwargs)

    def do_GET(self):
        parsed_path = urlparse(self.path).path
        if parsed_path == "/api/health":
            self.send_json(
                {
                    "status": "ok",
                    "chat_model": CHAT_MODEL,
                    "embedding_model": EMBEDDING_MODEL,
                    "model_options": get_model_options(),
                }
            )
            return

        if parsed_path == "/":
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self):
        parsed_path = urlparse(self.path).path
        if parsed_path not in {"/api/ask", "/api/compare", "/api/compare-stream"}:
            self.send_error(404, "Not found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                self.send_json({"error": "Question is required."}, status=400)
                return

            if parsed_path == "/api/compare-stream":
                models = payload.get("models", [])
                if not isinstance(models, list) or len(models) != 2:
                    self.send_json({"error": "Exactly two models are required."}, status=400)
                    return

                self.stream_compare_response(question, models, payload.get("judge"))
                return

            if parsed_path == "/api/compare":
                models = payload.get("models", [])
                if not isinstance(models, list) or len(models) != 2:
                    self.send_json({"error": "Exactly two models are required."}, status=400)
                    return

                results = [None, None]
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {
                        executor.submit(answer_with_model, question, model_config): index
                        for index, model_config in enumerate(models)
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        results[index] = future.result()

                payload_result = {"results": results}
                judge_config = payload.get("judge")
                if isinstance(judge_config, dict) and judge_config.get("enabled"):
                    judge_provider = str(judge_config.get("provider", "")).strip()
                    judge_model = str(judge_config.get("model", "")).strip()
                    validate_model_choice(judge_provider, judge_model)
                    payload_result["judge"] = {
                        "provider": judge_provider,
                        "chat_model": judge_model,
                        "verdict": judge_answers(
                            question,
                            get_vector_store(),
                            results[0],
                            results[1],
                            provider=judge_provider,
                            model=judge_model,
                        ),
                    }

                self.send_json(payload_result)
                return

            provider = str(payload.get("provider", "ollama")).strip()
            model = str(payload.get("model", CHAT_MODEL)).strip()
            self.send_json(answer_with_model(question, {"provider": provider, "model": model}))
        except Exception as error:
            self.send_json({"error": str(error)}, status=500)

    def stream_compare_response(self, question: str, models: list, judge_config: dict | None):
        validate_model_choice(
            str(models[0].get("provider", "")).strip(),
            str(models[0].get("model", "")).strip(),
        )
        validate_model_choice(
            str(models[1].get("provider", "")).strip(),
            str(models[1].get("model", "")).strip(),
        )

        judge_enabled = isinstance(judge_config, dict) and judge_config.get("enabled")
        if judge_enabled:
            judge_provider = str(judge_config.get("provider", "")).strip()
            judge_model = str(judge_config.get("model", "")).strip()
            validate_model_choice(judge_provider, judge_model)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def send_event(payload: dict) -> None:
            body = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                raise ClientDisconnected()

        try:
            send_event({"type": "setup", "message": "Preparing retrieved paper context..."})
            context = build_retrieval_context(question, get_vector_store())
            send_event({"type": "context_ready", "embedding_model": EMBEDDING_MODEL})
        except ClientDisconnected:
            return

        events = Queue()
        threads = [
            threading.Thread(
                target=stream_answer_with_model,
                args=(question, context, models[0], "left", events),
                daemon=True,
            ),
            threading.Thread(
                target=stream_answer_with_model,
                args=(question, context, models[1], "right", events),
                daemon=True,
            ),
        ]

        for thread in threads:
            thread.start()

        results = {}
        while len(results) < 2:
            event = events.get()
            try:
                send_event(event)
            except ClientDisconnected:
                return
            if event["type"] == "done":
                results[event["lane"]] = event
            elif event["type"] == "error":
                results[event["lane"]] = {
                    "provider": "unknown",
                    "chat_model": "unknown",
                    "answer": event["error"],
                }

        for thread in threads:
            thread.join(timeout=0.1)

        if judge_enabled:
            judge_result = {
                "type": "judge_start",
                "provider": judge_provider,
                "chat_model": judge_model,
            }
            try:
                send_event(judge_result)
            except ClientDisconnected:
                return
            judge_parts = []
            left_result = results["left"]
            right_result = results["right"]
            try:
                for text in stream_judge_answers(
                    question,
                    context,
                    left_result,
                    right_result,
                    provider=judge_provider,
                    model=judge_model,
                ):
                    judge_parts.append(text)
                    send_event({"type": "judge_delta", "text": text})
                send_event(
                    {
                        "type": "judge_done",
                        "provider": judge_provider,
                        "chat_model": judge_model,
                        "verdict": "".join(judge_parts),
                    }
                )
            except Exception as error:
                if isinstance(error, ClientDisconnected):
                    return
                try:
                    send_event({"type": "judge_error", "error": str(error)})
                except ClientDisconnected:
                    return

        try:
            send_event({"type": "complete"})
        except ClientDisconnected:
            return

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    server = ThreadingHTTPServer((HOST, PORT), RagRequestHandler)
    print(f"Personal model RAG testing UI running at http://{HOST}:{PORT}")
    print(f"Chat model: {CHAT_MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    server.serve_forever()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


HandlerFn = Callable[[dict[str, Any]], dict[str, Any]]


def serve(handler: HandlerFn, runtime_name: str) -> None:
    port = int(os.environ.get("PORT", "8000"))

    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.respond(200, {"ok": True, "runtime": runtime_name})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                job_payload = payload.get("input") if isinstance(payload.get("input"), dict) else payload
                self.respond(200, handler(job_payload))
            except Exception as exc:
                self.respond(500, {"ok": False, "runtime": runtime_name, "error": str(exc)})

        def respond(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{runtime_name}] {fmt % args}")

    server = ThreadingHTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"{runtime_name} listening on 0.0.0.0:{port}")
    server.serve_forever()

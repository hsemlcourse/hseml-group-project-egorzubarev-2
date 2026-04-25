from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.project_config import RAW_DIR

OUTPUT_DIR = RAW_DIR / "browser_batches"


class BrowserBridgeHandler(BaseHTTPRequestHandler):
    def _send_headers(self, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_headers()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        file_name = parse_qs(parsed.query).get("name", ["payload.json"])[0]
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target_path = OUTPUT_DIR / file_name
        target_path.write_bytes(payload)

        self._send_headers()
        self.wfile.write(json.dumps({"saved_to": str(target_path)}).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return None


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), BrowserBridgeHandler)
    print(f"Browser bridge listening on http://127.0.0.1:8765, writing to {OUTPUT_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
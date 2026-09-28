import json
import tarfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "0.0.0.0"
PORT = 8765

BASE_VERSION = 30
TARGET_LAYER = 0
TARGET_EXPERT = 6

M7_DIR = Path("checkpoints/m7")
UPLOAD_DIR = M7_DIR / "uploads"
BUNDLE_PATH = M7_DIR / "m7_worker_bundle.tar.gz"

SNAPSHOT = Path(
    "checkpoints/m5_candidates/global_step_0030"
)


def prepare_bundle():
    M7_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        BUNDLE_PATH,
        "w:gz",
    ) as tar:
        tar.add(
            "ml",
            arcname="ml",
        )

        tar.add(
            SNAPSHOT,
            arcname=(
                "checkpoints/"
                "m5_candidates/"
                "global_step_0030"
            ),
        )

    print(
        "Worker bundle:",
        BUNDLE_PATH,
    )

    print(
        "Bundle bytes:",
        BUNDLE_PATH.stat().st_size,
    )


def send_json(handler, data, status=200):
    payload = json.dumps(
        data,
        indent=2,
    ).encode("utf-8")

    handler.send_response(status)
    handler.send_header(
        "Content-Type",
        "application/json",
    )
    handler.send_header(
        "Content-Length",
        str(len(payload)),
    )
    handler.end_headers()
    handler.wfile.write(payload)


class Handler(BaseHTTPRequestHandler):

    def log_message(
        self,
        fmt,
        *args,
    ):
        print(
            f"[HTTP] {self.address_string()} "
            + fmt % args
        )

    def do_GET(self):
        if self.path == "/health":
            send_json(
                self,
                {
                    "status": "ok",
                    "round": 1,
                    "base_global_version":
                        BASE_VERSION,
                    "target_layer":
                        TARGET_LAYER,
                    "target_expert":
                        TARGET_EXPERT,
                },
            )
            return

        if self.path == "/bundle":
            data = BUNDLE_PATH.read_bytes()

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/gzip",
            )
            self.send_header(
                "Content-Length",
                str(len(data)),
            )
            self.end_headers()

            self.wfile.write(data)
            return

        if self.path == "/status":
            delta = (
                UPLOAD_DIR
                / "client_B_expert_delta.safetensors"
            )

            metadata = (
                UPLOAD_DIR
                / "client_B_metadata.json"
            )

            send_json(
                self,
                {
                    "delta_received":
                        delta.exists(),
                    "metadata_received":
                        metadata.exists(),
                    "delta_bytes":
                        delta.stat().st_size
                        if delta.exists()
                        else 0,
                },
            )
            return

        send_json(
            self,
            {"error": "not found"},
            status=404,
        )

    def do_POST(self):
        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        body = self.rfile.read(
            content_length
        )

        if self.path == "/upload/client_B/delta":
            path = (
                UPLOAD_DIR
                / "client_B_expert_delta.safetensors"
            )

            path.write_bytes(body)

            print()
            print("CLIENT B DELTA RECEIVED")
            print("Saved:", path)
            print("Bytes:", len(body))

            send_json(
                self,
                {
                    "status": "saved",
                    "bytes": len(body),
                },
            )
            return

        if self.path == "/upload/client_B/metadata":
            path = (
                UPLOAD_DIR
                / "client_B_metadata.json"
            )

            path.write_bytes(body)

            print()
            print("CLIENT B METADATA RECEIVED")
            print("Saved:", path)

            send_json(
                self,
                {
                    "status": "saved",
                    "bytes": len(body),
                },
            )
            return

        send_json(
            self,
            {"error": "not found"},
            status=404,
        )


def main():
    prepare_bundle()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    print()
    print("FedEdgeMoE - M7 Coordinator")
    print()
    print(
        f"Listening on 0.0.0.0:{PORT}"
    )

    print(
        "Health endpoint: /health"
    )

    print(
        "Bundle endpoint: /bundle"
    )

    print(
        "Status endpoint: /status"
    )

    print()
    print(
        "Waiting for physical edge client..."
    )

    server.serve_forever()


if __name__ == "__main__":
    main()

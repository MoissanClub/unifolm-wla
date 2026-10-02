"""WBT msgpack HTTP server, bound to loopback by default. Single model worker."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from model_server.tools.msgpack_numpy import packb, unpackb
from .contract import IMAGE_SIZE, PROTOCOL, WBTAdapter

MAX_MESSAGE = 24 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8601, type=int)
    parser.add_argument("--unnorm-key", default="UnifoLM_WBT")
    parser.add_argument("--profile", choices=("wbt", "stationary_arms"), default="wbt")
    parser.add_argument("--allow-missing-cameras", action="store_true")
    parser.add_argument("--image-size", nargs=2, type=int, default=IMAGE_SIZE, metavar=("H", "W"))
    parser.add_argument("--engine", help="Experimental TensorRT action-head engine")
    args = parser.parse_args()
    from .inference import Policy
    policy = Policy(args.checkpoint, args.image_size, args.engine)
    adapter = WBTAdapter(policy.model.norm_stats, args.unnorm_key, args.allow_missing_cameras, args.profile)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def reply(self, code, data):
            body = packb(data)
            self.send_response(code)
            self.send_header("Content-Type", "application/msgpack")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, {"error": "Unknown route"})
                return
            self.reply(200, {"protocol": PROTOCOL, "unnorm_key": adapter.key,
                             "horizon": policy.model.action_horizon, "fps": 30,
                             "image_size": args.image_size, "allow_missing_cameras": args.allow_missing_cameras,
                             "action_profile": args.profile,
                             "backend": "pytorch+trt-head" if args.engine else "pytorch-bf16"})

        def do_POST(self):
            if self.path != "/predict":
                self.reply(404, {"error": "Unknown route"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_MESSAGE:
                    raise ValueError("Invalid message length")
                obs = unpackb(self.rfile.read(size))
                example = adapter.example(obs)
                pred, metrics = policy.predict(example)
                result = adapter.decode(pred, obs)
                result.update(protocol=PROTOCOL, frame_id=obs["frame_id"], metrics=metrics)
                self.reply(200, result)
                print(json.dumps({"frame_id": obs["frame_id"], **metrics}), flush=True)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception as exc:
                self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    print(f"Listening on http://{args.host}:{args.port}; no robot commands", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

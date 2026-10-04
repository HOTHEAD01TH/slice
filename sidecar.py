#!/usr/bin/env python3
"""Local Laya sidecar.

Speaks POST /v1/systemone and POST /v1/systemone/batch the way laya.serve does,
and calls laya.Router. Never returns fabricated scores.

This is not the archived third-party laya-serve package. That package defaults
to a fake backend unless LAYA_SERVE_BACKEND=laya. If that variable is set to
anything other than "laya", this process exits rather than serve stand-in answers.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("laya-sidecar")

HOST = os.environ.get("LAYA_SIDECAR_HOST", "127.0.0.1")
PORT = int(os.environ.get("LAYA_SIDECAR_PORT", "8101"))
MAX_BODY = 2 * 1024 * 1024

_router: Any = None
_lock = threading.Lock()
_import_logged = False


def _refuse_fake_backend() -> None:
    backend = os.environ.get("LAYA_SERVE_BACKEND", "").strip().lower()
    if backend and backend != "laya":
        sys.exit(
            "refusing to start: LAYA_SERVE_BACKEND=%s would be a fake. "
            "Set LAYA_SERVE_BACKEND=laya or unset it. This sidecar only calls laya.Router."
            % backend
        )


def _probe_version() -> str | None:
    global _import_logged
    try:
        import laya
    except Exception as exc:
        if not _import_logged:
            log.warning("laya is not importable: %s", exc)
            _import_logged = True
        return None
    return str(getattr(laya, "__version__", "unknown"))


def _router_instance() -> Any:
    global _router
    if _router is not None:
        return _router
    from laya import Router
    # Lazy checkpoints. Do not preload and do not download weights here.
    _router = Router()
    return _router


def _model_loaded() -> bool:
    if _router is None:
        return False
    loaded = getattr(_router, "loaded", None)
    if isinstance(loaded, (list, tuple, set)):
        return len(loaded) > 0
    return bool(loaded)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # numpy scalars and similar
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _min_confidence(body: dict) -> float | None:
    if "min_confidence" not in body or body["min_confidence"] is None:
        return None
    raw = body["min_confidence"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("min_confidence must be a number from 0 to 1")
    value = float(raw)
    if value != value or value < 0 or value > 1:
        raise ValueError("min_confidence must be a number from 0 to 1")
    return value


def _strip_action(result: dict) -> dict:
    """Drop action.act_probability. Abstention is min_confidence only."""
    answers = result.get("answers")
    if isinstance(answers, dict):
        for answer in answers.values():
            if isinstance(answer, dict):
                answer.pop("action", None)
    return result


def _call_predict(state: Any, questions: Any, min_confidence: float | None) -> dict:
    router = _router_instance()
    kwargs: dict[str, Any] = {}
    if min_confidence is not None:
        kwargs["min_confidence"] = min_confidence
    result = router.predict(state, questions, **kwargs)
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict) or not result["answers"]:
        raise RuntimeError("laya returned no answers")
    return _strip_action(result)


def _call_batch(states: list, questions: Any, min_confidence: float | None) -> list:
    router = _router_instance()
    kwargs: dict[str, Any] = {}
    if min_confidence is not None:
        kwargs["min_confidence"] = min_confidence
    if hasattr(router, "predict_batch"):
        requests = [{"state": state, "questions": questions} for state in states]
        results = router.predict_batch(requests, **kwargs)
    else:
        results = [_call_predict(state, questions, min_confidence) for state in states]
    if not isinstance(results, list) or len(results) != len(states):
        raise RuntimeError("laya batch length mismatch")
    cleaned = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict) or not result["answers"]:
            raise RuntimeError("laya returned no answers")
        cleaned.append(_strip_action(result))
    return cleaned


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - " + fmt, self.address_string(), *args)

    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Any:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send(400, {"error": "Content-Length required"})
            return None
        try:
            length = int(length_header)
        except ValueError:
            self._send(400, {"error": "bad Content-Length"})
            return None
        if length < 0 or length > MAX_BODY:
            self._send(413, {"error": "request body too large"})
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._send(400, {"error": "request body must be valid JSON"})
            return None

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/health":
            self._send(404, {"error": "not found"})
            return
        version = _probe_version()
        if version is None:
            self._send(503, {"error": "Laya not running"})
            return
        self._send(200, {
            "status": "ok",
            "laya_version": version,
            "model_loaded": _model_loaded(),
        })

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/systemone", "/v1/systemone/batch"):
            self._send(404, {"error": "not found"})
            return
        if _probe_version() is None:
            self._send(503, {"error": "Laya not running"})
            return
        body = self._read_json()
        if body is None:
            return
        if not isinstance(body, dict) or not isinstance(body.get("questions"), dict) or not body["questions"]:
            self._send(400, {"error": "questions must be a non-empty object"})
            return
        try:
            min_confidence = _min_confidence(body)
        except ValueError as exc:
            self._send(422, {"error": str(exc)})
            return
        try:
            with _lock:
                if path == "/v1/systemone":
                    if "state" not in body or body.get("state") is None:
                        self._send(400, {"error": "state is required"})
                        return
                    result = _call_predict(body["state"], body["questions"], min_confidence)
                    self._send(200, _jsonable(result))
                    return
                states = body.get("states")
                if not isinstance(states, list) or not states or len(states) > 64:
                    self._send(400, {"error": "states must be a non-empty list of at most 64"})
                    return
                results = _call_batch(states, body["questions"], min_confidence)
                total_in = 0
                total_out = 0
                for result in results:
                    usage = result.get("usage") or {}
                    total_in += int(usage.get("input_tokens") or 0)
                    total_out += int(usage.get("output_tokens") or 0)
                self._send(200, _jsonable({
                    "results": results,
                    "total_usage": {"input_tokens": total_in, "output_tokens": total_out},
                }))
        except ValueError as exc:
            log.warning("rejected request: %s", exc)
            self._send(422, {"error": str(exc)})
        except Exception:
            log.exception("inference failed")
            self._send(503, {"error": "Laya not running"})


def main() -> None:
    _refuse_fake_backend()
    os.environ["LAYA_SERVE_BACKEND"] = "laya"
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("laya sidecar on http://%s:%s (weights download on first predict, not at boot)", HOST, PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

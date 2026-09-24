"""quran-jeb server.

jeb classifies the intent of an Arabic question; the verified index produces
the answer. Run:  python server.py  ->  http://127.0.0.1:8001

Single-tenant by design: one model in memory, one inference at a time.
"""
import json, os, time, threading, http.server, socketserver, urllib.parse, sys

# Log lines carry Arabic; a redirected stdout on Windows defaults to cp1252 and would raise.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                         # noqa: BLE001
        pass

HOST = os.environ.get("HOST", "127.0.0.1")       # loopback only unless you say otherwise
PORT = int(os.environ.get("PORT", 8001))
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.environ.get("JEB_MODEL", "IJyad/jeb")
THREADS = int(os.environ.get("JEB_THREADS", 4))
MAX_BODY = 64 * 1024
sys.path.insert(0, HERE)

_state = {"model": None, "status": "cold", "error": None,
          "index": "cold", "index_error": None}
_lock = threading.Lock()


def load_model():
    """Pull weights from the Hub on first use, then keep them resident."""
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        _state["status"] = "loading"
        import torch
        torch.set_num_threads(THREADS)
        from huggingface_hub import snapshot_download
        import model as jeb

        path = snapshot_download(
            MODEL_ID,
            allow_patterns=["model.safetensors", "encoder/*", "tokenizer/*"],
        )
        _state["model"] = jeb.load(path)
        _state["status"] = "ready"
        print(f"[jeb] loaded from {MODEL_ID} ({THREADS} threads)", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        _state["status"] = "error"
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[jeb] load failed: {_state['error']}", flush=True)


def load_quran():
    """Build the exact index alongside the model."""
    try:
        _state["index"] = "loading"
        from quran_index import get_index
        ix = get_index()
        _state["index"] = "ready"
        print(f"[quran] indexed {len(ix.words):,} verified words, {ix.page_count} pages", flush=True)
    except Exception as exc:                                  # noqa: BLE001
        _state["index"] = "error"
        _state["index_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[quran] index unavailable: {_state['index_error']}", flush=True)


def ask(state, questions):
    model = _state["model"]
    if model is None:
        raise RuntimeError(_state["error"] or "model not ready")
    with _lock:
        t0 = time.perf_counter()
        answers = model.predict(state, questions)
        ms = (time.perf_counter() - t0) * 1000
    return {"answers": answers, "ms": round(ms, 1)}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=os.path.join(HERE, "static"), **kw)

    def log_message(self, fmt, *args):                        # static file noise off
        pass

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path == "/api/status":
            ready = _state["status"] == "ready" and _state["index"] == "ready"
            return self._send(200, {
                "status": "ready" if ready else ("error" if "error" in (_state["status"], _state["index"]) else "loading"),
                "model": _state["status"], "index": _state["index"],
                "error": _state["error"] or _state["index_error"],
            })
        return super().do_GET()

    def _read_json(self):
        """Body -> dict, or (None, reason). Never lets a bad body escape as an exception."""
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return None, (400, "bad content-length")
        if n > MAX_BODY:
            return None, (413, "request too large")
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:                                     # noqa: BLE001
            return None, (400, "bad json")
        if not isinstance(req, dict):
            return None, (400, "body must be a JSON object")
        return req, None

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/quran":
            return self._send(404, {"error": "not found"})
        req, err = self._read_json()
        if err:
            return self._send(err[0], {"error": err[1]})
        if _state["status"] != "ready" or _state["index"] != "ready":
            return self._send(503, {"error": "not ready", "model": _state["status"], "index": _state["index"]})

        text = req.get("text")
        if not isinstance(text, str):
            return self._send(400, {"error": "text must be a string", "message": "اكتب سؤالاً أولاً"})
        text = text.strip()

        try:
            import quran_router as qr
            msg = qr.precheck(text)
            if msg:
                return self._send(400, {"error": "rejected", "message": msg})

            t0 = time.perf_counter()
            res = ask(text, qr.QUESTION)
            a = res["answers"]["intent"]
            if "error" in a or "answer" not in a:            # e.g. options truncated
                return self._send(400, {"error": "model", "message": "ما قدرت أفهم السؤال — جرّب صياغة أقصر"})

            intent, override = qr.refine(a["answer"], text, a["confidence"])
            payload, kind = qr.answer(intent, text)
            total = round((time.perf_counter() - t0) * 1000, 1)
            print(f"[q] intent={intent} model={a['answer']} conf={a['confidence']:.2f} "
                  f"override={'y' if override else 'n'} kind={kind} model_ms={res['ms']} total_ms={total}",
                  flush=True)
            return self._send(200, {
                "intent": intent,
                "model_intent": a["answer"],
                "override": override,
                "confidence": a["confidence"],
                "probabilities": a["probabilities"],
                "model_ms": res["ms"],
                "total_ms": total,
                "kind": kind,
                "result": payload,
            })
        except Exception as exc:                              # noqa: BLE001
            print(f"[q] error {type(exc).__name__}: {exc}", flush=True)
            return self._send(500, {"error": "internal", "message": "صار خطأ داخلي — راجع الطرفية"})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    threading.Thread(target=load_model, daemon=True).start()
    threading.Thread(target=load_quran, daemon=True).start()
    print(f"\n  quran-jeb  ->  http://{HOST}:{PORT}\n  (model loads in the background; needs ~2.5 GB RAM)\n", flush=True)
    with Server((HOST, PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")

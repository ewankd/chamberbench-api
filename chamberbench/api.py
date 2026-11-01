"""Small WSGI application. Every public endpoint returns JSON."""

import json
import re
from http import HTTPStatus

from .experiment import Config, Sample, score


ITEM = re.compile(r"/experiments/([0-9a-f]{24})(/score)?")
MAX_BODY = 8192
ROUTES = {
    "GET /health": "Check whether the service is running.",
    "POST /experiments": "Run or reuse an experiment. An empty JSON object uses defaults.",
    "GET /experiments/{id}": "Retrieve settings, fitted models, and held-out metrics.",
    "POST /experiments/{id}/score": "Score one setpoint and reading using both stored models.",
}


class RequestError(Exception):
    def __init__(self, status, message, headers=()):
        self.status = status
        self.message = message
        self.headers = headers


def reject_constant(token):
    raise ValueError(f"{token} is not a JSON number.")


def read_json(env):
    media_type = env.get("CONTENT_TYPE", "").split(";", 1)[0].lower().strip()
    if media_type != "application/json":
        raise RequestError(415, "Send Content-Type: application/json.")
    if not env.get("CONTENT_LENGTH"):
        raise RequestError(411, "Send a Content-Length header.")
    try:
        length = int(env["CONTENT_LENGTH"])
    except ValueError:
        raise RequestError(400, "Invalid Content-Length.") from None
    if length < 0:
        raise RequestError(400, "Invalid Content-Length.")
    if length > MAX_BODY:
        raise RequestError(413, f"JSON bodies are limited to {MAX_BODY} bytes.")
    raw = env["wsgi.input"].read(length)
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    except (ValueError, RecursionError):
        raise RequestError(400, "Expected a valid UTF-8 JSON body.") from None


def require_method(env, method):
    if env["REQUEST_METHOD"] != method:
        raise RequestError(405, f"Use {method} for this endpoint.", [("Allow", method)])


def dispatch(store, env):
    path = env["PATH_INFO"]
    if path in ("/", "/health"):
        require_method(env, "GET")
        return 200, ROUTES if path == "/" else {"status": "ok"}, []
    if path == "/experiments":
        require_method(env, "POST")
        config = Config.parse(read_json(env))
        result, created = store.get_or_run(config)
        return 201 if created else 200, result, [("Location", "/experiments/" + result["id"])]
    match = ITEM.fullmatch(path)
    if not match:
        raise RequestError(404, "Endpoint not found. GET / lists the routes.")
    experiment_id, score_path = match.groups()
    require_method(env, "POST" if score_path else "GET")
    result = store.get(experiment_id)
    if result is None:
        raise RequestError(404, "Experiment not found.")
    if score_path:
        sample = Sample.parse(read_json(env))
        return 200, {"experiment_id": experiment_id, "detectors": score(result, sample)}, []
    return 200, result, []


def application(store):
    def handle(env, start_response):
        try:
            status, data, headers = dispatch(store, env)
        except RequestError as exc:
            status, data, headers = exc.status, {"error": exc.message}, list(exc.headers)
        except ValueError as exc:
            status, data, headers = 422, {"error": str(exc)}, []
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        headers.extend([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        start_response(f"{status} {HTTPStatus(status).phrase}", headers)
        return [body]

    return handle

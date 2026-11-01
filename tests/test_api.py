import concurrent.futures
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import URLError
from wsgiref.util import setup_testing_defaults
from wsgiref.simple_server import WSGIRequestHandler, make_server

from chamberbench.api import application
from chamberbench import demo
from chamberbench.experiment import Config
from chamberbench.store import Store


class QuietHandler(WSGIRequestHandler):
    def log_message(self, fmt, *args):
        pass


class ApiTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "test.sqlite3"
        self.app = application(Store(self.path))

    def request(self, method, path, body=None, content_type="application/json"):
        payload = (body or "").encode()
        env = {}
        setup_testing_defaults(env)
        env.update(REQUEST_METHOD=method, PATH_INFO=path, CONTENT_TYPE=content_type,
                   CONTENT_LENGTH=str(len(payload)))
        env["wsgi.input"] = io.BytesIO(payload)
        response = {}

        def start_response(status, headers):
            response.update(status=int(status.split()[0]), headers=dict(headers))

        data = b"".join(self.app(env, start_response))
        self.assertEqual(int(response["headers"]["Content-Length"]), len(data))
        return response["status"], response["headers"], json.loads(data)

    def test_create_replay_restart_lookup_and_inference(self):
        status, headers, created = self.request("POST", "/experiments", "{}")
        self.assertEqual(status, 201)
        location = headers["Location"]
        self.assertEqual(location, "/experiments/" + created["id"])
        code, _, repeat = self.request("POST", "/experiments", '{"seed":42}')
        self.assertEqual((code, repeat), (200, created))
        self.app = application(Store(self.path))
        code, _, retrieved = self.request("GET", location)
        self.assertEqual((code, retrieved), (200, created))
        sample = {
            "setpoint": {"pressure_mtorr": 55, "rf_power_w": 785, "temperature_c": 94},
            "reading": {"pressure_mtorr": 55.2, "rf_power_w": 786, "temperature_c": 94.4},
        }
        code, _, scored = self.request("POST", location + "/score", json.dumps(sample))
        self.assertEqual(code, 200)
        self.assertEqual(scored["experiment_id"], created["id"])
        self.assertTrue(scored["detectors"]["raw"]["flagged"])
        self.assertFalse(scored["detectors"]["recipe_aware"]["flagged"])
        sample["reading"]["temperature_c"] = 105
        _, _, scored = self.request("POST", location + "/score", json.dumps(sample))
        self.assertTrue(scored["detectors"]["recipe_aware"]["flagged"])
        self.assertEqual(scored["detectors"]["recipe_aware"]["top_sensor"], "temperature_c")

    def test_error_responses_and_health(self):
        code, _, health = self.request("GET", "/health")
        self.assertEqual((code, health), (200, {"status": "ok"}))
        code, headers, _ = self.request("GET", "/experiments")
        self.assertEqual((code, headers["Allow"]), (405, "POST"))
        missing = "/experiments/" + "0" * 24
        self.assertEqual(self.request("GET", missing)[0], 404)
        self.assertEqual(self.request("GET", "/unknown")[0], 404)
        self.assertEqual(self.request("POST", "/experiments", "{}", "text/plain")[0], 415)
        self.assertEqual(self.request("POST", "/experiments", " " * 8193)[0], 413)
        for body in ("", "{", '{"target_fpr": NaN}'):
            self.assertEqual(self.request("POST", "/experiments", body)[0], 400)
        for payload in ([], {"seed": True}, {"seed": -1}, {"test": 1}, {"target_fpr": 0},
                        {"fault_strength": "3"}, {"fault_strength": 9}, {"seed": 1.0}):
            with self.subTest(payload=payload):
                code, _, data = self.request("POST", "/experiments", json.dumps(payload))
                self.assertEqual(code, 422)
                self.assertIn("error", data)

    def test_score_validation(self):
        _, headers, _ = self.request("POST", "/experiments", "{}")
        url = headers["Location"] + "/score"
        sensors = {"pressure_mtorr": 30, "rf_power_w": 500, "temperature_c": 65}
        for reading in ({}, dict(sensors, foo=1), dict(sensors, pressure_mtorr=-1),
                        dict(sensors, rf_power_w=True), dict(sensors, temperature_c=5000)):
            with self.subTest(reading=reading):
                data = {"setpoint": sensors, "reading": reading}
                self.assertEqual(self.request("POST", url, json.dumps(data))[0], 422)


class PersistenceTests(unittest.TestCase):
    def test_simultaneous_duplicate_runs_commit_one_result(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "concurrent.sqlite3")
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                calls = list(pool.map(store.get_or_run, [Config()] * 4))
            self.assertEqual(sum(created for _, created in calls), 1)
            saved = store.get(Config().experiment_id())
            for result, _ in calls:
                self.assertEqual(json.loads(json.dumps(result)), saved)


class NetworkTests(unittest.TestCase):
    def test_live_demo_over_http(self):
        with tempfile.TemporaryDirectory() as folder:
            app = application(Store(Path(folder) / "live.sqlite3"))
            with make_server("127.0.0.1", 0, app, handler_class=QuietHandler) as server:
                thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
                thread.start()
                base = f"http://127.0.0.1:{server.server_port}"
                try:
                    try:
                        _, health = demo.call(base, "/health")
                    except URLError as exc:
                        if isinstance(exc.reason, PermissionError) and not os.environ.get("CI"):
                            self.skipTest("Local socket connections are blocked in this environment.")
                        raise
                    self.assertEqual(health["status"], "ok")
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        demo.main(["--base-url", base])
                        demo.main(["--base-url", base])
                    self.assertIn("HTTP 201", output.getvalue())
                    self.assertIn("HTTP 200", output.getvalue())
                    self.assertIn("near-target reading", output.getvalue())
                finally:
                    server.shutdown()
                    thread.join()

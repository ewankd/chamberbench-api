"""Run the local development API with python -m chamberbench."""

import argparse
from wsgiref.simple_server import make_server

from .api import application
from .store import Store


def main():
    parser = argparse.ArgumentParser(description="Run the ChamberBench experiment API locally.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default="chamberbench.sqlite3", help="Path for saved experiments.")
    args = parser.parse_args()
    app = application(Store(args.db))
    with make_server("127.0.0.1", args.port, app) as server:
        print(f"ChamberBench is listening at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

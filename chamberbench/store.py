"""Atomic persistence of complete experiment results and their fitted models."""

import json
import sqlite3
from contextlib import closing

from .experiment import run


class Store:
    def __init__(self, path):
        self.path = path
        with closing(self.connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    result TEXT NOT NULL
                )
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def get(self, experiment_id):
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT result FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_or_run(self, config):
        experiment_id = config.experiment_id()
        existing = self.get(experiment_id)
        if existing is not None:
            return existing, False
        result = run(config)
        # A second request may have completed while this one was computing.
        # The first insert owns the result; duplicate requests read that same row.
        with closing(self.connect()) as db, db:
            inserted = db.execute(
                "INSERT INTO experiments (id, result) VALUES (?, ?) ON CONFLICT(id) DO NOTHING",
                (experiment_id, json.dumps(result, allow_nan=False)),
            ).rowcount == 1
            if not inserted:
                row = db.execute("SELECT result FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
                result = json.loads(row[0])
        return result, inserted

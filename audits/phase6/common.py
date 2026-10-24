from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPRO = PROJECT
PHASE4 = PROJECT / "docs" / "review" / "phase4"
EVIDENCE = HERE / "evidence"
RESULTS = HERE / "results"
LOGS = HERE / "logs"

for directory in (EVIDENCE, RESULTS, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPRO / "src"))
sys.path.insert(0, str(REPRO / "vendor"))


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path):
    return {
        "path": str(path.relative_to(PROJECT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

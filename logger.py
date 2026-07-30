import json
import os
from datetime import datetime, timezone
from pathlib import Path


class JSONLLogger:
    def __init__(self, filepath: str = "run.jsonl"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(self, data: dict):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **data}
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_all_raw(self) -> str:
        if not self.filepath.exists():
            return ""
        return self.filepath.read_text(encoding="utf-8")

    def clear(self):
        if self.filepath.exists():
            self.filepath.unlink()

"""Prediction events and delayed labels, appended as JSON lines.

P1 reads predictions.jsonl for feature drift; P2 joins labels.jsonl on
(request_id, instance_index) for labeled quality. The in-memory index is
rebuilt from the files on startup so feedback still works after a restart.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class FeedbackError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_lines(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written last line after a crash


class EventStore:
    def __init__(self, directory: Path | None):
        self.directory = directory
        self._lock = threading.Lock()
        self._requests: dict[str, dict] = {}
        self._labels: dict[tuple[str, int], int] = {}
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        self._predictions_path = directory / "predictions.jsonl"
        self._labels_path = directory / "labels.jsonl"
        for event in _read_lines(self._predictions_path):
            entry = self._requests.setdefault(
                event["request_id"], {"count": 0, "model_version": event["model_version"]}
            )
            entry["count"] = max(entry["count"], event["instance_index"] + 1)
        for event in _read_lines(self._labels_path):
            self._labels[(event["request_id"], event["instance_index"])] = event["label"]

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def _append(self, path: Path, records: list[dict]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()

    def record_predictions(self, request_id, model, instances, predictions) -> None:
        if not self.enabled:
            return
        predicted_at = _now()
        records = [
            {
                "contract_version": 1,
                "request_id": request_id,
                "instance_index": index,
                "predicted_at": predicted_at,
                "model_version": model.model_version,
                "schema_version": model.schema_version,
                "features": instance,
                "label": prediction["label"],
                "default_probability": prediction["default_probability"],
            }
            for index, (instance, prediction) in enumerate(zip(instances, predictions, strict=True))
        ]
        with self._lock:
            self._append(self._predictions_path, records)
            self._requests[request_id] = {
                "count": len(instances),
                "model_version": model.model_version,
            }

    def record_feedback(self, body) -> int:
        if not self.enabled:
            raise FeedbackError(503, "feedback_disabled", "This server does not keep events.")
        if not isinstance(body, dict) or set(body) - {"request_id", "labels", "observed_at"}:
            raise FeedbackError(
                422, "schema_violation", "Expected request_id, labels, observed_at."
            )
        request_id, labels, observed_at = (
            body.get("request_id"),
            body.get("labels"),
            body.get("observed_at"),
        )
        if not isinstance(request_id, str) or not request_id:
            raise FeedbackError(422, "schema_violation", "request_id must be a string.")
        if not isinstance(labels, list) or not labels:
            raise FeedbackError(422, "schema_violation", "labels must be a nonempty list.")
        try:
            observed = datetime.fromisoformat(observed_at)
        except (TypeError, ValueError):
            raise FeedbackError(422, "schema_violation", "observed_at must be ISO 8601.") from None
        if observed.tzinfo is None:
            raise FeedbackError(422, "schema_violation", "observed_at needs a timezone.")

        with self._lock:
            known = self._requests.get(request_id)
            if known is None:
                raise FeedbackError(404, "unknown_request", "No retained prediction has this ID.")
            pending: dict[int, int] = {}
            for item in labels:
                if not isinstance(item, dict) or set(item) != {"instance_index", "label"}:
                    raise FeedbackError(
                        422, "schema_violation", "Each label needs index and label."
                    )
                index, label = item["instance_index"], item["label"]
                if type(index) is not int or not 0 <= index < known["count"]:
                    raise FeedbackError(422, "invalid_index", "instance_index is out of range.")
                if type(label) is not int or label not in (0, 1):
                    raise FeedbackError(422, "invalid_label", "label must be integer 0 or 1.")
                previous = self._labels.get((request_id, index), pending.get(index))
                if previous is not None and previous != label:
                    raise FeedbackError(409, "label_conflict", "A different label already exists.")
                if previous is None:
                    pending[index] = label
            received_at = _now()
            new = [
                {
                    "contract_version": 1,
                    "request_id": request_id,
                    "instance_index": index,
                    "label": label,
                    "model_version": known["model_version"],
                    "observed_at": observed_at,
                    "received_at": received_at,
                }
                for index, label in pending.items()
            ]
            if new:
                self._append(self._labels_path, new)
                for record in new:
                    self._labels[(request_id, record["instance_index"])] = record["label"]
            return len(new)

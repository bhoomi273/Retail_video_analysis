"""Shared drawing, video, and CSV helpers."""

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

def load_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def draw_line(frame, start: Sequence[int], end: Sequence[int], color=(0, 255, 255), thickness=2):
    import cv2

    return cv2.line(frame, tuple(start), tuple(end), color, thickness)


def draw_polygon(frame, polygon: Sequence[Sequence[int]], color=(255, 180, 0), thickness=2):
    import cv2

    points = __import__("numpy").array(polygon, dtype="int32")
    return cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)


def write_csv(path: str | Path, rows: Iterable[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if not rows and fieldnames is None:
        return
    fields = list(fieldnames or rows[0].keys())
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

"""
process_interior.py

Runs Task 2 - Per-Shelf Customer Interest on interior.mp4.

Outputs:
    outputs/interior_annotated.mp4
    outputs/task2_results.csv

Usage:
    python src/process_interior.py --video data/interior.mp4
"""

import argparse
import os
import csv
import yaml
import cv2
import numpy as np

from tracker import PersonTracker
from shelf_task2_logic import ShelfInterestLogic

CONFIG_DIR = "configs"
OUTPUT_DIR = "outputs"

SHELF_COLORS = {
    "Shelf A": (0, 0, 255),
    "Shelf B": (0, 255, 0),
    "Shelf C": (255, 0, 0),
    "Shelf D": (0, 255, 255),
}


def draw_shelf_summary(frame, counts):
    """Draw a clear summary panel listing each shelf's interaction count."""
    shelf_names = list(counts.keys())
    if not shelf_names:
        return

    panel_x, panel_y = 20, 20
    margin = 12
    line_h = 26
    title = "Shelf interactions"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    max_width = title_size[0]
    lines = []
    for name in shelf_names:
        text = f"{name}: {counts.get(name, 0)}"
        lines.append(text)
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        max_width = max(max_width, width)

    panel_w = max_width + margin * 2
    panel_h = title_size[1] + 10 + len(lines) * line_h + margin * 2
    panel = frame.copy()
    cv2.rectangle(panel, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, title, (panel_x + margin, panel_y + margin + title_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for idx, name in enumerate(shelf_names):
        value = counts.get(name, 0)
        color = SHELF_COLORS.get(name, (255, 255, 255))
        y = panel_y + margin + title_size[1] + 18 + idx * line_h
        cv2.putText(frame, f"{name}: {value}", (panel_x + margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def load_interior_config():
    with open(os.path.join(CONFIG_DIR, "interior_config.yaml"), "r") as f:
        return yaml.safe_load(f)


def run(video_path, frame_skip=2):
    config = load_interior_config()
    shelves = {name: data["polygon"] for name, data in config["shelves"].items()}
    exclusion_zones = config.get("exclusion_zones", [])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = PersonTracker(fps=fps, exclusion_zones=exclusion_zones)    
    shelf_logic = ShelfInterestLogic(shelves, fps=fps)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "interior_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_index = 0
    last_tracks = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_skip == 0:
            last_tracks = tracker.update(frame, frame_index)
            shelf_logic.update(last_tracks, frame_index)
        tracks = last_tracks

        # draw shelf zones (semi-transparent) + cumulative counts
        overlay = frame.copy()
        counts = shelf_logic.get_counts()
        for name, poly in shelves.items():
            pts = np.array(poly, dtype="int32")
            color = SHELF_COLORS.get(name, (255, 255, 255))
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # always show the interaction count summary in a dedicated panel
        draw_shelf_summary(frame, counts)

        for name, poly in shelves.items():
            pts = np.array(poly, dtype="int32")
            color = SHELF_COLORS.get(name, (255, 255, 255))
            cv2.polylines(frame, [pts], True, color, 2)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(frame, f"{name}: {counts.get(name, 0)}", (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw people + their active shelf association/duration
        for t in tracks:
            tid = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            shelf_name, duration = shelf_logic.get_active_info(tid, frame_index)

            if shelf_name is not None:
                color = SHELF_COLORS.get(shelf_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID{tid} -> {shelf_name} ({duration:.1f}s)",
                            (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # visual association line from person to shelf centroid
                cx = int(sum(p[0] for p in shelves[shelf_name]) / len(shelves[shelf_name]))
                cy = int(sum(p[1] for p in shelves[shelf_name]) / len(shelves[shelf_name]))
                foot = (int((x1 + x2) / 2), int(y2))
                cv2.line(frame, foot, (cx, cy), color, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
                cv2.putText(frame, f"ID{tid}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        writer.write(frame)
        frame_index += 1
        if frame_index % 50 == 0:
            print(f"  processed {frame_index} frames...")

    cap.release()
    writer.release()

    final_counts = shelf_logic.get_counts()
    with open(os.path.join(OUTPUT_DIR, "task2_results.csv"), "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Shelf", "Interest Events"])
        for name, count in final_counts.items():
            writer_csv.writerow([name, count])

    print(f"\nDone. {frame_index} frames processed.")
    print(f"Task 2 final counts: {final_counts}")
    print(f"Saved video -> {out_path}")
    print(f"Saved CSV -> outputs/task2_results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--skip", type=int, default=2,
                         help="Process 1 out of every N frames (default 2) for speed.")
    args = parser.parse_args()
    run(args.video, frame_skip=args.skip)


if __name__ == "__main__":
    main()
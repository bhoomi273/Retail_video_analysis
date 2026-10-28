"""
process_entrance.py

The ONE script that produces the shared annotated video for Task 1 +
Task 3 (per assessment requirement: no separate video for Task 3).

Pipeline per frame:
    detect + track people
      -> identify which track is the seeded staff member
      -> feed customer tracks into Task 1 logic (interest/entry/passed-by)
      -> feed staff position + customer tracks into Task 3 logic (sessions)
      -> draw everything onto the same frame
      -> write frame to ONE output video

Outputs:
    outputs/entrance_annotated.mp4
    outputs/task1_results.csv
    outputs/task3_results.csv

Usage:
    python src/process_entrance.py --video data/entrance.mp4
"""

import argparse
import os
import csv
import yaml
import cv2

from tracker import PersonTracker
from entrance_task1_logic import EntranceInterestLogic, UNKNOWN, INTERESTED, ENTERED, PASSED_BY
from entrance_task3_logic import StaffCustomerInteractionLogic

CONFIG_DIR = "configs"
OUTPUT_DIR = "outputs"

STATE_COLORS = {
    UNKNOWN: (200, 200, 200),
    INTERESTED: (0, 255, 255),
    ENTERED: (0, 255, 0),
    PASSED_BY: (0, 0, 255),
}


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_entrance_config():
    with open(os.path.join(CONFIG_DIR, "entrance_config.yaml"), "r") as f:
        return yaml.safe_load(f)


def run(video_path, frame_skip=2):
    config = load_entrance_config()
    line_p1 = tuple(config["entrance_line"]["point_1"])
    line_p2 = tuple(config["entrance_line"]["point_2"])
    seed_box = config["staff_seed_box"]
    seed_bbox = (seed_box["x1"], seed_box["y1"], seed_box["x2"], seed_box["y2"])
    staff_id = seed_box.get("staff_id", "staff_1")
    seed_center = ((seed_bbox[0] + seed_bbox[2]) / 2, (seed_bbox[1] + seed_bbox[3]) / 2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = PersonTracker(fps=fps)
    task1 = EntranceInterestLogic(line_p1, line_p2, fps, known_store_point=seed_center)
    task3 = StaffCustomerInteractionLogic(staff_ids=[staff_id])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "entrance_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    staff_track_id = None  # resolved once we see a track overlapping the seed box
    frame_index = 0
    last_tracks = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Only run detect+track every `frame_skip` frames; reuse last result
        # in between for near real-time speed. Task logic still receives the
        # true frame_index each call, so all timing math stays correct.
        if frame_index % frame_skip == 0:
            last_tracks = tracker.update(frame, frame_index)
        tracks = last_tracks

        # Resolve which track_id is the staff member (best IoU match to seed box,
        # checked continuously in case detection is delayed by a few frames)
        if staff_track_id is None:
            best_iou, best_id = 0.0, None
            for t in tracks:
                iou = _iou(t["bbox"], seed_bbox)
                if iou > best_iou:
                    best_iou, best_id = iou, t["track_id"]
            if best_iou > 0.3:
                staff_track_id = best_id

        staff_track = None
        customer_tracks = []
        for t in tracks:
            if staff_track_id is not None and t["track_id"] == staff_track_id:
                staff_track = t
            else:
                customer_tracks.append(t)

        # --- Task 1 / Task 3 updates only run on frames with FRESH detections.
        # Re-running them on reused (stale) positions would make speed look
        # like zero every skipped frame, which could wrongly trigger the
        # "slowed down" interest signal. Drawing below still happens every
        # frame using the latest known states for a smooth-looking video.
        if frame_index % frame_skip == 0:
            task1.update(customer_tracks, frame_index)
            staff_positions = {}
            if staff_track is not None:
                x1, y1, x2, y2 = staff_track["bbox"]
                staff_positions[staff_id] = ((x1 + x2) / 2, y2)
            task3.update(staff_positions, customer_tracks)

        # --- Drawing ---
        cv2.line(frame, line_p1, line_p2, (255, 0, 0), 2)

        for t in customer_tracks:
            tid = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            state = task1.get_track_state(tid)
            color = STATE_COLORS.get(state, (200, 200, 200))

            # highlight if currently in an active staff interaction
            if tid in task3.active_customer_to_staff:
                color = (255, 0, 255)
                cv2.putText(frame, "INTERACTING", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # draw a connecting line to the staff member, same visual
                # pattern as the shelf-association lines in Task 2
                if staff_track is not None:
                    sx1, sy1, sx2, sy2 = staff_track["bbox"]
                    staff_point = (int((sx1 + sx2) / 2), int(sy2))
                    customer_point = (int((x1 + x2) / 2), int(y2))
                    cv2.line(frame, customer_point, staff_point, (255, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID{tid} {state}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if staff_track is not None:
            sx1, sy1, sx2, sy2 = staff_track["bbox"]
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 255, 0), 3)
            cv2.putText(frame, f"STAFF ({staff_id})", (sx1, max(0, sy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Overlay text: Task 1 + Task 3 running results, top-left ---
        counts = task1.get_counts()
        t3 = task3.get_results()
        lines = [
            f"Total Interested: {counts['Total Interested']}",
            f"Interested Entered: {counts['Interested Entered']}",
            f"Interested Passed By: {counts['Interested Passed By']}",
            "",
            f"Staff sessions ({staff_id}): {t3['per_staff'].get(staff_id, 0)}",
            f"Avg sessions/staff: {t3['average']:.2f}",
        ]
        y = 25
        for line in lines:
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 24

        writer.write(frame)
        frame_index += 1
        if frame_index % 50 == 0:
            print(f"  processed {frame_index} frames...")

    cap.release()
    writer.release()
    task1.finalize_all()

    # --- Write CSVs ---
    final_counts = task1.get_counts()
    with open(os.path.join(OUTPUT_DIR, "task1_results.csv"), "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Metric", "Count"])
        for k, v in final_counts.items():
            writer_csv.writerow([k, v])

    final_t3 = task3.get_results()
    with open(os.path.join(OUTPUT_DIR, "task3_results.csv"), "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Staff Instance", "Interaction Sessions"])
        for sid, count in final_t3["per_staff"].items():
            writer_csv.writerow([sid, count])
        writer_csv.writerow(["Average sessions per staff", round(final_t3["average"], 2)])

    print(f"\nDone. {frame_index} frames processed.")
    print(f"Task 1 final counts: {final_counts}")
    print(f"Task 3 final results: {final_t3}")
    print(f"Saved video -> {out_path}")
    print(f"Saved CSVs -> outputs/task1_results.csv, outputs/task3_results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--skip", type=int, default=2,
                         help="Process 1 out of every N frames (default 2) for speed.")
    args = parser.parse_args()
    run(args.video, frame_skip=args.skip)


if __name__ == "__main__":
    main()
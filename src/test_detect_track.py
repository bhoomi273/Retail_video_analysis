"""
test_detect_track.py

Standalone sanity check -- runs ONLY detection + tracking (no task logic
yet) on a video and writes an annotated output so we can visually judge:
  - Is YOLO-nano catching people reliably in this footage?
  - Does the same person keep the same track ID over time, or does the
    ID keep changing (a sign we may need to tune conf_threshold, the
    tracker config, or the occlusion grace window)?

Usage:
    python src/test_detect_track.py --video data/entrance.mp4
    python src/test_detect_track.py --video data/interior.mp4
"""

import argparse
import os
import cv2
from tracker import PersonTracker

OUTPUT_DIR = "outputs"


def run(video_path, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = PersonTracker(fps=fps)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"test_track_{basename}.mp4")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_index = 0
    max_id_seen = 0
    last_tracks = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Only run detection+tracking every `frame_skip` frames -- reuse the
        # last computed boxes in between. This cuts compute roughly
        # `frame_skip`x for near real-time throughput. Position history used
        # by task logic is keyed on the real frame_index either way, so
        # speed/duration math stays correct regardless of the skip factor.
        if frame_index % frame_skip == 0:
            last_tracks = tracker.update(frame, frame_index)
        tracks = last_tracks

        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            tid = t["track_id"]
            max_id_seen = max(max_id_seen, tid)
            # Show all tracks with the same positive/confirmed color and
            # omit the textual "(ghost)" marker so IDs remain visually
            # continuous even when the tracker is momentarily coasting.
            # Note: the tracker still sets t["is_ghost"] internally; this only
            # changes the visualization to hide that state.
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Frame {frame_index}  |  Max track ID so far: {max_id_seen}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        frame_index += 1

        if frame_index % 50 == 0:
            print(f"  processed {frame_index} frames...")

    cap.release()
    writer.release()
    print(f"\nDone. {frame_index} frames processed.")
    print(f"Max track ID assigned during the video: {max_id_seen}")
    print(f"Saved annotated video -> {out_path}")
    # print("\nWatch this video and check:")
    # print("  - Does each real person keep ONE consistent ID the whole time they're visible?")
    # print("  - Do IDs jump unexpectedly high (suggests IDs are being reassigned too often)?")
    # print("  - Orange boxes = 'ghost' frames where detection was missed but track was kept alive.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--skip", type=int, default=1,
                         help="Process 1 out of every N frames (default 2) for speed.")
    args = parser.parse_args()
    run(args.video, frame_skip=args.skip)


if __name__ == "__main__":
    main()
"""
verify_config.py

Draws the saved entrance/interior configs onto their reference frame
and saves an image, so you can visually confirm the line/boxes/polygons
were placed correctly before building detection logic on top of them.

Usage:
    python src/verify_config.py --mode entrance
    python src/verify_config.py --mode interior
"""

import argparse
import os
import cv2
import yaml

CONFIG_DIR = "configs"
OUTPUT_DIR = "outputs"

SHELF_COLORS = {
    "Shelf A": (0, 0, 255),      # red
    "Shelf B": (0, 255, 0),      # green
    "Shelf C": (255, 0, 0),      # blue
    "Shelf D": (0, 255, 255),    # yellow
}


def load_config(mode):
    path = os.path.join(CONFIG_DIR, f"{mode}_config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def verify_entrance():
    config = load_config("entrance")
    frame = get_frame(config["video"], config["reference_frame_index"])

    # entrance line
    p1 = tuple(config["entrance_line"]["point_1"])
    p2 = tuple(config["entrance_line"]["point_2"])
    cv2.line(frame, p1, p2, (0, 0, 255), 3)
    cv2.putText(frame, "ENTRANCE LINE", (p1[0], p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # staff seed box
    box = config["staff_seed_box"]
    cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (255, 0, 255), 3)
    cv2.putText(frame, box.get("staff_id", "staff"), (box["x1"], box["y1"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    out_path = os.path.join(OUTPUT_DIR, "verify_entrance_config.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"Saved -> {out_path}")


def verify_interior():
    config = load_config("interior")
    frame = get_frame(config["video"], config["reference_frame_index"])

    for shelf_name, shelf_data in config["shelves"].items():
        pts = shelf_data["polygon"]
        color = SHELF_COLORS.get(shelf_name, (255, 255, 255))

        # draw filled semi-transparent overlay so overlaps are visible
        overlay = frame.copy()
        pts_arr = [tuple(p) for p in pts]
        cv2.fillPoly(overlay, [_to_np_array(pts_arr)], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # draw outline + numbered corners so click order is visible
        for i in range(len(pts_arr)):
            cv2.line(frame, pts_arr[i], pts_arr[(i + 1) % len(pts_arr)], color, 2)
        for i, pt in enumerate(pts_arr):
            cv2.circle(frame, pt, 4, color, -1)
            cv2.putText(frame, str(i + 1), (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # label at centroid
        cx = sum(p[0] for p in pts_arr) // len(pts_arr)
        cy = sum(p[1] for p in pts_arr) // len(pts_arr)
        cv2.putText(frame, shelf_name, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = os.path.join(OUTPUT_DIR, "verify_interior_config.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"Saved -> {out_path}")


def _to_np_array(points):
    import numpy as np
    return np.array(points, dtype="int32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["entrance", "interior"])
    args = parser.parse_args()

    if args.mode == "entrance":
        verify_entrance()
    else:
        verify_interior()


if __name__ == "__main__":
    main()
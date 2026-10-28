"""
setup_zones.py

One-time manual calibration tool for a FIXED CCTV camera.

Since the camera does not move between frames, we only need to define
scene geometry (entrance line, staff seed box, shelf zones) ONCE, by
clicking points on a single reference frame. The saved coordinates are
then reused by the main pipeline for every frame of the video.

Usage:
    python src/setup_zones.py --video data/entrance.mp4 --mode entrance
    python src/setup_zones.py --video data/interior.mp4 --mode interior

Controls (for every shape you draw):
    - Left click   -> add a point
    - 'u'          -> undo last point
    - 'n'          -> confirm current shape and move to the next one
    - 'r'          -> restart the current shape from scratch
    - 'q'          -> quit without saving remaining shapes

Output:
    configs/entrance_config.yaml   (entrance line + staff seed box)
    configs/interior_config.yaml   (4 shelf polygons: A, B, C, D)
"""

import argparse
import os
import cv2
import yaml

CONFIG_DIR = "configs"


def get_reference_frame(video_path, frame_index=0):
    """Grab a single frame from the video to use as the calibration reference."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read frame {frame_index} from {video_path}. "
            f"Try a different --frame value."
        )
    return frame


class ShapeDrawer:
    """
    Interactive point collector for one shape (line, box, or polygon)
    on a static reference image.
    """

    def __init__(self, window_name, base_image, num_points, shape_label):
        self.window_name = window_name
        self.base_image = base_image
        self.num_points = num_points  # exact number of points expected
        self.shape_label = shape_label
        self.points = []

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.num_points:
            self.points.append((x, y))

    def _draw_preview(self):
        img = self.base_image.copy()
        cv2.putText(
            img,
            f"Drawing: {self.shape_label} "
            f"({len(self.points)}/{self.num_points} points) "
            f"[n=confirm  u=undo  r=restart  q=quit]",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        for i, pt in enumerate(self.points):
            cv2.circle(img, pt, 5, (0, 0, 255), -1)
            cv2.putText(
                img, str(i + 1), (pt[0] + 8, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(img, self.points[i], self.points[i + 1], (0, 255, 0), 2)
            # close the polygon visually once all points are placed
            if len(self.points) == self.num_points and self.num_points > 2:
                cv2.line(img, self.points[-1], self.points[0], (0, 255, 0), 2)
        return img

    def collect(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            preview = self._draw_preview()
            cv2.imshow(self.window_name, preview)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("u") and self.points:
                self.points.pop()
            elif key == ord("r"):
                self.points = []
            elif key == ord("n"):
                if len(self.points) == self.num_points:
                    return self.points
                else:
                    print(
                        f"  Need exactly {self.num_points} points before confirming "
                        f"(have {len(self.points)})."
                    )
            elif key == ord("q"):
                return None


def save_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Saved -> {path}")


def run_entrance_setup(video_path, frame_index):
    frame = get_reference_frame(video_path, frame_index)
    config = {"reference_frame_index": frame_index, "video": video_path}

    print("\nStep A: Draw the ENTRANCE LINE (2 points: left end, right end).")
    print("Draw it where the mall floor transitions into the store floor.")
    drawer = ShapeDrawer("Setup - Entrance Line", frame, 2, "Entrance line")
    pts = drawer.collect()
    if pts is None:
        print("Aborted before saving.")
        return
    config["entrance_line"] = {"point_1": list(pts[0]), "point_2": list(pts[1])}

    print("\nStep B: Draw the STAFF SEED BOX (2 points: top-left, bottom-right)")
    print("around the staff member wearing the black apron.")
    drawer = ShapeDrawer("Setup - Staff Seed Box", frame, 2, "Staff seed box")
    pts = drawer.collect()
    if pts is None:
        print("Aborted before saving.")
        return
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    config["staff_seed_box"] = {
        "x1": min(x1, x2), "y1": min(y1, y2),
        "x2": max(x1, x2), "y2": max(y1, y2),
        "staff_id": "staff_1",
    }

    cv2.destroyAllWindows()
    save_yaml(os.path.join(CONFIG_DIR, "entrance_config.yaml"), config)


def run_interior_setup(video_path, frame_index):
    frame = get_reference_frame(video_path, frame_index)
    config = {"reference_frame_index": frame_index, "video": video_path, "shelves": {}}

    shelf_names = ["Shelf A", "Shelf B", "Shelf C", "Shelf D"]
    for shelf_name in shelf_names:
        print(f"\nDraw {shelf_name} as a 4-point polygon (click its 4 corners in order).")
        drawer = ShapeDrawer(f"Setup - {shelf_name}", frame, 4, shelf_name)
        pts = drawer.collect()
        if pts is None:
            print("Aborted before saving.")
            return
        config["shelves"][shelf_name] = {"polygon": [list(p) for p in pts]}

    cv2.destroyAllWindows()
    save_yaml(os.path.join(CONFIG_DIR, "interior_config.yaml"), config)


def main():
    parser = argparse.ArgumentParser(description="One-time scene calibration tool.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument(
        "--mode", required=True, choices=["entrance", "interior"],
        help="Which calibration flow to run."
    )
    parser.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to use as the reference image (default: 0)."
    )
    args = parser.parse_args()

    if args.mode == "entrance":
        run_entrance_setup(args.video, args.frame)
    else:
        run_interior_setup(args.video, args.frame)


if __name__ == "__main__":
    main()
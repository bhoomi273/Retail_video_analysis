"""
setup_exclusion_zone.py

Adds one or more EXCLUSION ZONES (e.g. a mirror that reflects people,
causing false "person" detections) to an existing config file, without
touching anything already saved in it (entrance line, staff box, shelf
zones stay untouched).

Any detection whose center falls inside an exclusion zone is dropped
BEFORE it ever reaches the tracker -- so a mirror reflection never even
gets a track ID, rather than being detected and then filtered out later.

Usage:
    python src/setup_exclusion_zone.py --video data/interior.mp4 --mode interior
    python src/setup_exclusion_zone.py --video data/entrance.mp4 --mode entrance
"""

import argparse
import os
import cv2
import yaml

from setup_zones import get_reference_frame, ShapeDrawer

CONFIG_DIR = "configs"


def run(video_path, mode, frame_index):
    config_path = os.path.join(CONFIG_DIR, f"{mode}_config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {"video": video_path, "reference_frame_index": frame_index}

    config.setdefault("exclusion_zones", [])

    frame = get_reference_frame(video_path, frame_index)

    while True:
        print(f"\nDraw an exclusion zone (e.g. a mirror) as a 4-point polygon.")
        print(f"Current exclusion zones saved: {len(config['exclusion_zones'])}")
        drawer = ShapeDrawer("Setup - Exclusion Zone", frame, 4, "Exclusion zone")
        pts = drawer.collect()
        if pts is None:
            break
        config["exclusion_zones"].append([list(p) for p in pts])

        again = input("Add another exclusion zone? (y/n): ").strip().lower()
        if again != "y":
            break

    cv2.destroyAllWindows()
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved -> {config_path} ({len(config['exclusion_zones'])} exclusion zone(s) total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--mode", required=True, choices=["entrance", "interior"])
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    run(args.video, args.mode, args.frame)


if __name__ == "__main__":
    main()
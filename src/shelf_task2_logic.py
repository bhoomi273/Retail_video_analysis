"""
shelf_task2_logic.py

State machine for Task 2 - Per-Shelf Customer Interest.

Per (person) state, tracking which shelf (if any) they're currently
engaged with:
    NOT_INTERACTING -> INTERACTING -> COOLDOWN -> NOT_INTERACTING
                                          |
                                 (return to same shelf before
                                  cooldown expires -> back to
                                  INTERACTING, NOT a new event)

Definitions and thresholds:

- ZONE MEMBERSHIP:
    A person's foot-point (bottom-center of bbox) is tested against each
    shelf's polygon (drawn once during setup). If it falls inside more
    than one shelf's polygon (the tight-gap case between adjacent
    shelves), the shelf whose centroid is closest to the point wins --
    a simple, consistent tie-breaker matching the brief's requirement to
    "determine which shelf the customer is interacting with based on
    available visual evidence".

- DWELL CONFIRMATION (INTERACTING trigger):
    A person must remain inside the same shelf's zone for
    `min_dwell_frames` consecutive frames before it counts as a real
    interest event (filters out someone just walking through the zone
    without stopping). The shelf's cumulative counter increments at the
    MOMENT this is confirmed -- not at the end -- since a dwell of this
    length has already definitely happened by that point.

- COOLDOWN / re-entry logic (avoiding double counting):
    When a person leaves the shelf zone, we don't immediately end the
    interaction -- we allow a short `cooldown_frames` grace window
    (e.g. bending down, stepping back to view shoes from a distance,
    minor tracking jitter at the zone edge). If they return to the SAME
    shelf within that window, it is treated as a continuation of the
    same episode (no new count). Only if the cooldown window fully
    expires without their return do we consider the episode over --
    a LATER return to that shelf after this point is a brand new event,
    per the assessment's explicit rule.
"""

from collections import defaultdict
import numpy as np
import cv2

DEFAULT_CONFIG = {
    "min_dwell_seconds": 1.5,     # sustained presence required to count as "interest"
    "cooldown_seconds": 1.5,      # grace window before a departure ends the episode
}

NOT_INTERACTING = "NOT_INTERACTING"
INTERACTING = "INTERACTING"
COOLDOWN = "COOLDOWN"


class ShelfInterestLogic:
    def __init__(self, shelves, fps, config=None):
        """
        shelves: dict shelf_name -> list of (x, y) polygon points
        """
        self.shelves = {name: np.array(pts, dtype="int32") for name, pts in shelves.items()}
        self.centroids = {
            name: (pts[:, 0].mean(), pts[:, 1].mean()) for name, pts in self.shelves.items()
        }
        self.fps = fps
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.min_dwell_frames = max(1, int(self.cfg["min_dwell_seconds"] * fps))
        self.cooldown_frames = max(1, int(self.cfg["cooldown_seconds"] * fps))

        self.shelf_counts = {name: 0 for name in shelves}

        # track_id -> state dict
        self._track_states = {}

    def _shelf_at_point(self, point):
        matches = [
            name for name, poly in self.shelves.items()
            if cv2.pointPolygonTest(poly, point, False) >= 0
        ]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # tie-break: nearest centroid
        return min(matches, key=lambda n: (self.centroids[n][0] - point[0]) ** 2
                                            + (self.centroids[n][1] - point[1]) ** 2)

    def update(self, customer_tracks, frame_index):
        seen_ids = set()

        for t in customer_tracks:
            tid = t["track_id"]
            seen_ids.add(tid)
            x1, y1, x2, y2 = t["bbox"]
            body_point = (float((x1 + x2) / 2), float((y1 + y2) / 2))

            shelf_here = self._shelf_at_point(body_point)

            state_info = self._track_states.setdefault(tid, {
                "state": NOT_INTERACTING, "shelf": None,
                "dwell_frames": 0, "cooldown_counter": 0,
                "interaction_start_frame": None,
            })

            state = state_info["state"]

            if state == NOT_INTERACTING:
                if shelf_here is not None:
                    if state_info["shelf"] == shelf_here:
                        state_info["dwell_frames"] += 1
                    else:
                        state_info["shelf"] = shelf_here
                        state_info["dwell_frames"] = 1

                    if state_info["dwell_frames"] >= self.min_dwell_frames:
                        state_info["state"] = INTERACTING
                        state_info["interaction_start_frame"] = frame_index
                        self.shelf_counts[shelf_here] += 1
                else:
                    state_info["shelf"] = None
                    state_info["dwell_frames"] = 0

            elif state == INTERACTING:
                if shelf_here == state_info["shelf"]:
                    state_info["cooldown_counter"] = 0
                else:
                    state_info["state"] = COOLDOWN
                    state_info["cooldown_counter"] = 1

            elif state == COOLDOWN:
                if shelf_here == state_info["shelf"]:
                    # returned in time -- same episode continues
                    state_info["state"] = INTERACTING
                    state_info["cooldown_counter"] = 0
                else:
                    state_info["cooldown_counter"] += 1
                    if state_info["cooldown_counter"] >= self.cooldown_frames:
                        # episode genuinely over; re-arm for a future NEW event
                        state_info["state"] = NOT_INTERACTING
                        state_info["shelf"] = None
                        state_info["dwell_frames"] = 0
                        state_info["cooldown_counter"] = 0
                        state_info["interaction_start_frame"] = None

        # drop bookkeeping for tracks no longer present at all
        stale = set(self._track_states.keys()) - seen_ids
        for tid in stale:
            del self._track_states[tid]

    def get_active_info(self, track_id, current_frame_index):
        """Returns (shelf_name_or_None, duration_seconds) for drawing."""
        info = self._track_states.get(track_id)
        if info is None or info["state"] not in (INTERACTING, COOLDOWN):
            return None, 0.0
        if info["interaction_start_frame"] is None:
            return None, 0.0
        duration = (current_frame_index - info["interaction_start_frame"]) / self.fps
        return info["shelf"], duration

    def get_counts(self):
        return dict(self.shelf_counts)
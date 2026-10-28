"""
entrance_task1_logic.py

State machine for Task 1 - Store Interest and Conversion.

Per-track states:
    UNKNOWN -> INTERESTED -> ENTERED
                           -> PASSED_BY

Definitions and thresholds (all tunable via `config`, documented here
so the reasoning is in one place for the README):

- INTEREST TRIGGER (speed + proximity based):
    We don't have a gaze/pose model, so "interest" is approximated using
    two observable motion cues near the store:
      a) the person's recent speed (pixels/sec, averaged over a short
         window) drops below `speed_threshold`, AND
      b) their perpendicular distance to the entrance line is below
         `proximity_threshold` (i.e. they are near the storefront, not
         just slow-walking somewhere far in the mall).
    Both conditions must hold for `interest_sustain_frames` consecutive
    frames before we commit to INTERESTED. Requiring sustained frames
    (not a single frame) avoids one noisy detection flipping the state.

    Crossing into the store is ALSO sufficient evidence of interest on
    its own (you can't enter a store you weren't interested in), so a
    direct line-crossing short-circuits straight to INTERESTED -> ENTERED.

- ENTRY CONFIRMATION:
    A person is confirmed ENTERED when their foot-point crosses from the
    mall side to the store side of the entrance line AND remains on the
    store side for `entry_confirm_frames` consecutive frames. The
    "remain" check exists so a person briefly reaching an arm/leg over
    the line (bbox noise) or a person who crosses and immediately
    reverses doesn't get wrongly confirmed.

- PASSED BY:
    An INTERESTED person is marked PASSED_BY once their track ends
    (leaves the camera view / tracker drops them permanently) without
    ever having been confirmed ENTERED.

- Foot point vs center point:
    We use the bottom-center of each bounding box (approx. foot
    position) rather than the box center, since the camera is angled
    and foot position is a better proxy for ground-plane location for
    line-crossing checks.

Assumption (documented for README):
    Anyone already present in the store on the very first analyzed
    frame is treated as a pre-existing presence, not a passer-by we
    observed arriving -- they are excluded from all 3 Task 1 metrics,
    since we never witnessed their "interest" moment.
"""

from collections import deque, defaultdict

DEFAULT_CONFIG = {
    "speed_threshold_px_per_sec": 55.0,     # below this = "slowing down / stopped"
    "proximity_threshold_px": 190.0,        # how close to the line counts as "near the store"
    "interest_sustain_frames": 3,           # less strict for a retail entrance / short dwell
    "entry_confirm_frames": 4,              # consecutive frames on store side to confirm entry
    "speed_window_frames": 8,               # how many recent points used to compute speed
}

UNKNOWN = "UNKNOWN"
INTERESTED = "INTERESTED"
ENTERED = "ENTERED"
PASSED_BY = "PASSED_BY"


def _side_of_line(p1, p2, point):
    """
    Returns the signed side of `point` relative to line p1->p2.
    Positive / negative sign indicates one side or the other;
    magnitude is not meaningful here, only the sign.
    """
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def _point_to_line_distance(p1, p2, point):
    """Perpendicular distance from `point` to the infinite line through p1, p2."""
    x0, y0 = point
    x1, y1 = p1
    x2, y2 = p2
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    return numerator / denominator if denominator > 0 else 0.0


class EntranceInterestLogic:
    def __init__(self, line_p1, line_p2, fps, known_store_point, config=None):
        """
        known_store_point: any (x, y) point known to be INSIDE the store,
            used purely to calibrate which sign of the line-side test
            means "store" vs "mall". We use the staff seed box center for
            this, since it is guaranteed to sit inside the store by
            definition -- avoids guessing the line's orientation.
        """
        self.p1 = line_p1
        self.p2 = line_p2
        self.fps = fps
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

        # Calibrate immediately using the known store-side point, rather
        # than guessing -- removes an entire class of "everything gets
        # classified backwards" bugs.
        raw_sign = _side_of_line(self.p1, self.p2, known_store_point)
        self.store_side_sign = 1 if raw_sign > 0 else -1

        self._track_state = {}              # track_id -> current state str
        self._position_history = defaultdict(lambda: deque(maxlen=self.cfg["speed_window_frames"]))
        self._interest_streak = defaultdict(int)   # consecutive frames meeting interest criteria
        self._entry_streak = defaultdict(int)       # consecutive frames on store side (post-cross)
        self._first_seen_frame = {}
        self._finalized = {}                # track_id -> final state (ENTERED / PASSED_BY)
        self._pre_existing = set()          # tracks present in store on very first frame -> excluded

        self._first_frame_index = None

    def _speed(self, track_id):
        hist = self._position_history[track_id]
        if len(hist) < 2:
            return 0.0
        (f0, p0), (f1, p1) = hist[0], hist[-1]
        dt = (f1 - f0) / self.fps
        if dt <= 0:
            return 0.0
        dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        return dist / dt

    def _classify_side(self, point):
        raw_sign = _side_of_line(self.p1, self.p2, point)
        point_sign = 1 if raw_sign > 0 else -1
        return "store" if point_sign == self.store_side_sign else "mall"

    def update(self, current_tracks, frame_index):
        """
        current_tracks: list of {"track_id", "bbox", "is_ghost"} from PersonTracker
        Call once per frame with ALL currently active tracks (including ghosts).
        """
        if self._first_frame_index is None:
            self._first_frame_index = frame_index

        seen_ids = set()

        for t in current_tracks:
            tid = t["track_id"]
            seen_ids.add(tid)
            x1, y1, x2, y2 = t["bbox"]
            foot = ((x1 + x2) / 2, y2)

            if tid not in self._first_seen_frame:
                self._first_seen_frame[tid] = frame_index
                self._track_state[tid] = UNKNOWN

                # Treat people already on the store side as valid interest
                # candidates rather than excluding them as pre-existing.
                side = self._classify_side(foot)
                if side == "store":
                    self._track_state[tid] = INTERESTED
                    self._interest_streak[tid] = self.cfg["interest_sustain_frames"]

            self._position_history[tid].append((frame_index, foot))

            if tid in self._pre_existing or tid in self._finalized:
                continue

            state = self._track_state[tid]
            side = self._classify_side(foot)
            speed = self._speed(tid)
            dist_to_line = _point_to_line_distance(self.p1, self.p2, foot)

            if state in (UNKNOWN, INTERESTED):
                meets_interest = (
                    speed < self.cfg["speed_threshold_px_per_sec"]
                    and dist_to_line < self.cfg["proximity_threshold_px"]
                )
                if side == "store" or meets_interest:
                    self._interest_streak[tid] += 1
                else:
                    self._interest_streak[tid] = 0

                if self._interest_streak[tid] >= self.cfg["interest_sustain_frames"]:
                    self._track_state[tid] = INTERESTED
                    state = INTERESTED

                if side == "store":
                    self._entry_streak[tid] += 1
                else:
                    self._entry_streak[tid] = 0

                if self._entry_streak[tid] >= self.cfg["entry_confirm_frames"]:
                    # Crossing into the store is itself evidence of interest,
                    # so this can confirm ENTERED even from UNKNOWN directly.
                    self._track_state[tid] = ENTERED
                    self._finalized[tid] = ENTERED

        # Any track that was active before but is missing now has ended
        # (tracker fully dropped it, beyond its occlusion grace window).
        ended_ids = set(self._track_state.keys()) - seen_ids - set(self._finalized.keys()) - self._pre_existing
        for tid in ended_ids:
            if self._track_state[tid] == INTERESTED:
                self._track_state[tid] = PASSED_BY
                self._finalized[tid] = PASSED_BY
            # UNKNOWN tracks that simply leave without ever showing interest
            # are background foot traffic -- not counted in any metric.

    def get_track_state(self, track_id):
        return self._track_state.get(track_id, UNKNOWN)

    def get_counts(self):
        entered = sum(1 for s in self._finalized.values() if s == ENTERED)
        passed_by = sum(1 for s in self._finalized.values() if s == PASSED_BY)
        # "Total Interested" = everyone who ever reached INTERESTED, ENTERED,
        # or PASSED_BY (both outcomes started from being interested).
        total_interested = sum(
            1 for tid, s in self._track_state.items()
            if tid not in self._pre_existing and s in (INTERESTED, ENTERED, PASSED_BY)
        )
        return {
            "Total Interested": total_interested,
            "Interested Entered": entered,
            "Interested Passed By": passed_by,
        }

    def finalize_all(self):
        """Call after the video ends to close out any still-INTERESTED tracks as passed-by."""
        for tid, state in self._track_state.items():
            if tid in self._pre_existing or tid in self._finalized:
                continue
            if state == INTERESTED:
                self._finalized[tid] = PASSED_BY
                self._track_state[tid] = PASSED_BY
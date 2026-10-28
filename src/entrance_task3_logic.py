"""
entrance_task3_logic.py

State machine for Task 3 - Staff-Customer Interaction.

Per (staff, customer) pair:
    NOT_INTERACTING -> INTERACTING -> NOT_INTERACTING (session already counted)

Definitions and thresholds:

- INTERACTION TRIGGER (proximity based):
    A customer is considered actively interacting with staff when the
    distance between their foot-points stays under `proximity_threshold_px`
    for `interaction_sustain_frames` consecutive frames. Sustained
    proximity (not one frame) avoids counting someone merely walking past
    the staff member as an "interaction".

    A session is counted the MOMENT this transition is confirmed
    (NOT_INTERACTING -> INTERACTING), because at that point a session has
    definitely happened -- we don't need to wait for it to end to know it
    occurred.

- SESSION END (separation based):
    The pair returns to NOT_INTERACTING once they separate beyond
    `proximity_threshold_px` for `separation_sustain_frames` consecutive
    frames (or the customer's track ends). This does not affect the
    count (already counted at start) -- it only re-arms the pair so that
    if the SAME customer interacts with the SAME staff member again
    later, that later interaction is correctly counted as a NEW session,
    per the assessment's rule.

Staff identification (documented assumption):
    Staff is identified via a manually-seeded bounding box (drawn once
    during scene setup) that initializes a tracker instance as "staff".
    For as long as that track ID persists, it is treated as the same
    staff instance (per the assessment's own rule: "staff should be
    treated as the same instance for as long as they remain in view").
    If the seeded track is lost and never recovered, no further sessions
    are attributed after that point; re-identifying a returning staff
    member as a brand-new instance is a known limitation of this
    implementation and is optional per the assessment brief.
"""

from collections import defaultdict

DEFAULT_CONFIG = {
    "proximity_threshold_px": 150.0,      # how close counts as "interacting"
    "interaction_sustain_frames": 10,     # consecutive frames of proximity to confirm a session
    "separation_sustain_frames": 10,      # consecutive frames apart to allow a new session later
}

NOT_INTERACTING = "NOT_INTERACTING"
INTERACTING = "INTERACTING"


def _distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class StaffCustomerInteractionLogic:
    def __init__(self, staff_ids, config=None):
        """
        staff_ids: list of staff labels being tracked (usually just one,
            e.g. ["staff_1"]). Included with zero sessions by default so
            the final CSV/average reflects staff with no interactions too.
        """
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.session_counts = {sid: 0 for sid in staff_ids}

        # (staff_id, customer_track_id) -> state
        self._pair_state = defaultdict(lambda: NOT_INTERACTING)
        self._proximity_streak = defaultdict(int)
        self._separation_streak = defaultdict(int)
        # which staff a given customer is CURRENTLY interacting with (for drawing)
        self.active_customer_to_staff = {}

    def update(self, staff_positions, customer_tracks):
        """
        staff_positions: dict staff_id -> foot_point (x, y), only for staff
            visible (tracked) this frame. Missing staff = no interactions
            possible against them this frame.
        customer_tracks: list of {"track_id", "bbox", ...} for everyone
            NOT identified as staff this frame.
        """
        self.active_customer_to_staff = {}

        for staff_id, staff_point in staff_positions.items():
            for c in customer_tracks:
                cid = c["track_id"]
                x1, y1, x2, y2 = c["bbox"]
                cust_point = ((x1 + x2) / 2, y2)

                key = (staff_id, cid)
                dist = _distance(staff_point, cust_point)
                state = self._pair_state[key]

                if dist < self.cfg["proximity_threshold_px"]:
                    self._proximity_streak[key] += 1
                    self._separation_streak[key] = 0
                else:
                    self._separation_streak[key] += 1
                    self._proximity_streak[key] = 0

                if state == NOT_INTERACTING:
                    if self._proximity_streak[key] >= self.cfg["interaction_sustain_frames"]:
                        self._pair_state[key] = INTERACTING
                        self.session_counts[staff_id] += 1
                elif state == INTERACTING:
                    if self._separation_streak[key] >= self.cfg["separation_sustain_frames"]:
                        self._pair_state[key] = NOT_INTERACTING

                if self._pair_state[key] == INTERACTING:
                    self.active_customer_to_staff[cid] = staff_id

    def get_results(self):
        """Returns {"per_staff": {...}, "average": float}."""
        per_staff = dict(self.session_counts)
        n = len(per_staff) if per_staff else 1
        average = sum(per_staff.values()) / n
        return {"per_staff": per_staff, "average": average}
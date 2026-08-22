import math
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


Point = Tuple[float, float]


def _l2(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    # angle between vectors in radians
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    c = float(np.dot(a, b) / (na * nb))
    c = _clamp(c, -1.0, 1.0)
    return float(math.acos(c))


@dataclass
class HandFeatures:
    handedness: str  # "Left" or "Right" (as reported by MediaPipe)
    extended: Dict[str, bool]  # thumb/index/middle/ring/pinky
    pinch: Dict[str, bool]  # thumb touching index/middle/ring/pinky
    openness: float  # 0..1-ish


class ASLRecognizer:
    """
    Lightweight, local, rule-based ASL recognizer for:
    - Numbers 1..10 (approx.)
    - Letters A..Z (static approximations; J/Z are motion-based in real ASL)
    - A few common words (heuristic)

    This is NOT a trained ML model; accuracy varies by lighting, orientation, and signing style.
    """

    FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

    # MediaPipe Hands landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self) -> None:
        self._history: Deque[Tuple[float, str]] = __import__("collections").deque(maxlen=30)
        self._last_commit_t = 0.0
        self._stable_label: Optional[str] = None
        self._stable_since_t = 0.0
        self.text = ""
        self.last_label = ""
        self.has_committed_anything = False

    def _to_points(self, hand_landmarks, w: int, h: int) -> List[Point]:
        pts: List[Point] = []
        for lm in hand_landmarks.landmark:
            pts.append((lm.x * w, lm.y * h))
        return pts

    def _hand_scale(self, pts: List[Point]) -> float:
        # Palm scale reference: distance wrist->middle_mcp
        s = _l2(pts[self.WRIST], pts[self.MIDDLE_MCP])
        return max(s, 1.0)

    def _finger_extended(self, pts: List[Point], mcp: int, pip: int, dip: int, tip: int) -> bool:
        # Finger is "extended" if the finger joints are mostly straight and the tip is farther
        # from the wrist than the PIP joint (robust-ish to rotations).
        w = np.array(pts[self.WRIST], dtype=np.float32)
        v1 = np.array(pts[pip], dtype=np.float32) - np.array(pts[mcp], dtype=np.float32)
        v2 = np.array(pts[tip], dtype=np.float32) - np.array(pts[pip], dtype=np.float32)
        straight = _angle(v1, v2) < math.radians(30)
        farther = np.linalg.norm(np.array(pts[tip]) - w) > np.linalg.norm(np.array(pts[pip]) - w)
        return bool(straight and farther)

    def _thumb_extended(self, pts: List[Point], handedness: str) -> bool:
        # Thumb tends to extend sideways. Check if tip is far from index_mcp and wrist,
        # and whether it's on the "outside" direction based on handedness.
        s = self._hand_scale(pts)
        tip = np.array(pts[self.THUMB_TIP], dtype=np.float32)
        ip = np.array(pts[self.THUMB_IP], dtype=np.float32)
        mcp = np.array(pts[self.THUMB_MCP], dtype=np.float32)
        idx_mcp = np.array(pts[self.INDEX_MCP], dtype=np.float32)
        wrist = np.array(pts[self.WRIST], dtype=np.float32)

        straight = _angle(ip - mcp, tip - ip) < math.radians(35)
        far_from_index = float(np.linalg.norm(tip - idx_mcp)) / s > 0.55
        far_from_wrist = float(np.linalg.norm(tip - wrist)) / s > 0.75

        # Outside direction heuristic: thumb tip x relative to palm center
        palm_center = (idx_mcp + np.array(pts[self.PINKY_MCP], dtype=np.float32)) * 0.5
        if handedness.lower().startswith("right"):
            outside = tip[0] > palm_center[0]
        else:
            outside = tip[0] < palm_center[0]

        return bool(straight and far_from_index and far_from_wrist and outside)

    def _pinches(self, pts: List[Point]) -> Dict[str, bool]:
        s = self._hand_scale(pts)
        thumb = pts[self.THUMB_TIP]
        out: Dict[str, bool] = {}
        for name, tip_idx in [("index", self.INDEX_TIP), ("middle", self.MIDDLE_TIP), ("ring", self.RING_TIP), ("pinky", self.PINKY_TIP)]:
            d = _l2(thumb, pts[tip_idx]) / s
            out[name] = bool(d < 0.28)
        return out

    def extract_features(self, hand_landmarks, handedness: str, w: int, h: int) -> HandFeatures:
        pts = self._to_points(hand_landmarks, w, h)
        ext = {
            "thumb": self._thumb_extended(pts, handedness),
            "index": self._finger_extended(pts, self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            "middle": self._finger_extended(pts, self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            "ring": self._finger_extended(pts, self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            "pinky": self._finger_extended(pts, self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        }
        pinch = self._pinches(pts)

        # Openness: mean tip distance from wrist normalized
        s = self._hand_scale(pts)
        wrist = pts[self.WRIST]
        mean_tip = float(
            np.mean(
                [
                    _l2(wrist, pts[self.THUMB_TIP]),
                    _l2(wrist, pts[self.INDEX_TIP]),
                    _l2(wrist, pts[self.MIDDLE_TIP]),
                    _l2(wrist, pts[self.RING_TIP]),
                    _l2(wrist, pts[self.PINKY_TIP]),
                ]
            )
            / s
        )
        openness = float(_clamp((mean_tip - 0.9) / 0.8, 0.0, 1.0))

        return HandFeatures(handedness=handedness, extended=ext, pinch=pinch, openness=openness)

    def _count_extended(self, f: HandFeatures) -> int:
        return int(sum(1 for k in self.FINGERS if f.extended.get(k, False)))

    def _is_fist(self, f: HandFeatures) -> bool:
        return self._count_extended(f) <= 1 and f.openness < 0.35

    def _is_open_palm(self, f: HandFeatures) -> bool:
        return self._count_extended(f) >= 4 and f.openness > 0.65

    def recognize_number(self, f: HandFeatures) -> Optional[str]:
        # 1..5: count of extended fingers excluding thumb unless it is clearly extended
        ext = f.extended
        n = int(ext["index"]) + int(ext["middle"]) + int(ext["ring"]) + int(ext["pinky"])
        thumb = ext["thumb"]

        if self._is_fist(f) and thumb:
            return "10"  # approximate (ASL 10 is typically thumb up + motion)

        if n in (1, 2, 3, 4) and not thumb:
            return str(n)
        if n == 4 and thumb:
            return "5"

        # 6..9: thumb touches a specific finger (approx)
        # 6=thumb+pinky pinch, 7=thumb+ring, 8=thumb+middle, 9=thumb+index
        if f.pinch.get("pinky") and n >= 3:
            return "6"
        if f.pinch.get("ring") and n >= 2:
            return "7"
        if f.pinch.get("middle") and n >= 2:
            return "8"
        if f.pinch.get("index") and n >= 1:
            return "9"

        return None

    def recognize_letter(self, f: HandFeatures) -> Optional[str]:
        e = f.extended
        c = self._count_extended(f)

        # Strong, simpler shapes first
        if self._is_open_palm(f) and not e["thumb"]:
            return "B"
        if e["index"] and e["thumb"] and not e["middle"] and not e["ring"] and not e["pinky"]:
            return "L"
        if e["index"] and not e["middle"] and not e["ring"] and not e["pinky"] and not e["thumb"]:
            return "D"  # can also look like 1; we'll allow letter mode to output D
        if e["index"] and e["middle"] and not e["ring"] and not e["pinky"]:
            return "V"
        if e["index"] and e["middle"] and e["ring"] and not e["pinky"]:
            return "W"
        if e["pinky"] and not e["index"] and not e["middle"] and not e["ring"]:
            return "I"
        if e["thumb"] and e["pinky"] and not e["index"] and not e["middle"] and not e["ring"]:
            return "Y"
        if e["thumb"] and e["index"] and e["pinky"] and not e["middle"] and not e["ring"]:
            return "I LOVE YOU"

        # F: thumb+index pinch with other 3 extended
        if f.pinch.get("index") and e["middle"] and e["ring"] and e["pinky"]:
            return "F"

        # A/E/S/T/M/N are fist-like; we approximate using thumb extension + openness
        if self._is_fist(f):
            if e["thumb"]:
                return "A"
            return "S"

        # C/O: open hand curved; approximate by openness + extended count
        if c >= 4 and 0.40 <= f.openness <= 0.70:
            return "C"
        if c >= 4 and f.openness < 0.55 and e["thumb"]:
            return "O"

        # U/R: index+middle up; R has crossed fingers (hard). We approximate:
        if e["index"] and e["middle"] and not e["ring"] and not e["pinky"] and not e["thumb"]:
            return "U"

        # K/P: thumb+middle+index (like K), but P is rotated down. We'll output K.
        if e["index"] and e["middle"] and e["thumb"] and not e["ring"] and not e["pinky"]:
            return "K"

        # H: index+middle extended, thumb tucked; like "V" but flatter; we approximate with same as V.
        if e["index"] and e["middle"] and not e["ring"] and not e["pinky"] and not e["thumb"]:
            return "H"

        # X: index bent (not extended) but up-ish; we approximate by only thumb+index somewhat active (fails often)
        if not e["index"] and not e["middle"] and not e["ring"] and not e["pinky"] and e["thumb"]:
            return "X"

        # J/Z are motion-based; best-effort static fallback:
        if e["pinky"] and c == 1:
            return "J"
        if e["index"] and c == 1:
            return "Z"

        return None

    def recognize_word(self, f_left: Optional[HandFeatures], f_right: Optional[HandFeatures]) -> Optional[str]:
        # Simple single-frame words (heuristic)
        # YES: fist
        # NO: index+middle extended + thumb out (approx)
        # HELLO: open palm
        if f_left and self._is_fist(f_left):
            return "YES"
        if f_right and self._is_fist(f_right):
            return "YES"

        for f in [f_right, f_left]:
            if not f:
                continue
            if self._is_open_palm(f):
                return "HELLO"
            e = f.extended
            if e["index"] and e["middle"] and not e["ring"] and not e["pinky"] and e["thumb"]:
                return "NO"
        return None

    def _update_stable(self, raw_label: str, now: float) -> Optional[str]:
        # Require raw label to be stable for a short window before "committing" into text.
        if not raw_label:
            self._stable_label = None
            self._stable_since_t = 0.0
            return None

        if self._stable_label != raw_label:
            self._stable_label = raw_label
            self._stable_since_t = now
            return None

        if now - self._stable_since_t >= 0.45 and now - self._last_commit_t >= 0.55:
            self._last_commit_t = now
            return raw_label
        return None

    def _append_text(self, token: str) -> None:
        if not token:
            return
        if token in ("YES", "NO", "HELLO", "SOS", "I LOVE YOU"):
            # Words as separate tokens
            if self.text and not self.text.endswith(" "):
                self.text += " "
            self.text += token + " "
            return
        # Letters/numbers: append without extra spaces (but keep words separated)
        self.text += token

    def commit_token(self, token: str) -> None:
        """
        Force-commit a token (from an external classifier).
        Token may be: letter, number, or word like "NO", "I LOVE YOU", "SOS".
        """
        token = (token or "").strip()
        if not token:
            return
        now = time.time()
        if now - self._last_commit_t < 0.55:
            return
        self._last_commit_t = now
        self.has_committed_anything = True
        self.last_label = token
        self._history.append((now, token))
        self._append_text(token)

    def _clear_all(self) -> None:
        self.text = ""
        self.last_label = ""
        self._history.clear()
        self._stable_label = None
        self._stable_since_t = 0.0
        self._last_commit_t = 0.0
        self.has_committed_anything = False

    def _is_heart_two_hands(self, pts_a: List[Point], pts_b: List[Point]) -> bool:
        # Simple heart heuristic:
        # - both hands have thumb-index pinch (tips close within each hand)
        # - the two index tips are close to each other
        # - the two thumb tips are close to each other
        # - wrists are not too close (so it's actually two hands)
        def pinch_index(pts: List[Point]) -> bool:
            s = self._hand_scale(pts)
            return (_l2(pts[self.THUMB_TIP], pts[self.INDEX_TIP]) / s) < 0.30

        if not (pinch_index(pts_a) and pinch_index(pts_b)):
            return False

        sa = self._hand_scale(pts_a)
        sb = self._hand_scale(pts_b)
        s = float(max(sa, sb, 1.0))

        idx_close = (_l2(pts_a[self.INDEX_TIP], pts_b[self.INDEX_TIP]) / s) < 0.65
        th_close = (_l2(pts_a[self.THUMB_TIP], pts_b[self.THUMB_TIP]) / s) < 0.75
        wrists_apart = (_l2(pts_a[self.WRIST], pts_b[self.WRIST]) / s) > 0.55
        return bool(idx_close and th_close and wrists_apart)

    def _detect_sos(self, now: float) -> bool:
        recent = [lab for (t, lab) in self._history if now - t <= 2.5]
        # Look for S-O-S in recent committed tokens
        s = "".join([r for r in recent if len(r) == 1 and r.isalpha()])
        return "SOS" in s[-10:]

    def step(self, hands: List[Tuple[object, str]], w: int, h: int) -> Tuple[str, str]:
        """
        hands: list of (hand_landmarks, handedness_label) for up to 2 hands
        Returns (overlay_line, raw_label_debug)
        """
        now = time.time()

        f_left: Optional[HandFeatures] = None
        f_right: Optional[HandFeatures] = None
        pts_left: Optional[List[Point]] = None
        pts_right: Optional[List[Point]] = None
        for hlm, hand_label in hands[:2]:
            pts = self._to_points(hlm, w, h)
            f = self.extract_features(hlm, hand_label, w, h)
            if hand_label.lower().startswith("left"):
                f_left = f
                pts_left = pts
            else:
                f_right = f
                pts_right = pts

        # High-priority command gestures (work anytime):
        # 1) Thumb + middle touch => clear all text
        for f in [f_right, f_left]:
            if f and f.pinch.get("middle"):
                self._clear_all()
                return "", ""

        # 1b) Thumb + pinky touch => "CLAPPING"
        for f in [f_right, f_left]:
            if f and f.pinch.get("pinky") and (now - self._last_commit_t) >= 0.60:
                self._last_commit_t = now
                self.has_committed_anything = True
                self.last_label = "CLAPPING"
                self._history.append((now, "CLAPPING"))
                self._append_text("CLAPPING")
                overlay = self.text.strip()
                return overlay, "CLAPPING"

        # 1c) Right hand: fingertips (except thumb) to top of palm (near finger bases) => "HELP"
        # Interpreted as index/middle/ring/pinky tips close to their MCP knuckles at once.
        if pts_right is not None and (now - self._last_commit_t) >= 0.70:
            s = float(max(self._hand_scale(pts_right), 1.0))
            pairs = [
                (pts_right[self.INDEX_TIP], pts_right[self.INDEX_MCP]),
                (pts_right[self.MIDDLE_TIP], pts_right[self.MIDDLE_MCP]),
                (pts_right[self.RING_TIP], pts_right[self.RING_MCP]),
                (pts_right[self.PINKY_TIP], pts_right[self.PINKY_MCP]),
            ]
            near_mcp = [(_l2(tip, mcp) / s) < 0.46 for (tip, mcp) in pairs]
            if all(near_mcp):
                self._last_commit_t = now
                self.has_committed_anything = True
                self.last_label = "HELP"
                self._history.append((now, "HELP"))
                self._append_text("HELP")
                overlay = self.text.strip()
                return overlay, "HELP"

        # 2) Two-hand heart => "I LOVE YOU"
        if pts_left is not None and pts_right is not None:
            # 1c) Two thumbs touch AND two index fingertips touch => "I LOVE YOU"
            s = float(max(self._hand_scale(pts_left), self._hand_scale(pts_right), 1.0))
            thumbs_touch = (_l2(pts_left[self.THUMB_TIP], pts_right[self.THUMB_TIP]) / s) < 0.60
            indexes_touch = (_l2(pts_left[self.INDEX_TIP], pts_right[self.INDEX_TIP]) / s) < 0.55
            wrists_apart = (_l2(pts_left[self.WRIST], pts_right[self.WRIST]) / s) > 0.50
            if thumbs_touch and indexes_touch and wrists_apart and (now - self._last_commit_t) >= 0.70:
                self._last_commit_t = now
                self.has_committed_anything = True
                self.last_label = "I LOVE YOU"
                self._history.append((now, "I LOVE YOU"))
                self._append_text("I LOVE YOU")
                overlay = self.text.strip()
                return overlay, "I LOVE YOU"

            # 1c) Two index fingertips touch => "UWU"
            if (_l2(pts_left[self.INDEX_TIP], pts_right[self.INDEX_TIP]) / s) < 0.55 and (now - self._last_commit_t) >= 0.70:
                self._last_commit_t = now
                self.has_committed_anything = True
                self.last_label = "UWU"
                self._history.append((now, "UWU"))
                self._append_text("UWU")
                overlay = self.text.strip()
                return overlay, "UWU"

            if self._is_heart_two_hands(pts_left, pts_right) and (now - self._last_commit_t) >= 0.70:
                self._last_commit_t = now
                self.has_committed_anything = True
                self.last_label = "I LOVE YOU"
                self._history.append((now, "I LOVE YOU"))
                self._append_text("I LOVE YOU")
                overlay = self.text.strip()
                return overlay, "I LOVE YOU"

        # 3) Thumb + index touch => "NO" (word)
        for f in [f_right, f_left]:
            if f and f.pinch.get("index") and (now - self._last_commit_t) >= 0.60:
                self._last_commit_t = now
                self.has_committed_anything = True
                self.last_label = "NO"
                self._history.append((now, "NO"))
                self._append_text("NO")
                overlay = self.text.strip()
                return overlay, "NO"

        # Prefer "word" detections, then numbers, then letters.
        raw_label = ""
        word = self.recognize_word(f_left, f_right)
        if word:
            raw_label = word
        else:
            # If either hand shows a number, use it (right hand preferred)
            for f in [f_right, f_left]:
                if not f:
                    continue
                num = self.recognize_number(f)
                if num:
                    raw_label = num
                    break
            if not raw_label:
                for f in [f_right, f_left]:
                    if not f:
                        continue
                    let = self.recognize_letter(f)
                    if let:
                        raw_label = let
                        break

        commit = self._update_stable(raw_label, now)
        if commit:
            self.last_label = commit
            self.has_committed_anything = True
            self._history.append((now, commit))

            if commit == "I LOVE YOU":
                self._append_text("I LOVE YOU")
            elif commit in ("YES", "NO", "HELLO"):
                self._append_text(commit)
            elif commit.isdigit():
                # add separator if prior char was a digit
                if self.text and self.text[-1].isdigit():
                    self.text += " "
                self._append_text(commit)
                self.text += " "
            elif len(commit) == 1 and commit.isalpha():
                self._append_text(commit)

            if self._detect_sos(now):
                self._append_text(" SOS")
                self._history.append((now, "SOS"))

            # Keep text from growing forever
            if len(self.text) > 80:
                self.text = self.text[-80:]

        # Only show overlay after the user has actually produced a committed symbol.
        overlay = self.text.strip() if self.text.strip() else ""
        committed = commit or ""
        return overlay, committed


from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ControlState:
    active: bool = False
    anchor_x: float = 0.0
    anchor_y: float = 0.0
    anchor_brightness: Optional[int] = None
    anchor_volume: Optional[float] = None
    last_apply_t: float = 0.0


class BrightnessVolumeController:
    """
    Windows control helper.

    Gesture mapping (left hand):
    - thumb+index pinch held => control mode
      - move vertically => brightness
      - move horizontally => volume
    """

    def __init__(self) -> None:
        self.state = ControlState()

        # Lazily imported optional deps
        self._sbc = None
        self._pycaw = None
        self._vol = None

    def _ensure_brightness(self) -> bool:
        if self._sbc is not None:
            return True
        try:
            import screen_brightness_control as sbc  # type: ignore

            self._sbc = sbc
            return True
        except Exception:
            return False

    def _ensure_volume(self) -> bool:
        if self._vol is not None:
            return True
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL  # type: ignore
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            self._vol = volume
            return True
        except Exception:
            return False

    def get_brightness(self) -> Optional[int]:
        if not self._ensure_brightness():
            return None
        try:
            cur = self._sbc.get_brightness()
            if isinstance(cur, list) and cur:
                return int(cur[0])
            return int(cur)
        except Exception:
            return None

    def set_brightness(self, value: int) -> bool:
        if not self._ensure_brightness():
            return False
        try:
            self._sbc.set_brightness(int(max(0, min(100, value))))
            return True
        except Exception:
            return False

    def get_volume(self) -> Optional[float]:
        if not self._ensure_volume():
            return None
        try:
            # 0..1 scalar
            return float(self._vol.GetMasterVolumeLevelScalar())
        except Exception:
            return None

    def set_volume(self, value01: float) -> bool:
        if not self._ensure_volume():
            return False
        try:
            v = float(max(0.0, min(1.0, value01)))
            self._vol.SetMasterVolumeLevelScalar(v, None)
            return True
        except Exception:
            return False

    def update(
        self,
        pinch_active: bool,
        palm_xy: Tuple[float, float],
        frame_wh: Tuple[int, int],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns (brightness_percent or None, volume_percent or None) as feedback for UI.
        """
        now = time.time()
        x, y = palm_xy
        w, h = frame_wh

        # Throttle OS calls
        if now - self.state.last_apply_t < 0.06:
            return None, None

        if not pinch_active:
            self.state.active = False
            self.state.anchor_brightness = None
            self.state.anchor_volume = None
            return None, None

        if not self.state.active:
            self.state.active = True
            self.state.anchor_x = x
            self.state.anchor_y = y
            self.state.anchor_brightness = self.get_brightness()
            self.state.anchor_volume = self.get_volume()
            return None, None

        dx = (x - self.state.anchor_x) / max(float(w), 1.0)
        dy = (y - self.state.anchor_y) / max(float(h), 1.0)

        # Decide mode by dominant axis (with deadzone)
        if abs(dy) > abs(dx) and abs(dy) > 0.03:
            # Brightness: up increases, down decreases
            if self.state.anchor_brightness is None:
                self.state.anchor_brightness = self.get_brightness() or 50
            delta = int((-dy) * 180)  # sensitivity
            target = int(self.state.anchor_brightness + delta)
            ok = self.set_brightness(target)
            self.state.last_apply_t = now
            if ok:
                return int(max(0, min(100, target))), None
            return None, None

        if abs(dx) >= abs(dy) and abs(dx) > 0.03:
            # Volume: right increases, left decreases
            if self.state.anchor_volume is None:
                self.state.anchor_volume = self.get_volume() or 0.5
            delta01 = float(dx * 1.8)  # sensitivity
            target01 = float(self.state.anchor_volume + delta01)
            ok = self.set_volume(target01)
            self.state.last_apply_t = now
            if ok:
                return None, int(max(0, min(100, round(target01 * 100))))
            return None, None

        return None, None


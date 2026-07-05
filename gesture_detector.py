"""
gesture_detector.py

Translates raw HandResult data (from hand_tracker.py) into high-level,
semantic gestures used by the application: Draw, Selection, Save, Clear.

Designed to be easily extensible: to add a new gesture, add a new
GestureType enum value and a corresponding detection branch in
`GestureDetector.detect`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import config
from hand_tracker import HandResult


class GestureType(Enum):
    """Enumerates all semantic gestures recognized by the application."""

    NONE = auto()
    DRAW = auto()
    SELECTION = auto()
    SAVE = auto()
    CLEAR = auto()


@dataclass
class GestureEvent:
    """
    Result of a single gesture detection pass.

    Attributes:
        gesture_type: The recognized gesture for this frame.
        point: The relevant pixel coordinate for the gesture (e.g. fingertip
            position for DRAW/SELECTION), or None if not applicable.
        is_new_action: True only on the frame a discrete, cooldown-gated
            action (SAVE/CLEAR) is first triggered. Continuous gestures
            (DRAW/SELECTION) do not use this flag.
    """

    gesture_type: GestureType
    point: Optional[tuple[int, int]] = None
    is_new_action: bool = False


class GestureDetector:
    """
    Stateful gesture recognizer that applies cooldown logic to discrete,
    one-shot actions (Save / Clear) so that holding a pose doesn't fire
    the action repeatedly every frame.
    """

    def __init__(self, action_cooldown_seconds: float = config.ACTION_COOLDOWN_SECONDS) -> None:
        """
        Args:
            action_cooldown_seconds: Minimum time between repeated
                discrete-action triggers (Save/Clear).
        """
        self._action_cooldown_seconds = action_cooldown_seconds
        self._last_action_time: float = 0.0

    def detect(self, hand: Optional[HandResult]) -> GestureEvent:
        """
        Determine the current gesture based on the latest hand tracking result.

        Args:
            hand: The tracked hand for the current frame, or None if no
                hand is currently detected.

        Returns:
            A `GestureEvent` describing the recognized gesture this frame.
        """
        if hand is None:
            return GestureEvent(gesture_type=GestureType.NONE)

        if hand.is_index_and_middle_up():
            return GestureEvent(gesture_type=GestureType.SELECTION, point=hand.index_tip())

        if hand.is_index_up_only():
            return GestureEvent(gesture_type=GestureType.DRAW, point=hand.index_tip())

        if hand.is_thumb_up_only():
            return self._discrete_action(GestureType.SAVE, hand.center)

        if hand.is_fist():
            return self._discrete_action(GestureType.CLEAR, hand.center)

        return GestureEvent(gesture_type=GestureType.NONE)

    def _discrete_action(self, gesture_type: GestureType, point: tuple[int, int]) -> GestureEvent:
        """
        Apply cooldown gating to a discrete (one-shot) gesture action.

        Args:
            gesture_type: The discrete gesture type detected this frame.
            point: Reference point associated with the gesture.

        Returns:
            A `GestureEvent` with `is_new_action=True` only if the cooldown
            period has elapsed since the last discrete action.
        """
        now = time.time()
        if now - self._last_action_time >= self._action_cooldown_seconds:
            self._last_action_time = now
            return GestureEvent(gesture_type=gesture_type, point=point, is_new_action=True)
        return GestureEvent(gesture_type=gesture_type, point=point, is_new_action=False)

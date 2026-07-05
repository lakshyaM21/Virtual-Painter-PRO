"""
hand_tracker.py

Wraps the latest MediaPipe Tasks API (HandLandmarker) to provide a clean,
typed interface for retrieving hand landmarks, finger states, gesture
information, hand center and bounding box for each detected hand.

This module intentionally avoids the deprecated `mp.solutions.hands` API
and instead uses `mediapipe.tasks.python.vision.HandLandmarker`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config


@dataclass
class HandResult:
    """
    Structured result for a single detected hand.

    Attributes:
        landmarks_px: List of (x, y) pixel coordinates for all 21 landmarks.
        landmarks_norm: List of (x, y, z) normalized coordinates (0-1) as
            returned directly by MediaPipe.
        finger_states: List of 5 booleans [thumb, index, middle, ring, pinky]
            indicating whether each finger is extended ("up").
        handedness: "Left" or "Right" as reported by MediaPipe.
        bounding_box: (x_min, y_min, x_max, y_max) in pixel coordinates.
        center: (cx, cy) centroid of the hand in pixel coordinates.
    """

    landmarks_px: List[Tuple[int, int]] = field(default_factory=list)
    landmarks_norm: List[Tuple[float, float, float]] = field(default_factory=list)
    finger_states: List[bool] = field(default_factory=lambda: [False] * 5)
    handedness: str = "Unknown"
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[int, int] = (0, 0)

    def fingers_up_count(self) -> int:
        """Return the number of extended fingers."""
        return sum(1 for state in self.finger_states if state)

    def is_index_up_only(self) -> bool:
        """True if only the index finger is extended (drawing gesture)."""
        thumb, index, middle, ring, pinky = self.finger_states
        return index and not middle and not ring and not pinky

    def is_index_and_middle_up(self) -> bool:
        """True if index and middle are extended, ring/pinky are not (selection gesture)."""
        thumb, index, middle, ring, pinky = self.finger_states
        return index and middle and not ring and not pinky

    def is_fist(self) -> bool:
        """True if all fingers are closed (clear-canvas gesture)."""
        return not any(self.finger_states)

    def is_thumb_up_only(self) -> bool:
        """True if only the thumb is extended (save gesture)."""
        thumb, index, middle, ring, pinky = self.finger_states
        return thumb and not index and not middle and not ring and not pinky

    def index_tip(self) -> Tuple[int, int]:
        """Pixel coordinates of the index fingertip."""
        return self.landmarks_px[config.INDEX_TIP]

    def middle_tip(self) -> Tuple[int, int]:
        """Pixel coordinates of the middle fingertip."""
        return self.landmarks_px[config.MIDDLE_TIP]


class HandTracker:
    """
    High-level hand tracking interface built on MediaPipe Tasks HandLandmarker.
    """

    def __init__(
        self,
        model_path: str = config.HAND_LANDMARKER_MODEL_PATH,
        num_hands: int = config.NUM_HANDS,
        min_detection_confidence: float = config.MIN_HAND_DETECTION_CONFIDENCE,
        min_presence_confidence: float = config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        """
        Initialize the HandLandmarker in VIDEO running mode.

        Args:
            model_path: Filesystem path to `hand_landmarker.task`.
            num_hands: Maximum number of hands to detect simultaneously.
            min_detection_confidence: Minimum confidence for initial hand detection.
            min_presence_confidence: Minimum confidence for hand presence.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms: int = 0

    def process(self, frame_bgr: np.ndarray) -> List[HandResult]:
        """
        Run hand detection on a single BGR frame.

        Args:
            frame_bgr: The input frame in BGR color order (as returned by OpenCV).

        Returns:
            A list of `HandResult`, one per detected hand (possibly empty).
        """
        height, width = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB without extra copy overhead of cvtColor
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

        self._timestamp_ms += 1
        detection_result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        results: List[HandResult] = []
        if not detection_result.hand_landmarks:
            return results

        for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
            landmarks_px: List[Tuple[int, int]] = []
            landmarks_norm: List[Tuple[float, float, float]] = []
            xs: List[int] = []
            ys: List[int] = []

            for landmark in hand_landmarks:
                px = int(landmark.x * width)
                py = int(landmark.y * height)
                landmarks_px.append((px, py))
                landmarks_norm.append((landmark.x, landmark.y, landmark.z))
                xs.append(px)
                ys.append(py)

            handedness_label = "Unknown"
            if detection_result.handedness and hand_index < len(detection_result.handedness):
                handedness_label = detection_result.handedness[hand_index][0].category_name

            finger_states = self._compute_finger_states(landmarks_px, handedness_label)

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            center = (int(np.mean(xs)), int(np.mean(ys)))

            results.append(
                HandResult(
                    landmarks_px=landmarks_px,
                    landmarks_norm=landmarks_norm,
                    finger_states=finger_states,
                    handedness=handedness_label,
                    bounding_box=(x_min, y_min, x_max, y_max),
                    center=center,
                )
            )

        return results

    @staticmethod
    def _compute_finger_states(
        landmarks_px: Sequence[Tuple[int, int]],
        handedness_label: str,
    ) -> List[bool]:
        """
        Determine which fingers are extended ("up") using landmark geometry.

        The thumb is evaluated based on horizontal displacement (relative to
        handedness) since it moves sideways rather than vertically. The
        remaining four fingers are evaluated by comparing the tip's Y
        coordinate against its PIP joint's Y coordinate (tip above PIP means
        extended, since image Y grows downward).

        Args:
            landmarks_px: 21 (x, y) pixel-coordinate landmarks for the hand.
            handedness_label: "Left" or "Right" as reported by MediaPipe.

        Returns:
            List of 5 booleans: [thumb, index, middle, ring, pinky].
        """
        states: List[bool] = []

        # Thumb: compare tip X to IP joint X, direction depends on handedness.
        thumb_tip_x = landmarks_px[config.THUMB_TIP][0]
        thumb_ip_x = landmarks_px[config.THUMB_IP][0]
        if handedness_label == "Right":
            thumb_up = thumb_tip_x < thumb_ip_x
        else:
            thumb_up = thumb_tip_x > thumb_ip_x
        states.append(thumb_up)

        # Remaining four fingers: tip above pip (smaller Y) means extended.
        for tip_idx, pip_idx in config.FINGER_TIP_PIP_PAIRS:
            tip_y = landmarks_px[tip_idx][1]
            pip_y = landmarks_px[pip_idx][1]
            states.append(tip_y < pip_y)

        return states

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

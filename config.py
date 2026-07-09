"""
config.py

Centralized configuration for Virtual Painter Pro
All constants, paths, thresholds and tunable parameters live here so that
the rest of the codebase never hardcodes "magic numbers".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


# =====================================================================
# PATHS
# =====================================================================

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
HEADER_DIR: str = os.path.join(BASE_DIR, "Header")
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
OUTPUT_DIR: str = os.path.join(BASE_DIR, "Output")

HAND_LANDMARKER_MODEL_PATH: str = os.path.join(MODELS_DIR, "hand_landmarker.task")

# Header images used for the color/tool selection bar (in display order).
HEADER_IMAGE_FILENAMES: Tuple[str, ...] = ("1.jpg", "2.jpg", "3.jpg", "4.jpg")


# =====================================================================
# CAMERA / WINDOW
# =====================================================================

CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
CAMERA_FPS_TARGET: int = 30

WINDOW_NAME: str = "Virtual Painter Pro"

# Height (in px) reserved at the top of the frame for the header bar.
HEADER_HEIGHT: int = 125


# =====================================================================
# MEDIAPIPE HAND LANDMARKER SETTINGS
# =====================================================================

NUM_HANDS: int = 1
MIN_HAND_DETECTION_CONFIDENCE: float = 0.7
MIN_HAND_PRESENCE_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.6

# MediaPipe hand landmark indices (21-point hand model).
WRIST: int = 0
THUMB_CMC: int = 1
THUMB_MCP: int = 2
THUMB_IP: int = 3
THUMB_TIP: int = 4
INDEX_MCP: int = 5
INDEX_PIP: int = 6
INDEX_DIP: int = 7
INDEX_TIP: int = 8
MIDDLE_MCP: int = 9
MIDDLE_PIP: int = 10
MIDDLE_DIP: int = 11
MIDDLE_TIP: int = 12
RING_MCP: int = 13
RING_PIP: int = 14
RING_DIP: int = 15
RING_TIP: int = 16
PINKY_MCP: int = 17
PINKY_PIP: int = 18
PINKY_DIP: int = 19
PINKY_TIP: int = 20

# Finger tip / pip pairs used for the "is finger up" heuristic.
# (tip_index, pip_index)
FINGER_TIP_PIP_PAIRS: Tuple[Tuple[int, int], ...] = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)


# =====================================================================
# DRAWING / BRUSH SETTINGS
# =====================================================================

BRUSH_THICKNESS: int = 12
ERASER_THICKNESS: int = 80

# Minimum distance (in px) the fingertip must move before a new line
# segment is drawn. Helps to reduce jitter / shaky lines.
MIN_DRAW_DISTANCE: float = 4.0

# Smoothing factor for exponential moving average applied to the
# fingertip position before drawing (0 < alpha <= 1). Lower = smoother.
SMOOTHING_ALPHA: float = 0.25

# Maximum number of canvas snapshots kept for undo functionality.
MAX_UNDO_STACK: int = 25

CANVAS_BLEND_THRESHOLD: int = 50  # grayscale threshold used to build the inverse mask


# =====================================================================
# COLORS (BGR, since OpenCV uses BGR ordering)
# =====================================================================

COLOR_PURPLE: Tuple[int, int, int] = (255, 0, 155)
COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)
COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
COLOR_ERASER: Tuple[int, int, int] = (0, 0, 0)  # black == erase on canvas
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: Tuple[int, int, int] = (0, 0, 0)
COLOR_YELLOW: Tuple[int, int, int] = (0, 255, 255)
COLOR_GRAY: Tuple[int, int, int] = (60, 60, 60)

# Maps header index (which header image is currently shown) -> draw color.
# Index 3 (the 4th header image) is reserved for the Eraser tool.
HEADER_INDEX_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: COLOR_PURPLE,
    1: COLOR_BLUE,
    2: COLOR_GREEN,
    3: COLOR_ERASER,
}

HEADER_INDEX_TO_NAME: Dict[int, str] = {
    0: "Purple",
    1: "Blue",
    2: "Green",
    3: "Eraser",
}

# Approximate horizontal segment width of each header option, used to
# translate an X coordinate into a header selection index.
HEADER_SEGMENT_COUNT: int = 4


# =====================================================================
# GESTURE / SELECTION SETTINGS
# =====================================================================

# Cooldown (in seconds) applied after a discrete gesture action
# (save / clear / undo) fires, to avoid repeated triggers on a held pose.
ACTION_COOLDOWN_SECONDS: float = 1.2

# Distance (in px, in image space) below which the thumb tip and index
# MCP joint are considered "far enough apart" to call a thumb a valid
# "thumbs up" pose along with the other closed-finger checks.
THUMB_UP_Y_MARGIN: int = 20


# =====================================================================
# UI / TEXT SETTINGS
# =====================================================================

FONT = None  # populated lazily in utils.py using cv2.FONT_HERSHEY_SIMPLEX to avoid importing cv2 at import-time here
FONT_SCALE_LARGE: float = 1.0
FONT_SCALE_MEDIUM: float = 0.75
FONT_SCALE_SMALL: float = 0.6
FONT_THICKNESS: int = 2

TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
FPS_TEXT_POSITION: Tuple[int, int] = (20, 690)
MODE_TEXT_POSITION: Tuple[int, int] = (20, 650)
COLOR_SWATCH_POSITION: Tuple[int, int] = (1180, 660)
COLOR_SWATCH_RADIUS: int = 25


# =====================================================================
# KEYBOARD SHORTCUTS
# =====================================================================

KEY_SAVE: str = "s"
KEY_CLEAR: str = "c"
KEY_UNDO: str = "z"
KEY_ESC: int = 27


# =====================================================================
# MISC
# =====================================================================

SAVE_FILENAME_PREFIX: str = "painting"
SAVE_FILE_EXTENSION: str = ".png"


@dataclass
class RuntimeState:
    """
    Mutable runtime state shared across the application loop.

    Using a dataclass keeps main.py free of a large number of loose
    local variables and makes state passing between helper functions
    explicit and type-safe.
    """

    draw_color: Tuple[int, int, int] = field(default_factory=lambda: COLOR_PURPLE)
    header_index: int = 0
    brush_thickness: int = BRUSH_THICKNESS
    eraser_thickness: int = ERASER_THICKNESS
    current_mode: str = "Idle"
    prev_point: Tuple[int, int] | None = None
    smoothed_point: Tuple[float, float] | None = None
    last_action_time: float = 0.0

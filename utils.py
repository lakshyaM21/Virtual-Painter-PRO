"""
utils.py

General-purpose helper utilities used across Virtual Painter Pro:
- FPS calculation
- Header image loading
- Canvas saving
- Small drawing / geometry helpers
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


class FPSCounter:
    """Tracks and smooths frames-per-second using an exponential moving average."""

    def __init__(self, smoothing: float = 0.9) -> None:
        """
        Args:
            smoothing: Weight given to the previous FPS estimate (0-1).
                       Higher values produce a smoother, slower-reacting counter.
        """
        self._smoothing: float = smoothing
        self._prev_time: float = time.time()
        self._fps: float = 0.0

    def update(self) -> float:
        """
        Compute the current FPS based on time elapsed since the last call.

        Returns:
            The smoothed FPS value.
        """
        current_time = time.time()
        elapsed = current_time - self._prev_time
        self._prev_time = current_time

        if elapsed <= 0:
            return self._fps

        instantaneous_fps = 1.0 / elapsed
        self._fps = (self._smoothing * self._fps) + ((1.0 - self._smoothing) * instantaneous_fps)
        return self._fps


def load_header_images(header_dir: str, filenames: Tuple[str, ...], target_width: int) -> List[np.ndarray]:
    """
    Load and resize header option images from disk.

    Args:
        header_dir: Directory containing the header images.
        filenames: Ordered tuple of image filenames to load.
        target_width: Width (in px) each header image should be resized to.

    Returns:
        List of loaded, resized BGR images in the same order as `filenames`.

    Raises:
        FileNotFoundError: If a header image cannot be found or loaded.
    """
    images: List[np.ndarray] = []
    for filename in filenames:
        path = os.path.join(header_dir, filename)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(
                f"Could not load header image '{path}'. "
                f"Ensure the Header/ folder contains: {filenames}"
            )
        resized = cv2.resize(image, (target_width, config.HEADER_HEIGHT))
        images.append(resized)
    return images


def ensure_output_dir(output_dir: str) -> None:
    """Create the output directory if it does not already exist."""
    os.makedirs(output_dir, exist_ok=True)


def save_canvas(canvas: np.ndarray, output_dir: str, prefix: str, extension: str) -> str:
    """
    Save the given canvas image to disk with a timestamped filename.

    Args:
        canvas: The image (BGR numpy array) to save.
        output_dir: Directory to save into (created if missing).
        prefix: Filename prefix.
        extension: File extension, including the leading dot (e.g. ".png").

    Returns:
        The full path the file was saved to.
    """
    ensure_output_dir(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}{extension}"
    full_path = os.path.join(output_dir, filename)
    cv2.imwrite(full_path, canvas)
    return full_path


def header_index_from_x(x: int, frame_width: int, segment_count: int) -> int:
    """
    Translate an X pixel coordinate into a header segment index.

    Args:
        x: X coordinate (in pixels) of the selection point.
        frame_width: Total width of the frame.
        segment_count: Number of equally-sized header segments.

    Returns:
        The clamped segment index in range [0, segment_count - 1].
    """
    segment_width = frame_width / float(segment_count)
    index = int(x // segment_width)
    return max(0, min(segment_count - 1, index))


def distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def exponential_smooth(
    previous: Optional[Tuple[float, float]],
    new_point: Tuple[float, float],
    alpha: float,
) -> Tuple[float, float]:
    """
    Apply exponential moving-average smoothing to a 2D point stream to
    remove jitter/shakiness from fingertip tracking.

    Args:
        previous: Previously smoothed point, or None if this is the first sample.
        new_point: The newly observed raw point.
        alpha: Smoothing factor in (0, 1]. 1.0 disables smoothing entirely.

    Returns:
        The newly smoothed point.
    """
    if previous is None:
        return new_point
    smoothed_x = alpha * new_point[0] + (1.0 - alpha) * previous[0]
    smoothed_y = alpha * new_point[1] + (1.0 - alpha) * previous[1]
    return smoothed_x, smoothed_y


def put_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    scale: float = config.FONT_SCALE_MEDIUM,
    color: Tuple[int, int, int] = config.TEXT_COLOR,
    thickness: int = config.FONT_THICKNESS,
) -> None:
    """
    Draw anti-aliased text onto an image in-place, with a subtle drop
    shadow for readability against busy webcam backgrounds.
    """
    shadow_color = (0, 0, 0)
    shadow_offset = 2
    shadow_pos = (position[0] + shadow_offset, position[1] + shadow_offset)
    cv2.putText(image, text, shadow_pos, cv2.FONT_HERSHEY_SIMPLEX, scale, shadow_color, thickness, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_color_swatch(image: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> None:
    """Draw a filled circle with a white border indicating the current draw color."""
    cv2.circle(image, center, radius, color, cv2.FILLED, lineType=cv2.LINE_AA)
    cv2.circle(image, center, radius, config.COLOR_WHITE, 2, lineType=cv2.LINE_AA)

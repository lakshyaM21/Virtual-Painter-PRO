"""
drawing.py

Implements the DrawingCanvas engine: maintains the persistent drawing
layer, an undo stack, brush/eraser strokes, and merges the canvas onto
the live webcam frame using alpha-blended masking so that only the
painted pixels overwrite the video feed (anti-aliased, no visible
canvas background).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

import config
import utils


class DrawingCanvas:
    """
    Manages a persistent drawing layer that is composited on top of the
    live camera feed, along with undo history and header UI overlay.
    """

    def __init__(self, width: int, height: int, max_undo: int = config.MAX_UNDO_STACK) -> None:
        """
        Args:
            width: Canvas width in pixels (matches the camera frame width).
            height: Canvas height in pixels (matches the camera frame height).
            max_undo: Maximum number of undo snapshots retained.
        """
        self._width = width
        self._height = height
        self._canvas: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        self._undo_stack: Deque[np.ndarray] = deque(maxlen=max_undo)
        self._stroke_in_progress: bool = False

    @property
    def canvas(self) -> np.ndarray:
        """The raw drawing layer (black background, colored strokes)."""
        return self._canvas

    def begin_stroke_if_needed(self) -> None:
        """
        Push a snapshot of the canvas onto the undo stack at the start of a
        new stroke. Should be called once per stroke, not once per frame.
        """
        if not self._stroke_in_progress:
            self._undo_stack.append(self._canvas.copy())
            self._stroke_in_progress = True

    def end_stroke(self) -> None:
        """Mark the current stroke as finished so the next draw call snapshots again."""
        self._stroke_in_progress = False

    def draw_line(
        self,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Draw an anti-aliased line segment on the canvas layer.

        Args:
            start_point: Line start (x, y) in pixel coordinates.
            end_point: Line end (x, y) in pixel coordinates.
            color: BGR color tuple. Use config.COLOR_ERASER for eraser mode.
            thickness: Line thickness in pixels.
        """
        cv2.line(self._canvas, start_point, end_point, color, thickness, lineType=cv2.LINE_AA)

    def clear(self) -> None:
        """Clear the entire canvas, saving the previous state for undo."""
        self._undo_stack.append(self._canvas.copy())
        self._canvas = np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def undo(self) -> bool:
        """
        Restore the most recent canvas snapshot from the undo stack.

        Returns:
            True if an undo was performed, False if the undo stack was empty.
        """
        if not self._undo_stack:
            return False
        self._canvas = self._undo_stack.pop()
        return True

    def merge_with_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Composite the drawing layer onto a live camera frame using a binary
        mask derived from the canvas, so that only painted regions obscure
        the underlying video (unpainted canvas stays fully transparent).

        Args:
            frame: The live BGR camera frame to composite onto.

        Returns:
            The merged BGR frame (a new array; `frame` is not modified in place).
        """
        canvas_gray = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, inverse_mask = cv2.threshold(
            canvas_gray, config.CANVAS_BLEND_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )
        inverse_mask_bgr = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)

        frame_with_hole = cv2.bitwise_and(frame, inverse_mask_bgr)
        merged = cv2.bitwise_or(frame_with_hole, self._canvas)
        return merged

    def render_header(self, frame: np.ndarray, header_image: np.ndarray) -> None:
        """
        Overlay the currently selected header image onto the top of the frame.

        Args:
            frame: The frame to draw the header onto, modified in place.
            header_image: Pre-resized header image matching frame width.
        """
        frame[0 : config.HEADER_HEIGHT, 0 : self._width] = header_image

    def render_selection_indicator(
        self,
        frame: np.ndarray,
        index_tip: Tuple[int, int],
        middle_tip: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """
        Draw a visual indicator (rectangle bridging index+middle tips) while
        the user is in selection mode, to give clear visual feedback.

        Args:
            frame: Frame to draw on, modified in place.
            index_tip: Pixel coordinate of the index fingertip.
            middle_tip: Pixel coordinate of the middle fingertip.
            color: Color to render the indicator in.
        """
        cv2.rectangle(
            frame,
            (index_tip[0] - 15, index_tip[1] - 25),
            (middle_tip[0] + 15, middle_tip[1] + 15),
            color,
            cv2.FILLED,
            lineType=cv2.LINE_AA,
        )

    def render_brush_cursor(
        self,
        frame: np.ndarray,
        point: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Draw a small circle at the current draw point to visualize brush size.

        Args:
            frame: Frame to draw on, modified in place.
            point: Pixel coordinate of the fingertip.
            color: Brush/eraser color.
            thickness: Current brush/eraser thickness (determines circle radius).
        """
        radius = max(2, thickness // 2)
        cv2.circle(frame, point, radius, color, cv2.FILLED, lineType=cv2.LINE_AA)

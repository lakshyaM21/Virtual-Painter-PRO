"""
main.py

Entry point for Virtual Painter Pro — a real-time, hand-gesture-controlled
virtual painting application built with OpenCV and the latest MediaPipe
Tasks API (HandLandmarker).

Run with:
    python main.py

Controls (keyboard, window must be focused):
    S   -> Save canvas to Output/
    C   -> Clear canvas
    Z   -> Undo last stroke
    ESC -> Exit

Hand gestures:
    Index finger up              -> Draw
    Index + middle finger up     -> Selection mode (choose header tool)
    Thumb up only                -> Save
    Closed fist                  -> Clear canvas
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
import utils
from drawing import DrawingCanvas
from gesture_detector import GestureDetector, GestureEvent, GestureType
from hand_tracker import HandResult, HandTracker


class VirtualPainterApp:
    """
    Top-level application object. Owns the camera capture, hand tracker,
    gesture detector, drawing canvas and the main event/render loop.
    """

    def __init__(self) -> None:
        """Initialize camera, models, canvas and UI resources."""
        self._capture = self._open_camera()
        frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._frame_width = frame_width
        self._frame_height = frame_height

        self._header_images: List[np.ndarray] = utils.load_header_images(
            config.HEADER_DIR, config.HEADER_IMAGE_FILENAMES, frame_width
        )

        self._hand_tracker = HandTracker()
        self._gesture_detector = GestureDetector()
        self._canvas = DrawingCanvas(frame_width, frame_height)
        self._fps_counter = utils.FPSCounter()

        self._state = config.RuntimeState()
        self._state.draw_color = config.HEADER_INDEX_TO_COLOR[0]

        utils.ensure_output_dir(config.OUTPUT_DIR)

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    @staticmethod
    def _open_camera() -> cv2.VideoCapture:
        """
        Open and configure the webcam capture device.

        Returns:
            A configured `cv2.VideoCapture` instance.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        capture = cv2.VideoCapture(config.CAMERA_INDEX)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {config.CAMERA_INDEX}. "
                "Check that a webcam is connected and not in use by another application."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS_TARGET)
        return capture

    def run(self) -> None:
        """Run the main capture -> process -> render loop until the user exits."""
        try:
            while True:
                success, frame = self._capture.read()
                if not success:
                    print("Warning: failed to read frame from camera. Retrying...")
                    continue

                frame = cv2.flip(frame, 1)  # mirror for a natural "selfie" interaction
                frame = self._process_frame(frame)

                cv2.imshow(config.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key):
                    break
        finally:
            self._shutdown()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run hand tracking + gesture detection for a single frame, update the
        canvas accordingly, and produce the final composited frame with UI.

        Args:
            frame: The raw (already flipped) BGR camera frame.

        Returns:
            The fully composited frame ready for display.
        """
        hands = self._hand_tracker.process(frame)
        primary_hand: Optional[HandResult] = hands[0] if hands else None

        event = self._gesture_detector.detect(primary_hand)
        self._apply_gesture(event, frame)

        merged = self._canvas.merge_with_frame(frame)
        self._canvas.render_header(merged, self._header_images[self._state.header_index])

        if event.gesture_type == GestureType.SELECTION and primary_hand is not None:
            self._canvas.render_selection_indicator(
                merged, primary_hand.index_tip(), primary_hand.middle_tip(), self._state.draw_color
            )
        elif event.gesture_type == GestureType.DRAW and event.point is not None:
            thickness = (
                self._state.eraser_thickness
                if self._state.draw_color == config.COLOR_ERASER
                else self._state.brush_thickness
            )
            self._canvas.render_brush_cursor(merged, event.point, self._state.draw_color, thickness)

        self._render_ui_overlay(merged)
        return merged

    def _apply_gesture(self, event: GestureEvent, frame: np.ndarray) -> None:
        """
        Update application/canvas state in response to the detected gesture.

        Args:
            event: The gesture event produced by the GestureDetector this frame.
            frame: The current camera frame (used only for bounds checks).
        """
        if event.gesture_type == GestureType.SELECTION and event.point is not None:
            self._state.current_mode = "Selection"
            self._canvas.end_stroke()
            self._state.prev_point = None
            self._state.smoothed_point = None

            x, y = event.point
            if y < config.HEADER_HEIGHT:
                new_index = utils.header_index_from_x(x, self._frame_width, config.HEADER_SEGMENT_COUNT)
                self._state.header_index = new_index
                self._state.draw_color = config.HEADER_INDEX_TO_COLOR[new_index]

        elif event.gesture_type == GestureType.DRAW and event.point is not None:
            self._state.current_mode = "Draw" if self._state.draw_color != config.COLOR_ERASER else "Eraser"
            self._handle_draw(event.point)

        elif event.gesture_type == GestureType.SAVE:
            self._state.current_mode = "Save"
            self._canvas.end_stroke()
            self._state.prev_point = None
            self._state.smoothed_point = None
            if event.is_new_action:
                self._save_canvas()

        elif event.gesture_type == GestureType.CLEAR:
            self._state.current_mode = "Clear"
            self._canvas.end_stroke()
            self._state.prev_point = None
            self._state.smoothed_point = None
            if event.is_new_action:
                self._canvas.clear()

        else:
            self._state.current_mode = "Idle"
            self._canvas.end_stroke()
            self._state.prev_point = None
            self._state.smoothed_point = None

    def _handle_draw(self, raw_point: Tuple[int, int]) -> None:
        """
        Apply smoothing to the raw fingertip point and draw a line segment
        from the previous smoothed point to the new one.

        Args:
            raw_point: The unsmoothed fingertip pixel coordinate for this frame.
        """
        x, y = raw_point
        if y < config.HEADER_HEIGHT:
            # Ignore draw gestures while the finger is over the header bar.
            self._state.prev_point = None
            self._state.smoothed_point = None
            return

        smoothed = utils.exponential_smooth(self._state.smoothed_point, (float(x), float(y)), config.SMOOTHING_ALPHA)
        self._state.smoothed_point = smoothed
        current_point = (int(smoothed[0]), int(smoothed[1]))

        if self._state.prev_point is None:
            self._state.prev_point = current_point
            self._canvas.begin_stroke_if_needed()
            return

        if utils.distance(self._state.prev_point, current_point) < config.MIN_DRAW_DISTANCE:
            return

        self._canvas.begin_stroke_if_needed()
        thickness = (
            self._state.eraser_thickness
            if self._state.draw_color == config.COLOR_ERASER
            else self._state.brush_thickness
        )
        self._canvas.draw_line(self._state.prev_point, current_point, self._state.draw_color, thickness)
        self._state.prev_point = current_point

    def _save_canvas(self) -> None:
        """Save the current canvas drawing to the Output/ directory."""
        path = utils.save_canvas(
            self._canvas.canvas, config.OUTPUT_DIR, config.SAVE_FILENAME_PREFIX, config.SAVE_FILE_EXTENSION
        )
        print(f"Canvas saved to: {path}")

    def _render_ui_overlay(self, frame: np.ndarray) -> None:
        """
        Draw FPS counter, current mode, and current color swatch onto the frame.

        Args:
            frame: The frame to annotate, modified in place.
        """
        fps = self._fps_counter.update()
        utils.put_text(frame, f"FPS: {fps:.1f}", config.FPS_TEXT_POSITION, config.FONT_SCALE_SMALL)

        color_name = config.HEADER_INDEX_TO_NAME.get(self._state.header_index, "Custom")
        utils.put_text(frame, f"Mode: {self._state.current_mode}", config.MODE_TEXT_POSITION, config.FONT_SCALE_SMALL)
        utils.put_text(
            frame,
            f"Tool: {color_name}",
            (config.MODE_TEXT_POSITION[0], config.MODE_TEXT_POSITION[1] + 30),
            config.FONT_SCALE_SMALL,
        )

        swatch_color = self._state.draw_color if self._state.draw_color != config.COLOR_ERASER else config.COLOR_GRAY
        utils.draw_color_swatch(frame, config.COLOR_SWATCH_POSITION, config.COLOR_SWATCH_RADIUS, swatch_color)

    def _handle_keyboard(self, key: int) -> bool:
        """
        Process a single keyboard event.

        Args:
            key: The masked key code returned by `cv2.waitKey`.

        Returns:
            False if the application should exit, True to keep running.
        """
        if key == config.KEY_ESC:
            return False

        char = chr(key) if 0 <= key < 256 else ""

        if char.lower() == config.KEY_SAVE:
            self._save_canvas()
        elif char.lower() == config.KEY_CLEAR:
            self._canvas.clear()
        elif char.lower() == config.KEY_UNDO:
            self._canvas.undo()

        return True

    def _shutdown(self) -> None:
        """Release all resources cleanly."""
        self._hand_tracker.close()
        self._capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    """
    Application entry point.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    try:
        app = VirtualPainterApp()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Fatal error during initialization: {exc}", file=sys.stderr)
        return 1

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

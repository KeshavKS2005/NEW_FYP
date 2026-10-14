"""
Stream B: Diagnostic Visualizer and HUD Renderer
Renders annotated video frames with:
1. High-precision known-object bounding boxes, class labels, and confidence tags.
2. Real-time telemetry HUD overlay (FPS, latency, detection count, known class status).
3. Video writer export for end-to-end pipeline demonstration.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from stream_b.config import StreamBFrameResult


class StreamBVisualizer:
    """Renders Stream B annotations and telemetry HUD onto video frames."""

    def __init__(
        self,
        output_path: Optional[str] = None,
        fps: float = 30.0,
        box_color: Tuple[int, int, int] = (255, 180, 0),  # Bright Cyan/Amber (BGR)
        box_thickness: int = 2
    ):
        """
        Args:
            output_path: Optional path to write output video (.mp4).
            fps: Playback frame rate.
            box_color: BGR color for known object bounding boxes.
            box_thickness: Line thickness for boxes.
        """
        self.output_path = output_path
        self.fps = fps
        self.box_color = box_color
        self.box_thickness = box_thickness

        self._writer: Optional[cv2.VideoWriter] = None
        self._target_size: Optional[Tuple[int, int]] = None

    def draw_hud(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        fps: float,
        latency_ms: float,
        known_count: int,
        active_class: str = "BICYCLE"
    ) -> None:
        """Draws a sleek, semi-transparent telemetry card in the top-left."""
        x, y, w, h = 15, 15, 360, 120
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 25), -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 0), 1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        # Header
        cv2.putText(
            canvas,
            "STREAM B: SEMANTICS (KNOWN WORLD)",
            (x + 12, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            1,
            cv2.LINE_AA
        )

        # Telemetry metrics
        fps_text = f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f}ms  |  Frame: #{frame_idx}"
        cv2.putText(
            canvas,
            fps_text,
            (x + 12, y + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        count_text = f"Known Objects: {known_count}  (Class: {active_class.upper()})"
        cv2.putText(
            canvas,
            count_text,
            (x + 12, y + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 255, 150) if known_count > 0 else (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        status_text = "Status: Person & Car -> UNKNOWN (Preserved for Module 3)"
        cv2.putText(
            canvas,
            status_text,
            (x + 12, y + 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 200, 255),
            1,
            cv2.LINE_AA
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: StreamBFrameResult,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        Renders bounding boxes, label pills, and HUD onto the frame.
        """
        annotated = frame.copy()

        for box in result.boxes:
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (box.x1, box.y1),
                (box.x2, box.y2),
                self.box_color,
                self.box_thickness
            )

            # Draw label pill
            label = f"{box.class_name.upper()} {box.confidence * 100:.1f}%"
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1
            )
            pill_y1 = max(0, box.y1 - text_h - 8)
            pill_y2 = box.y1
            pill_x2 = min(annotated.shape[1], box.x1 + text_w + 10)

            # Fill label background
            cv2.rectangle(
                annotated,
                (box.x1, pill_y1),
                (pill_x2, pill_y2),
                self.box_color,
                -1
            )
            # Label text
            cv2.putText(
                annotated,
                label,
                (box.x1 + 5, pill_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 25),
                1,
                cv2.LINE_AA
            )

        # Draw HUD
        active_class = result.boxes[0].class_name if len(result.boxes) > 0 else "BICYCLE"
        self.draw_hud(
            annotated,
            frame_idx=result.frame_index,
            fps=fps,
            latency_ms=result.inference_time_ms,
            known_count=result.known_count,
            active_class=active_class
        )

        return annotated

    def write_frame(self, canvas: np.ndarray) -> None:
        """Writes annotated frame to video writer if configured."""
        if self.output_path is None:
            return

        h, w = canvas.shape[:2]
        if self._writer is None:
            self._target_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

        if (w, h) != self._target_size:
            canvas = cv2.resize(canvas, self._target_size)

        self._writer.write(canvas)

    def release(self) -> None:
        """Flushes and closes the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

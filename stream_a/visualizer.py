"""
Stream A: Diagnostic Visualizer and HUD Renderer
Generates a synchronized composite visualization displaying:
1. Annotated original camera frame with detected moving part bounding boxes and centroids.
2. Raw OpenCV MOG2 mask (showing moving foreground and detected shadows).
3. Cleaned morphological binary motion mask (ready for Module 3 erasure).
4. Real-time telemetry HUD overlay (FPS, frame counter, moving parts count, motion area).
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from stream_a.config import MotionFrameResult


class StreamAVisualizer:
    """Renders composite diagnostic views and handles video export."""

    def __init__(
        self,
        output_path: Optional[str] = None,
        fps: float = 30.0,
        composite_height: int = 540
    ):
        """
        Args:
            output_path: Optional file path to write output video (.mp4).
            fps: Target playback frame rate.
            composite_height: Target height for the composite multi-panel canvas.
        """
        self.output_path = output_path
        self.fps = fps
        self.composite_height = composite_height

        self._writer: Optional[cv2.VideoWriter] = None
        self._target_size: Optional[Tuple[int, int]] = None  # (width, height)

    def draw_hud(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        fps: float,
        blob_count: int,
        motion_pixels: int,
        motion_ratio: float
    ) -> None:
        """Draws a sleek, semi-transparent telemetry card in the top-left."""
        # Backdrop box: x=15, y=15, w=330, h=110
        x, y, w, h = 15, 15, 330, 115
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 25), -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 200, 100), 1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        # Header
        cv2.putText(
            canvas,
            "STREAM A: MOTION DETECTOR",
            (x + 12, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 136),
            2,
            cv2.LINE_AA
        )

        # Telemetry metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        color = (235, 235, 235)

        cv2.putText(
            canvas,
            f"Frame: {frame_idx:05d}   |   Speed: {fps:5.1f} FPS",
            (x + 12, y + 50),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Moving Parts Detected: {blob_count:d}",
            (x + 12, y + 74),
            font,
            scale,
            (50, 220, 255),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Motion Pixels: {motion_pixels:,} ({motion_ratio * 100:.2f}%)",
            (x + 12, y + 98),
            font,
            scale,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

    def render(
        self,
        frame: np.ndarray,
        result: MotionFrameResult,
        current_fps: float = 0.0
    ) -> np.ndarray:
        """
        Builds a composite multi-panel frame:
        [ Main Annotated Frame (960x540) ] | [ Top Right: Raw MOG2 Mask (480x270)   ]
                                           | [ Bottom Right: Clean Mask (480x270)   ]

        Args:
            frame: Original BGR camera frame (H, W, 3).
            result: MotionFrameResult from MOG2MotionExtractor.
            current_fps: Estimated processing frames per second.

        Returns:
            Composite BGR frame (1440x540).
        """
        annotated = frame.copy()

        # Draw detected moving components
        for idx, box in enumerate(result.boxes, start=1):
            # Bounding box
            cv2.rectangle(
                annotated,
                (box.x1, box.y1),
                (box.x2, box.y2),
                (0, 255, 100),  # Bright lime green
                2
            )

            # Centroid dot
            cv2.circle(annotated, box.centroid, 4, (0, 0, 255), -1)

            # Box info tag
            tag = f"Part #{idx} | {int(box.area)}px"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, box.y1 - th - 6)
            cv2.rectangle(
                annotated,
                (box.x1, tag_y1),
                (box.x1 + tw + 6, tag_y1 + th + 6),
                (0, 255, 100),
                -1
            )
            cv2.putText(
                annotated,
                tag,
                (box.x1 + 3, tag_y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (10, 10, 10),
                1,
                cv2.LINE_AA
            )

        # Scale main annotated frame to 960x540
        main_w = int(self.composite_height * 16 / 9)  # 960
        main_h = self.composite_height               # 540
        main_resized = cv2.resize(annotated, (main_w, main_h), interpolation=cv2.INTER_LINEAR)

        # Draw HUD on main view
        self.draw_hud(
            main_resized,
            frame_idx=result.frame_idx,
            fps=current_fps,
            blob_count=len(result.boxes),
            motion_pixels=result.total_motion_pixels,
            motion_ratio=result.motion_ratio
        )

        # Right panel dimensions: 480 x 270 each
        side_w = int(main_w / 2)  # 480
        side_h = int(main_h / 2)  # 270

        # 1. Raw MOG2 Mask Panel (Colorize: Gray=Shadow (127), White=Foreground (255))
        raw_bgr = cv2.cvtColor(result.raw_mask, cv2.COLOR_GRAY2BGR)
        # Highlight shadow pixels in orange/yellow for diagnostic visibility
        shadow_mask = (result.raw_mask == 127)
        raw_bgr[shadow_mask] = (0, 165, 255)  # Orange for shadows
        raw_resized = cv2.resize(raw_bgr, (side_w, side_h), interpolation=cv2.INTER_NEAREST)

        # Header for Raw Mask
        cv2.rectangle(raw_resized, (0, 0), (side_w, 24), (20, 20, 20), -1)
        cv2.putText(
            raw_resized,
            "1. Raw MOG2 Mask (White: Motion, Orange: Shadows)",
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # 2. Cleaned Binary Mask Panel (Bright green motion mask)
        clean_bgr = np.zeros((result.clean_mask.shape[0], result.clean_mask.shape[1], 3), dtype=np.uint8)
        clean_bgr[result.clean_mask > 0] = (0, 255, 120)  # Neon green for clean motion
        clean_resized = cv2.resize(clean_bgr, (side_w, side_h), interpolation=cv2.INTER_NEAREST)

        # Header for Clean Mask
        cv2.rectangle(clean_resized, (0, 0), (side_w, 24), (20, 20, 20), -1)
        cv2.putText(
            clean_resized,
            "2. Cleaned Binary Mask (Shadows Dropped + Morph Filter)",
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 120),
            1,
            cv2.LINE_AA
        )

        # Stack side panels vertically
        right_column = np.vstack([raw_resized, clean_resized])

        # Combine main view and right column horizontally
        composite = np.hstack([main_resized, right_column])

        # Write frame to video if output path is configured
        if self.output_path:
            self._write_frame(composite)

        return composite

    def _write_frame(self, frame: np.ndarray) -> None:
        """Initializes VideoWriter if needed and writes the current frame."""
        h, w = frame.shape[:2]
        if self._writer is None:
            # Try mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                # Fallback to XVID
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

        self._writer.write(frame)

    def close(self) -> None:
        """Releases video writer resources."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

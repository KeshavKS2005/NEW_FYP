"""
Dual-Stream Motion–YOLO Discrepancy Pipeline: End-to-End CLI Runner
Orchestrates:
- Module 1 (Stream A): Class-Agnostic Motion Extraction (OpenCV MOG2)
- Module 2 (Stream B): Known-Object Detection (Ultralytics YOLO)
- Module 3: Mask-Level Erasure & Candidate Extraction (The Discrepancy Engine)
- Module 4: Temporal Trajectory Gating & Active Learning (ByteTrack + Noise Filtering)

Generates 4-panel diagnostic telemetry video (.mp4) and active learning crop datasets.

Usage:
    python run_pipeline.py --video "datasets/test_videos/bike_traffic_montreal.webm" --output "outputs/pipeline_montreal.mp4" --max_frames 120
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

# Ensure workspace root in sys.path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from stream_a.config import MotionConfig
from stream_a.motion_extractor import MOG2MotionExtractor
from stream_b.config import StreamBConfig
from stream_b.detector import YOLOKnownDetector
from module_3.config import Module3Config
from module_3.erasure_engine import MaskErasureEngine
from module_4.config import Module4Config, ConfirmedUnknownTrack
from module_4.temporal_gater import TemporalTrajectoryGater


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-Stream Motion-YOLO Discrepancy Pipeline (Modules 1-4)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="datasets/test_videos/bike_traffic_montreal.webm",
        help="Path to input video file (e.g. datasets/test_videos/... or CCTV 01/...)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/pipeline_demo.mp4",
        help="Path for generated 4-panel diagnostic video (.mp4)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="stream_b/weights/known_single_object.pt",
        help="Path to trained YOLO weights file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Stream B confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--dilation",
        type=float,
        default=0.10,
        help="Module 3 dilation margin (default: 0.10 = 10% padding)"
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=100.0,
        help="Module 3 minimum contour area in pixels (default: 100.0)"
    )
    parser.add_argument(
        "--merge_proximity",
        type=int,
        default=20,
        help="Proximity threshold in pixels to merge nearby fragmented candidate boxes (default: 20)"
    )
    parser.add_argument(
        "--no_morph_close",
        action="store_true",
        help="Disable morphological closing on Module 3 residual mask"
    )
    parser.add_argument(
        "--survival_window",
        type=int,
        default=5,
        help="Module 4 frame survival window W (default: 5)"
    )
    parser.add_argument(
        "--no_active_learning",
        action="store_true",
        help="Disable Active Learning dataset auto-crop hook"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: entire video)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Compute device for YOLO: '0' for CUDA GPU, 'cpu' for CPU"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display live cv2 preview window"
    )
    return parser.parse_args()


class PipelineVisualizer:
    """Renders a diagnostic 4-panel dashboard combining all 4 modules."""

    PANEL_W = 640
    PANEL_H = 360

    COLOR_KNOWN = (255, 220, 0)       # Cyan/Yellow-Green for Known (Bicycle)
    COLOR_UNKNOWN = (50, 50, 255)     # Bright Red for Confirmed Unknown
    COLOR_CANDIDATE = (0, 200, 255)   # Amber/Orange for Module 3 Candidates
    COLOR_TRAIL = (180, 100, 255)     # Magenta/Purple for Trajectory trails
    COLOR_BG = (20, 24, 30)           # Dark charcoal for HUD panel

    def render_panel1_detections(
        self,
        frame: np.ndarray,
        known_boxes: list,
        confirmed_tracks: List[ConfirmedUnknownTrack]
    ) -> np.ndarray:
        """Top-Left: Original RGB with Known (Stream B) and Confirmed Unknown (Module 4) boxes."""
        canvas = cv2.resize(frame, (self.PANEL_W, self.PANEL_H))
        scale_x = self.PANEL_W / frame.shape[1]
        scale_y = self.PANEL_H / frame.shape[0]

        # Draw Known Bounding Boxes (Cyan)
        for kb in known_boxes:
            kx1, ky1, kx2, ky2 = (
                int(kb.x1 * scale_x), int(kb.y1 * scale_y),
                int(kb.x2 * scale_x), int(kb.y2 * scale_y)
            )
            cv2.rectangle(canvas, (kx1, ky1), (kx2, ky2), self.COLOR_KNOWN, 2)
            label = f"KNOWN: {kb.class_name.upper()} {kb.confidence:.2f}"
            cv2.putText(canvas, label, (kx1, max(14, ky1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_KNOWN, 1, cv2.LINE_AA)

        # Draw Confirmed Unknown Bounding Boxes & Trajectory (Red)
        for trk in confirmed_tracks:
            ux1, uy1, ux2, uy2 = (
                int(trk.x1 * scale_x), int(trk.y1 * scale_y),
                int(trk.x2 * scale_x), int(trk.y2 * scale_y)
            )
            cv2.rectangle(canvas, (ux1, uy1), (ux2, uy2), self.COLOR_UNKNOWN, 2)
            label = f"UNKNOWN #{trk.track_id:02d} [Age={trk.age}]"
            cv2.putText(canvas, label, (ux1, max(14, uy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_UNKNOWN, 1, cv2.LINE_AA)

            # Draw trajectory trail
            if len(trk.trajectory) > 1:
                pts = np.array([
                    [int(p[0] * scale_x), int(p[1] * scale_y)] for p in trk.trajectory[-25:]
                ], dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=False, color=self.COLOR_TRAIL, thickness=2)

        # Header tag
        self._draw_header(canvas, "1. RGB TRACKING HUD (Known vs. Unknown)")
        return canvas

    def render_panel2_motion(self, clean_mask: np.ndarray) -> np.ndarray:
        """Top-Right: Stream A Binary Motion Mask."""
        resized_mask = cv2.resize(clean_mask, (self.PANEL_W, self.PANEL_H))
        canvas = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        self._draw_header(canvas, "2. STREAM A: Class-Agnostic Motion Mask (MOG2)")
        return canvas

    def render_panel3_erasure(
        self,
        erased_mask: np.ndarray,
        candidate_boxes: list
    ) -> np.ndarray:
        """Bottom-Left: Module 3 Discrepancy Mask with Holes Punched & Candidate Boxes."""
        resized_mask = cv2.resize(erased_mask, (self.PANEL_W, self.PANEL_H))
        canvas = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)

        scale_x = self.PANEL_W / erased_mask.shape[1] if erased_mask.shape[1] > 0 else 1.0
        scale_y = self.PANEL_H / erased_mask.shape[0] if erased_mask.shape[0] > 0 else 1.0

        for cb in candidate_boxes:
            cx1, cy1, cx2, cy2 = (
                int(cb.x1 * scale_x), int(cb.y1 * scale_y),
                int(cb.x2 * scale_x), int(cb.y2 * scale_y)
            )
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), self.COLOR_CANDIDATE, 1)

        self._draw_header(canvas, "3. MODULE 3: Discrepancy Residual Mask (Erased Knowns)")
        return canvas

    def render_panel4_telemetry(
        self,
        frame_idx: int,
        fps: float,
        stream_a_pixels: int,
        stream_b_known: int,
        module_3_erased: int,
        module_3_candidates: int,
        module_4_confirmed: int,
        module_4_total_active: int,
        saved_crops_count: int
    ) -> np.ndarray:
        """Bottom-Right: Pipeline Statistics & Telemetry HUD."""
        canvas = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)
        canvas[:] = self.COLOR_BG

        self._draw_header(canvas, "4. SYSTEM TELEMETRY & ACTIVE LEARNING")

        lines = [
            f"Pipeline Rate:        {fps:.1f} FPS",
            f"Frame Number:         {frame_idx}",
            "--------------------------------------------",
            f"Stream A Motion Pixels: {stream_a_pixels:,}",
            f"Stream B Known Objects: {stream_b_known} (Bicycles)",
            f"Module 3 Erased Boxes:  {module_3_erased}",
            f"Module 3 Candidates:    {module_3_candidates}",
            "--------------------------------------------",
            f"Module 4 Confirmed:     {module_4_confirmed} (Unknown Moving)",
            f"Module 4 Total Tracks:  {module_4_total_active}",
            f"Active Learning Crops:  {saved_crops_count} saved to disk"
        ]

        y_offset = 45
        for line in lines:
            color = (220, 220, 220)
            if "Confirmed:" in line:
                color = (100, 100, 255)
            elif "Known Objects:" in line:
                color = (255, 220, 0)
            elif "Active Learning" in line:
                color = (100, 255, 120)

            cv2.putText(canvas, line, (25, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y_offset += 24

        return canvas

    def compose_quad_view(
        self,
        panel1: np.ndarray,
        panel2: np.ndarray,
        panel3: np.ndarray,
        panel4: np.ndarray
    ) -> np.ndarray:
        """Combines the 4 panels into a unified 1280x720 canvas."""
        top_row = np.hstack([panel1, panel2])
        bot_row = np.hstack([panel3, panel4])
        return np.vstack([top_row, bot_row])

    @staticmethod
    def _draw_header(canvas: np.ndarray, text: str) -> None:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (10, 14, 20), -1)
        cv2.putText(canvas, text, (10, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"[ERROR] Input video not found at: {video_path}")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Dual-Stream Discrepancy Pipeline: Modules 1 to 4")
    print(f"Input Video:   {video_path}")
    print(f"Output Video:  {out_path}")
    print(f"Model Weights: {args.weights}")
    print(f"YOLO Conf:     {args.conf:.2f}")
    print(f"Erasure Dilation Margin: {args.dilation * 100:.0f}%")
    print(f"Min Contour Area:        {args.min_area} px")
    print(f"Box Proximity Merging:   {args.merge_proximity} px (morph_close={not args.no_morph_close})")
    print(f"Survival Window:         {args.survival_window} frames")
    print(f"Active Learning Hook:    {not args.no_active_learning}")
    print("=" * 65)

    # 1. Initialize Pipeline Modules
    stream_a = MOG2MotionExtractor(MotionConfig(
        history=500, var_threshold=16.0, detect_shadows=True, min_contour_area=args.min_area
    ))
    stream_b = YOLOKnownDetector(StreamBConfig(
        model_path=args.weights, conf_threshold=args.conf, device=args.device, known_classes=["bicycle"]
    ))
    module_3 = MaskErasureEngine(Module3Config(
        dilation_margin=args.dilation,
        min_contour_area=args.min_area,
        morph_close=not args.no_morph_close,
        merge_proximity_px=args.merge_proximity
    ))
    module_4 = TemporalTrajectoryGater(Module4Config(
        survival_window=args.survival_window,
        active_learning=not args.no_active_learning,
        active_learning_frame_thresh=30,
        active_learning_crop_dir="outputs/active_learning_crops"
    ))

    visualizer = PipelineVisualizer()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_w = visualizer.PANEL_W * 2
    output_h = visualizer.PANEL_H * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, video_fps, (output_w, output_h))

    processed_frames = 0
    t_start = time.perf_counter()
    saved_crops_total = 0

    metrics_log = {
        "video": str(video_path),
        "total_frames_processed": 0,
        "total_known_detected": 0,
        "total_candidate_blobs": 0,
        "total_confirmed_unknowns": 0,
        "active_learning_crops_saved": 0,
        "average_fps": 0.0
    }

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            if args.max_frames and processed_frames > args.max_frames:
                break

            t0 = time.perf_counter()

            # Module 1: Stream A Motion Extraction
            res_a = stream_a.process(frame)

            # Module 2: Stream B Known-Object Semantics
            res_b = stream_b.detect(frame, frame_index=processed_frames)

            # Module 3: Mask-Level Erasure & Candidate Extraction
            res_3 = module_3.process(
                motion_mask=res_a.clean_mask,
                known_boxes=res_b.boxes,
                frame_index=processed_frames
            )

            # Module 4: Temporal Trajectory Gating & Tracking
            res_4 = module_4.update(
                candidate_boxes=res_3.box_matrix,
                frame=frame,
                frame_index=processed_frames
            )

            step_time = time.perf_counter() - t0
            current_fps = 1.0 / max(1e-5, step_time)

            if res_4.saved_crops:
                saved_crops_total += len(res_4.saved_crops)

            # Update metrics
            metrics_log["total_known_detected"] += res_b.known_count
            metrics_log["total_candidate_blobs"] += res_3.candidate_count
            metrics_log["total_confirmed_unknowns"] += res_4.confirmed_count

            # Render 4 panels
            p1 = visualizer.render_panel1_detections(frame, res_b.boxes, res_4.confirmed_tracks)
            p2 = visualizer.render_panel2_motion(res_a.clean_mask)
            p3 = visualizer.render_panel3_erasure(res_3.erased_mask, res_3.boxes)
            p4 = visualizer.render_panel4_telemetry(
                frame_idx=processed_frames,
                fps=current_fps,
                stream_a_pixels=res_a.total_motion_pixels,
                stream_b_known=res_b.known_count,
                module_3_erased=res_3.erased_known_count,
                module_3_candidates=res_3.candidate_count,
                module_4_confirmed=res_4.confirmed_count,
                module_4_total_active=len(res_4.all_active_tracks),
                saved_crops_count=saved_crops_total
            )

            quad_view = visualizer.compose_quad_view(p1, p2, p3, p4)
            writer.write(quad_view)

            if args.preview:
                cv2.imshow("Dual-Stream Discrepancy Pipeline", quad_view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if processed_frames % 25 == 0 or processed_frames == 1:
                print(
                    f"Frame [{processed_frames:04d}/{total_video_frames}] | "
                    f"FPS: {current_fps:4.1f} | "
                    f"Known (Bikes): {res_b.known_count} | "
                    f"Candidates: {res_3.candidate_count} | "
                    f"Confirmed UNKNOWN: {res_4.confirmed_count} | "
                    f"Crops: {saved_crops_total}"
                )

    finally:
        cap.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    total_duration = time.perf_counter() - t_start
    avg_fps = processed_frames / max(1e-4, total_duration)

    metrics_log["total_frames_processed"] = processed_frames
    metrics_log["active_learning_crops_saved"] = saved_crops_total
    metrics_log["average_fps"] = round(avg_fps, 2)

    metrics_path = out_path.parent / f"{out_path.stem}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    print("=" * 65)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Total Frames Processed: {processed_frames}")
    print(f"Average Pipeline Speed: {avg_fps:.1f} FPS")
    print(f"Total Confirmed Unknown Hits: {metrics_log['total_confirmed_unknowns']}")
    print(f"Active Learning Crops Saved:  {saved_crops_total}")
    print(f"Diagnostic Video Saved:       {out_path}")
    print(f"Metrics Saved:                {metrics_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

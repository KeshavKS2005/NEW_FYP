"""
Stream B: CLI Video Runner
Processes video footage using Stream B (Known-Object Semantic Detection via YOLO)
and renders diagnostic video with known bounding boxes and telemetry HUD.

Usage:
    python stream_b/run_stream_b.py --video "CCTV 01/VIRAT_S_010001_00_000019_000065.mp4" --output "outputs/stream_b_demo.mp4"
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from stream_b.config import StreamBConfig
from stream_b.detector import YOLOKnownDetector
from stream_b.visualizer import StreamBVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream B: Known-Object Detection on Surveillance Video"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="CCTV 01/VIRAT_S_010001_00_000019_000065.mp4",
        help="Path to input video file (e.g. CCTV 01/...)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/stream_b_demo.mp4",
        help="Path for generated diagnostic video (.mp4)"
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
        default=0.40,
        help="Confidence threshold for known object detection (default: 0.40)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Compute device: '0' for CUDA GPU, 'cpu' for CPU"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process (default: entire video)"
    )
    parser.add_argument(
        "--snapshot_interval",
        type=int,
        default=50,
        help="Frame interval for saving snapshot preview images (default: 50)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"[ERROR] Input video not found at: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = Path("outputs") / "snapshots_stream_b"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 60)
    print("STREAM B: KNOWN OBJECT DETECTOR (CLI RUNNER)")
    print(f"Input Video:      {video_path} ({width}x{height} @ {video_fps:.1f} FPS, {total_video_frames} frames)")
    print(f"Output Video:     {args.output}")
    print(f"Model Weights:    {args.weights}")
    print(f"Conf Threshold:   {args.conf}")
    print(f"Device:           {args.device}")
    print("=" * 60)

    config = StreamBConfig(
        model_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        known_classes=["bicycle"]
    )

    detector = YOLOKnownDetector(config=config)
    visualizer = StreamBVisualizer(
        output_path=args.output,
        fps=video_fps
    )

    frame_idx = 0
    start_time = time.time()
    total_detections = 0
    total_inference_ms = 0.0

    target_frames = min(args.max_frames, total_video_frames) if args.max_frames else total_video_frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            # Process frame with Stream B
            result = detector.detect(frame, frame_index=frame_idx)
            total_detections += result.known_count
            total_inference_ms += result.inference_time_ms

            # Compute smoothed FPS
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0.0

            # Render visualization
            annotated = visualizer.annotate_frame(frame, result, fps=current_fps)
            visualizer.write_frame(annotated)

            # Periodically save diagnostic snapshots
            if frame_idx % args.snapshot_interval == 0 or frame_idx == 1:
                snap_path = snapshot_dir / f"stream_b_frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(snap_path), annotated)

            # Progress log
            if frame_idx % 25 == 0 or frame_idx == target_frames:
                print(
                    f"Frame [{frame_idx:04d}/{target_frames}] "
                    f"FPS: {current_fps:5.1f} | "
                    f"Latency: {result.inference_time_ms:5.1f}ms | "
                    f"Known Objects: {result.known_count}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user.")
    finally:
        cap.release()
        visualizer.release()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0
    avg_latency = total_inference_ms / frame_idx if frame_idx > 0 else 0.0

    print("=" * 60)
    print("STREAM B PROCESSING COMPLETE")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Elapsed Time:     {total_time:.2f}s")
    print(f"Average Pipeline FPS:   {avg_fps:.1f}")
    print(f"Average Model Latency:  {avg_latency:.1f}ms")
    print(f"Total Known Detections: {total_detections}")
    print(f"Diagnostic Video Saved: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

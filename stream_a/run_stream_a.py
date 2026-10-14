"""
Stream A: CLI Runner
Processes video files using Stream A (MOG2 Motion Extraction) and outputs
diagnostic video with moving parts bounding boxes and telemetry.

Usage:
    python stream_a/run_stream_a.py --video "CCTV 01/VIRAT_S_010001_00_000019_000065.mp4" --output "outputs/stream_a_demo.mp4"
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2

# Add workspace root to sys.path so stream_a can be imported directly
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from stream_a.config import MotionConfig
from stream_a.motion_extractor import MOG2MotionExtractor
from stream_a.visualizer import StreamAVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream A: Class-Agnostic Motion Extraction on Surveillance Video"
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
        default="outputs/stream_a_demo.mp4",
        help="Path for generated multi-view diagnostic video (.mp4)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: entire video)"
    )
    parser.add_argument(
        "--history",
        type=int,
        default=500,
        help="MOG2 background history length (default: 500)"
    )
    parser.add_argument(
        "--var_threshold",
        type=float,
        default=16.0,
        help="MOG2 variance threshold for background subtraction (default: 16.0)"
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=200.0,
        help="Minimum contour area in pixels to register as a moving part (default: 200.0)"
    )
    parser.add_argument(
        "--no_shadows",
        action="store_true",
        help="Disable shadow detection in MOG2"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open an interactive cv2.imshow GUI window (Press 'q' to quit, Space to pause)"
    )
    parser.add_argument(
        "--box_pad_ratio",
        type=float,
        default=0.25,
        help="Expansion ratio for bounding boxes (default: 0.25 = 25% bigger)"
    )
    parser.add_argument(
        "--box_pad_px",
        type=int,
        default=10,
        help="Minimum pixel padding added on each side of bounding boxes (default: 10)"
    )
    parser.add_argument(
        "--merge_boxes",
        action="store_true",
        default=False,
        help="Enable merging of overlapping bounding boxes (default: False, boxes stay separate)"
    )
    parser.add_argument(
        "--save_snapshots",
        action="store_true",
        default=True,
        help="Save snapshot images of key frames to outputs/snapshots/ (default: True)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        # Try resolving relative to workspace root
        video_path = workspace_root / args.video
        if not video_path.is_file():
            print(f"[Error] Video file not found: {args.video}", file=sys.stderr)
            sys.exit(1)

    # Ensure output directories exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_dir = output_path.parent / "snapshots"
    if args.save_snapshots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Error] Failed to open video file: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"==================================================")
    print(f" STREAM A: CLASS-AGNOSTIC MOTION EXTRACTION")
    print(f"==================================================")
    print(f"Input video       : {video_path.name}")
    print(f"Resolution        : {width}x{height}")
    print(f"Video FPS         : {fps:.2f}")
    print(f"Total Video Frames: {total_frames}")
    print(f"Diagnostic Output : {output_path}")
    print(f"Parameters        : history={args.history}, varThreshold={args.var_threshold}, minArea={args.min_area}")
    print(f"Box Inflation     : +{int(args.box_pad_ratio * 100)}% (min +{args.box_pad_px}px) | Combine Overlap: {args.merge_boxes}")
    print(f"==================================================")

    # Configure Motion Extractor
    config = MotionConfig(
        history=args.history,
        var_threshold=args.var_threshold,
        detect_shadows=not args.no_shadows,
        min_contour_area=args.min_area,
        box_pad_ratio=args.box_pad_ratio,
        box_pad_px=args.box_pad_px,
        merge_overlapping_boxes=args.merge_boxes
    )
    extractor = MOG2MotionExtractor(config=config)

    # Configure Visualizer
    visualizer = StreamAVisualizer(
        output_path=str(output_path),
        fps=fps,
        composite_height=540
    )

    frame_idx = 0
    start_time = time.time()
    last_log_time = start_time
    total_parts_detected = 0
    max_parts_in_frame = 0

    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if args.max_frames and frame_idx > args.max_frames:
                    break

                # Process frame with Stream A Motion Extractor
                result = extractor.process(frame)

                # Compute rolling FPS
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else 0.0

                num_boxes = len(result.boxes)
                total_parts_detected += num_boxes
                if num_boxes > max_parts_in_frame:
                    max_parts_in_frame = num_boxes

                # Render composite visualization
                composite = visualizer.render(
                    frame=frame,
                    result=result,
                    current_fps=current_fps
                )

                # Save periodic snapshots (e.g. frame 20, 50, 100, 200, or when moving parts detected)
                if args.save_snapshots and (
                    frame_idx in [25, 50, 100, 150, 200, 250, 300] or
                    (num_boxes >= 2 and frame_idx % 40 == 0)
                ):
                    snap_path = snapshot_dir / f"stream_a_frame_{frame_idx:05d}.jpg"
                    cv2.imwrite(str(snap_path), composite)

                # Periodic terminal progress report
                now = time.time()
                if now - last_log_time >= 1.5 or frame_idx == 1:
                    pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    print(
                        f"Frame {frame_idx:04d}/{total_frames} ({pct:5.1f}%) | "
                        f"Speed: {current_fps:5.1f} FPS | "
                        f"Moving Parts: {num_boxes:2d} | "
                        f"Motion Area: {result.total_motion_pixels:6d} px"
                    )
                    last_log_time = now

            if args.preview:
                cv2.imshow("Stream A: Motion Extraction Preview", composite)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord('q'):
                    print("[User] Quit requested via preview window.")
                    break
                elif key == 32:  # Space bar to toggle pause
                    paused = not paused

    except KeyboardInterrupt:
        print("\n[Interrupted] Processing stopped by user.")
    finally:
        cap.release()
        visualizer.close()
        if args.preview:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0

    print(f"\n==================================================")
    print(f" STREAM A PROCESSING COMPLETE")
    print(f"==================================================")
    print(f"Frames processed     : {frame_idx}")
    print(f"Total time taken     : {total_time:.2f} seconds")
    print(f"Average speed        : {avg_fps:.2f} FPS")
    print(f"Peak moving parts    : {max_parts_in_frame} in a single frame")
    print(f"Diagnostic video saved: {output_path}")
    if args.save_snapshots:
        print(f"Snapshots saved to   : {snapshot_dir}")
    print(f"==================================================")


if __name__ == "__main__":
    main()

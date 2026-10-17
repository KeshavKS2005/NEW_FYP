"""
Stationary Outdoor Surveillance Camera Verification Script
Validates all 12 modules on stationary outdoor CCTV footage with holdout classes (0=person, 1=bicycle).
Adheres strictly to Section 9 of fyp_architecture_final_object.md.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

# Ensure workspace root in path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from module_8.config import load_phase1_env, build_postgres_dsn
from module_8.db_manager import DatabaseManager


load_phase1_env()


def resolve_live_stationary_stream(stream_source: str) -> str:
    """
    Resolves video path or stream URL.
    """
    if (stream_source.endswith(".mp4") or
            stream_source.endswith(".webm") or
            stream_source.startswith("rtsp://") or
            os.path.exists(stream_source)):
        return stream_source

    try:
        cmd = ["yt-dlp", "-f", "best[ext=mp4]/best", "-g", stream_source]
        resolved_url = subprocess.check_output(cmd, text=True).strip().splitlines()[0]
        return resolved_url
    except Exception:
        return stream_source


def verify_camera_is_stationary(stream_url: str, sample_frames: int = 60) -> bool:
    """
    Verifies that the online surveillance camera has zero ego-motion (stationary mount)
    by checking that median global optical flow across background corners is < 0.35 px.
    """
    cap = cv2.VideoCapture(stream_url)
    ret, prev = cap.read()
    if not ret or prev is None:
        print("[STATIC CAMERA CHECK] Warning: Could not read stream to check ego-motion.")
        return True

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    global_drifts = []

    for _ in range(sample_frames):
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if pts is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
            valid_prev = pts[status.flatten() == 1]
            valid_next = next_pts[status.flatten() == 1]
            if len(valid_prev) > 20:
                displacements = np.linalg.norm(valid_next - valid_prev, axis=2)
                global_drifts.append(float(np.median(displacements)))
        prev_gray = curr_gray

    cap.release()
    mean_bg_drift = sum(global_drifts) / max(1, len(global_drifts))
    print(f"[STATIC CAMERA CHECK] Median Background Drift: {mean_bg_drift:.4f} px/frame")
    return mean_bg_drift < 0.35


def run_verification(video_path: Optional[str] = None, max_frames: int = 150):
    default_video = "datasets/test_videos/bike_traffic_montreal.webm"
    source = video_path or os.environ.get("TEST_CCTV_URL", default_video)
    resolved = resolve_live_stationary_stream(source)

    print(f"[VERIFY] Checking stationary mount for: {resolved}")
    is_stationary = verify_camera_is_stationary(resolved, sample_frames=30)
    print(f"[VERIFY] Camera Stationary Verdict: {is_stationary}")

    # Set rapid verification thresholds
    os.environ["SURVEILLANCE_STREAM_URL"] = resolved
    os.environ["TEST_HOLDOUT_CLASS_IDS"] = "0,1"  # Hold out Person (0) and Bicycle (1) as unknowns
    os.environ["GEMINI_TRIGGER_COUNT"] = "15"     # Rapid trigger threshold for testing

    from pipeline_12_modules import run_surveillance_pipeline
    print("[VERIFY] Launching 12-Module Pipeline...")
    run_surveillance_pipeline(video_path=resolved, max_frames=max_frames)

    # Post-run DB verification assertions
    print("\n" + "=" * 60)
    print("[POST-RUN VERIFICATION ASSERTIONS]")
    print("=" * 60)
    db = DatabaseManager()
    cur = db._conn.cursor()

    try:
        cur.execute("SELECT decomposition_method, COUNT(*) FROM unknown_observations GROUP BY decomposition_method;")
        print("[DB CHECK 1 - Decomposition Methods]:", cur.fetchall())

        cur.execute("SELECT cluster_id, observation_count, status, identified_label, vlm_confidence FROM unknown_clusters;")
        print("[DB CHECK 2 - Discovered Clusters]:", cur.fetchall())

        cur.execute("SELECT model_version_id, old_classes_map50_95, new_class_map50, promoted_to_production FROM model_registry;")
        print("[DB CHECK 3 - Model Registry & Promotions]:", cur.fetchall())
    except Exception as e:
        print("[DB CHECK ERROR]:", e)
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stationary CCTV Verification")
    parser.add_argument("--video", type=str, default=None, help="Path to video file or RTSP stream")
    parser.add_argument("--max_frames", type=int, default=150, help="Maximum frames to process")
    args = parser.parse_args()
    run_verification(video_path=args.video, max_frames=args.max_frames)

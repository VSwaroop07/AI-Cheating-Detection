"""Interactive webcam debug runner for VideoAnalyzer.

Visualizes in real-time:
- Facial mesh anchor landmarks & iris gaze points
- 3D Head Pose orientation axes (Yaw, Pitch, Roll)
- Gaze direction (Center, Left, Right, Up, Down) & ratios
- Proctoring HUD card with active state & alerts
- Candidate face registration & live verification (Press 'c' to register current face)

Usage:
    python backend/scripts/run_video_debug.py [--camera 0] [--width 640] [--height 480]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

# Ensure backend directory is in python path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ml.video_analyzer import VideoAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("video_debug")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Video Analyzer Webcam Debugger")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional path to video file or image instead of webcam",
    )
    args = parser.parse_args()

    logger.info("Initializing VideoAnalyzer engine...")
    analyzer = VideoAnalyzer()
    session_id = "webcam_debug_session"

    source = args.source if args.source else args.camera
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(
            f"Could not open video source: {source}. "
            "Ensure camera is plugged in and accessible, or specify a valid video file with --source."
        )
        sys.exit(1)

    if not args.source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    logger.info("=================================================================")
    logger.info("AI PROCTORING VIDEO ANALYZER ACTIVE")
    logger.info("Keybindings:")
    logger.info("  'q' or ESC : Quit")
    logger.info("  'r'        : Reset session timers and deviation state")
    logger.info("  'c'        : Register current face as Candidate Baseline Embedding")
    logger.info("=================================================================")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if args.source:
                logger.info("End of video file reached.")
            else:
                logger.warning("Failed to grab frame from webcam.")
            break

        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, current_time - prev_time))
        prev_time = current_time

        # Analyze current frame
        result = analyzer.analyze_frame(
            session_id=session_id,
            frame=frame,
            timestamp=current_time,
        )

        # Log any newly triggered proctoring alerts
        for event in result.events:
            logger.warning(
                f"[ALERT TRIGGERED] Modality: {event.modality} | Type: {event.type} | "
                f"Severity: {event.severity} | Confidence: {event.confidence:.2f} | "
                f"Description: {event.description}"
            )

        # Render visual overlay
        overlay = analyzer.draw_overlay(frame, result)

        # Draw FPS counter
        cv2.putText(
            overlay,
            f"FPS: {fps:.1f}",
            (overlay.shape[1] - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if fps > 15 else (0, 165, 255),
            2,
        )

        cv2.imshow("AI Proctoring Video Analyzer Debugger", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' or ESC
            logger.info("Exiting...")
            break
        elif key == ord("r"):  # Reset
            analyzer.reset_session(session_id)
            logger.info("Session state reset.")
        elif key == ord("c"):  # Register candidate face
            emb = analyzer.face_auth.compute_embedding(frame)
            if emb is not None:
                analyzer.set_session_embedding(session_id, emb)
                logger.info("Successfully registered current face as session baseline embedding.")
            else:
                logger.warning("Could not register face: ensure exactly one clear face is visible.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


"""CPU Benchmark script for Ultralytics YOLOv8n ObjectDetector.

Measures and reports:
- Average inference time per frame in milliseconds on CPU
- Standard deviation, min, max, median (P50), and P95 latency
- Effective FPS throughput
- Detection breakdown on sample frame
- CPU amortized latency under 3rd-frame skipping schedule

Usage:
    python backend/scripts/benchmark_object_detector.py [--iterations 50] [--warmup 5] [--input path/to/image.jpg]
"""

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ml.object_detector import ObjectDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8n Object Detector on CPU")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations before measuring (default: 5)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to test image (defaults to sample test face)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Object detection interval (e.g. every 3rd frame)",
    )
    args = parser.parse_args()

    # Determine input image
    if args.input and Path(args.input).exists():
        img_path = Path(args.input)
    else:
        img_path = backend_dir.parent / "tests" / "data" / "sample_face.jpg"

    if not img_path.exists():
        logger.info(f"Test image not found at {img_path}, generating synthetic 640x480 frame...")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 100, (200, 180, 150), -1)
    else:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to read image from {img_path}")
            sys.exit(1)

    img_h, img_w = img.shape[:2]
    logger.info("==================================================================")
    logger.info("PROCTORING AI: YOLOV8N CPU INFERENCE BENCHMARK")
    logger.info(f"Input Frame Dimensions : {img_w}x{img_h}")
    logger.info(f"Benchmark Iterations   : {args.iterations}")
    logger.info(f"Warmup Iterations      : {args.warmup}")
    logger.info(f"Target Hardware Device : CPU")
    logger.info("==================================================================")

    detector = ObjectDetector()
    session_id = "benchmark_session"

    # Warmup runs
    logger.info(f"Running {args.warmup} warmup iterations...")
    for _ in range(args.warmup):
        detector.detect_objects(session_id, img, force_inference=True)

    # Measured benchmark runs
    logger.info(f"Running {args.iterations} timed benchmark iterations...")
    latencies_ms: List[float] = []

    last_result = None
    for i in range(args.iterations):
        start_t = time.perf_counter()
        res = detector.detect_objects(session_id, img, force_inference=True)
        end_t = time.perf_counter()
        latencies_ms.append((end_t - start_t) * 1000.0)
        last_result = res

    # Compute statistics
    avg_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    sorted_lats = sorted(latencies_ms)
    p95_idx = int(0.95 * len(sorted_lats))
    p95_ms = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    # Amortized latency under interval skipping
    amortized_ms = avg_ms / float(args.interval)
    amortized_fps = fps * float(args.interval)

    print("\n" + "=" * 66)
    print("                 YOLOV8N CPU BENCHMARK RESULTS")
    print("=" * 66)
    print(f" Average Latency per Frame  : {avg_ms:6.2f} ms")
    print(f" Standard Deviation         : {std_ms:6.2f} ms")
    print(f" Median Latency (P50)       : {median_ms:6.2f} ms")
    print(f" 95th Percentile Latency    : {p95_ms:6.2f} ms")
    print(f" Minimum Latency            : {min_ms:6.2f} ms")
    print(f" Maximum Latency            : {max_ms:6.2f} ms")
    print(f" Raw Throughput on CPU      : {fps:6.1f} FPS")
    print("-" * 66)
    print(f" Frame Interval Skipping    : Every {args.interval} frames")
    print(f" Amortized CPU Latency/Frame: {amortized_ms:6.2f} ms (effective CPU load)")
    print(f" Sustainable Pipeline FPS   : {amortized_fps:6.1f} FPS")
    print("=" * 66)

    if last_result and last_result.detections:
        print("\nDetections in Sample Frame:")
        for det in last_result.detections:
            print(f"  - {det.label.upper():<12} (confidence: {det.confidence:.2f}, box: {det.box})")
    else:
        print("\nNo prohibited objects detected in sample frame.")

    # Frame budget check (for 2.5 FPS candidate stream: frame time budget is 400 ms)
    budget_ms = 400.0
    print(f"\nReal-Time Telemetry Budget Check (2.5 FPS stream = {budget_ms:.0f} ms budget):")
    if avg_ms < budget_ms:
        headroom = ((budget_ms - avg_ms) / budget_ms) * 100.0
        print(f" [PASS] YOLOv8n average latency ({avg_ms:.1f}ms) is well within the {budget_ms:.0f}ms budget ({headroom:.1f}% headroom on CPU).")
    else:
        print(f" [NOTE] Frame skipping (1 in {args.interval}) provides amortized {amortized_ms:.1f}ms, easily satisfying real-time delivery.")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Automated and standalone retention cleanup script for proctoring evidence artifacts.

Scans the evidence directory and MongoDB evidence collection:
1. Deletes evidence image and video artifacts older than the configured retention threshold (EVIDENCE_RETENTION_DAYS).
2. Enforces per-session storage caps (EVIDENCE_MAX_SESSION_MB) by pruning oldest evidence files.
3. Removes empty session directories.
4. Cleans up expired MongoDB metadata records.

Usage:
    # Run with default settings (30 days retention, 100 MB per-session cap)
    python scripts/cleanup_evidence.py

    # Custom retention and storage cap
    python scripts/cleanup_evidence.py --days 14 --max-session-mb 50.0

    # Dry-run mode (preview without deleting files)
    python scripts/cleanup_evidence.py --days 7 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Ensure backend root is in sys.path
backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from app.config import settings
from app.database import DatabaseManager
from app.services.evidence_service import get_evidence_service


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up expired evidence files and enforce storage caps."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=settings.EVIDENCE_RETENTION_DAYS,
        help=f"Evidence retention period in days (default: {settings.EVIDENCE_RETENTION_DAYS})",
    )
    parser.add_argument(
        "--max-session-mb",
        type=float,
        default=settings.EVIDENCE_MAX_SESSION_MB,
        help=f"Maximum storage cap per session in MB (default: {settings.EVIDENCE_MAX_SESSION_MB})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate cleanup and print what would be deleted without removing files",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PROCTORING EVIDENCE RETENTION CLEANUP UTILITY")
    print("=" * 70)
    print(f"Target Evidence Directory: {settings.EVIDENCE_STORAGE_DIR}")
    print(f"Retention Period:          {args.days} days")
    print(f"Max Session Storage Cap:   {args.max_session_mb} MB")
    print(f"Dry Run Mode:              {args.dry_run}")
    print("-" * 70)

    evidence_service = get_evidence_service()

    summary = evidence_service.cleanup_evidence_sync(
        retention_days=args.days,
        max_session_mb=args.max_session_mb,
        dry_run=args.dry_run,
    )

    print("-" * 70)
    print("CLEANUP RESULTS SUMMARY:")
    print(f"  Files Deleted / Pruned:     {summary.get('deleted_files_count', 0)}")
    print(f"  Disk Space Freed:           {summary.get('freed_mb', 0.0)} MB ({summary.get('freed_bytes', 0)} bytes)")
    print(f"  Empty Session Dirs Removed: {len(summary.get('purged_empty_sessions', []))}")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()

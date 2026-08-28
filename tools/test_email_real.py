"""
Real SMTP verification — sends one test email per notification type
to the configured NOTIFICATION_EMAIL_TO inbox.

Requires SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and NOTIFICATION_EMAIL_TO
to be set in .env.

Usage:
    python tools/test_email_real.py --confirm
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from notifications.email_sender import (
    EmailConfig,
    EmailError,
    SmtpEmailProvider,
    send_job_failure,
    send_job_success,
    send_partial_success,
)
from config import settings


def _load_config() -> EmailConfig | None:
    host = settings.SMTP_HOST
    port = settings.SMTP_PORT
    username = settings.SMTP_USERNAME
    password = settings.SMTP_PASSWORD
    email_to = settings.NOTIFICATION_EMAIL_TO

    missing = []
    if not host:
        missing.append("SMTP_HOST")
    if not username:
        missing.append("SMTP_USERNAME")
    if not password:
        missing.append("SMTP_PASSWORD")
    if not email_to:
        missing.append("NOTIFICATION_EMAIL_TO")

    if missing:
        print(f"  Missing or empty: {', '.join(missing)}")
        print("  Set them in .env and re-run.")
        return None

    return EmailConfig(
        smtp_host=host,
        smtp_port=port,
        smtp_username=username,
        smtp_password=password,
        email_to=email_to,
        max_retries=1,
    )


def _check_result(label: str, result) -> bool:
    if result.success:
        print(f"  PASS  {label}  ->  sent to {settings.NOTIFICATION_EMAIL_TO}")
        return True
    else:
        print(f"  FAIL  {label}  ->  {result.error}")
        return False


def main() -> int:
    print("=" * 60)
    print("  Real SMTP Email Verification")
    print("=" * 60)
    print()

    if "--confirm" not in sys.argv:
        print("  SAFETY: This will send real emails to:")
        print(f"    {settings.NOTIFICATION_EMAIL_TO or '(not configured)'}")
        print()
        print("  Re-run with --confirm to proceed:")
        print("    python tools/test_email_real.py --confirm")
        return 1

    cfg = _load_config()
    if cfg is None:
        return 1

    provider = SmtpEmailProvider()
    passed = 0
    total = 3

    print(f"  Sending to: {cfg.email_to}")
    print(f"  SMTP host:  {cfg.smtp_host}:{cfg.smtp_port}")
    print(f"  SMTP user:  {cfg.smtp_username}")
    print()

    # --- Test 1: Success ---
    print("[1/3] send_job_success() ...")
    r1 = send_job_success(
        job_id="verify-success-001",
        topic="Real SMTP Verification",
        video_id="test_video_real",
        video_url="https://youtube.com/shorts/test_video_real",
        upload_status="uploaded",
        thumbnail_status="uploaded",
        config=cfg,
        provider=provider,
    )
    if _check_result("SUCCESS", r1):
        passed += 1
    print()

    # --- Test 2: Failure ---
    print("[2/3] send_job_failure() ...")
    r2 = send_job_failure(
        job_id="verify-failure-002",
        failed_stage="images",
        error_type="ImageGenError",
        error_message="Pexels API returned 429",
        details={"scene": 3, "retries_exhausted": True},
        config=cfg,
        provider=provider,
    )
    if _check_result("FAILURE", r2):
        passed += 1
    print()

    # --- Test 3: Partial ---
    print("[3/3] send_partial_success() ...")
    r3 = send_partial_success(
        job_id="verify-partial-003",
        topic="Real SMTP Partial",
        video_id="partial_vid_real",
        video_url="https://youtube.com/shorts/partial_vid_real",
        upload_status="uploaded",
        thumbnail_status="failed",
        warnings=["Thumbnail upload failed", "Short narration detected"],
        config=cfg,
        provider=provider,
    )
    if _check_result("PARTIAL", r3):
        passed += 1
    print()

    # --- Summary ---
    print("=" * 60)
    if passed == total:
        print(f"  ALL {total} TESTS PASSED")
        print(f"  Check your inbox at {cfg.email_to}")
    else:
        print(f"  {total - passed} of {total} TESTS FAILED")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

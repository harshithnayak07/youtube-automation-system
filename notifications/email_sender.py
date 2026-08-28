"""Notifications module — email notifications for job outcomes."""

from __future__ import annotations

import logging
import smtplib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 2.0

_RETRYABLE_SMTP_CODES = frozenset({421, 451, 452, 454, 458, 459})
_NON_RETRYABLE_SMTP_CODES = frozenset({500, 501, 502, 503, 504, 550, 551, 552, 553, 554})


@dataclass(frozen=True)
class EmailConfig:
    """Email notification configuration loaded from environment."""
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    email_to: str
    max_retries: int = _MAX_RETRIES


def load_email_config() -> EmailConfig:
    """Load email configuration from environment variables.

    Returns
    -------
    EmailConfig
        Validated configuration object.

    Raises
    ------
    EmailConfigError
        If required configuration is missing.
    """
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
        raise EmailConfigError(
            f"Missing required email configuration: {', '.join(missing)}. "
            "Set them in your .env file or environment."
        )

    return EmailConfig(
        smtp_host=host,
        smtp_port=port,
        smtp_username=username,
        smtp_password=password,
        email_to=email_to,
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EmailError(Exception):
    """Raised when email sending fails."""


class EmailConfigError(EmailError):
    """Raised for missing or invalid email configuration."""


class EmailAuthError(EmailError):
    """Raised when SMTP authentication fails."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SendResult:
    """Structured result of an email send operation."""
    success: bool
    subject: str
    error: str = ""
    details: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Provider interface (replaceable)
# ---------------------------------------------------------------------------

class EmailProvider(ABC):
    """Interface for pluggable email backends."""

    @abstractmethod
    def send_email(
        self,
        *,
        subject: str,
        body: str,
        config: EmailConfig,
    ) -> None:
        """Send an email.

        Parameters
        ----------
        subject:
            Email subject line.
        body:
            Plain text email body.
        config:
            Email configuration with SMTP credentials.

        Raises
        ------
        EmailAuthError
            If SMTP authentication fails.
        EmailError
            If sending fails for other reasons.
        """
        ...


# ---------------------------------------------------------------------------
# Default provider — SMTP
# ---------------------------------------------------------------------------

class SmtpEmailProvider(EmailProvider):
    """Standard SMTP email provider."""

    def send_email(
        self,
        *,
        subject: str,
        body: str,
        config: EmailConfig,
    ) -> None:
        msg = MIMEMultipart()
        msg["From"] = config.smtp_username
        msg["To"] = config.email_to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(config.smtp_username, config.smtp_password)
                server.sendmail(config.smtp_username, [config.email_to], msg.as_string())
        except smtplib.SMTPAuthenticationError as exc:
            raise EmailAuthError(
                "SMTP authentication failed — check SMTP_USERNAME and SMTP_PASSWORD."
            ) from exc
        except smtplib.SMTPException as exc:
            raise EmailError(f"SMTP error: {exc}") from exc
        except OSError as exc:
            raise EmailError(f"SMTP connection error: {exc}") from exc

        logger.info("Email sent to %s: %s", config.email_to, subject)


# ---------------------------------------------------------------------------
# Email templates (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _build_success_body(
    *,
    job_id: str,
    topic: str,
    video_id: str,
    video_url: str,
    upload_status: str,
    thumbnail_status: str,
) -> str:
    """Build the plain-text body for a job success notification."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"Job Completed Successfully\n"
        f"==========================\n\n"
        f"Job ID:      {job_id}\n"
        f"Topic:       {topic}\n"
        f"Video ID:    {video_id}\n"
        f"Video URL:   {video_url}\n"
        f"Upload:      {upload_status}\n"
        f"Thumbnail:   {thumbnail_status}\n\n"
        f"Completed at: {ts}\n"
    )


def _build_failure_body(
    *,
    job_id: str,
    failed_stage: str,
    error_type: str,
    error_message: str,
    details: dict[str, Any] | None = None,
) -> str:
    """Build the plain-text body for a job failure notification."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"Job Failed",
        f"==========",
        f"",
        f"Job ID:         {job_id}",
        f"Failed Stage:   {failed_stage}",
        f"Error Type:     {error_type}",
        f"Error Message:  {error_message}",
        f"",
        f"Failed at: {ts}",
    ]
    if details:
        lines.append("")
        lines.append("Additional Details:")
        for key, val in details.items():
            lines.append(f"  {key}: {val}")
    return "\n".join(lines) + "\n"


def _build_partial_body(
    *,
    job_id: str,
    topic: str,
    video_id: str,
    video_url: str,
    upload_status: str,
    thumbnail_status: str,
    warnings: list[str],
) -> str:
    """Build the plain-text body for a partial success notification."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"Job Completed with Warnings",
        f"============================",
        f"",
        f"Job ID:      {job_id}",
        f"Topic:       {topic}",
        f"Video ID:    {video_id}",
        f"Video URL:   {video_url}",
        f"Upload:      {upload_status}",
        f"Thumbnail:   {thumbnail_status}",
        f"",
        f"Warnings:",
    ]
    for w in warnings:
        lines.append(f"  - {w}")
    lines.append("")
    lines.append(f"Completed at: {ts}")
    return "\n".join(lines) + "\n"


def _build_success_subject(job_id: str) -> str:
    return f"[YouTube Automation] Job {job_id} completed successfully"


def _build_failure_subject(job_id: str) -> str:
    return f"[YouTube Automation] Job {job_id} FAILED"


def _build_partial_subject(job_id: str) -> str:
    return f"[YouTube Automation] Job {job_id} completed with warnings"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def _is_transient(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying."""
    if isinstance(exc, EmailAuthError):
        return False
    if isinstance(exc, EmailError):
        msg = str(exc)
        if "SMTP" in msg:
            # Heuristic: connection errors are retryable, auth errors are not
            return "connection" in msg.lower() or "timeout" in msg.lower()
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def _send_with_retry(
    provider: EmailProvider,
    *,
    subject: str,
    body: str,
    config: EmailConfig,
) -> None:
    """Send email with bounded retries for transient failures."""
    last_exc: Exception | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            provider.send_email(subject=subject, body=body, config=config)
            return
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc):
                raise
            if attempt >= config.max_retries:
                break
            delay = _RETRY_DELAY_BASE ** attempt
            logger.warning(
                "Email send attempt %d/%d failed, retrying in %.1fs: %s",
                attempt, config.max_retries, delay, exc,
            )
            time.sleep(delay)

    raise EmailError(
        f"Email send failed after {config.max_retries} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Public API — send functions
# ---------------------------------------------------------------------------

def send_job_success(
    *,
    job_id: str,
    topic: str,
    video_id: str,
    video_url: str,
    upload_status: str,
    thumbnail_status: str,
    config: EmailConfig | None = None,
    provider: EmailProvider | None = None,
) -> SendResult:
    """Send a success notification for a completed job.

    Parameters
    ----------
    job_id:
        The job identifier.
    topic:
        The video topic/title.
    video_id:
        The YouTube video ID.
    video_url:
        The YouTube video URL.
    upload_status:
        Upload status (e.g. "uploaded").
    thumbnail_status:
        Thumbnail status (e.g. "uploaded", "failed").
    config:
        Email configuration. Loaded from environment if None.
    provider:
        Email provider. Uses SmtpEmailProvider if None.

    Returns
    -------
    SendResult
        Structured result of the send operation.
    """
    cfg = config or load_email_config()
    prov = provider or SmtpEmailProvider()
    subject = _build_success_subject(job_id)

    try:
        body = _build_success_body(
            job_id=job_id,
            topic=topic,
            video_id=video_id,
            video_url=video_url,
            upload_status=upload_status,
            thumbnail_status=thumbnail_status,
        )
        _send_with_retry(prov, subject=subject, body=body, config=cfg)
        return SendResult(success=True, subject=subject)
    except EmailError as exc:
        logger.error("Failed to send success notification: %s", exc)
        return SendResult(success=False, subject=subject, error=str(exc))
    except Exception as exc:
        logger.error("Unexpected error sending success notification: %s", exc)
        return SendResult(success=False, subject=subject, error=str(exc))


def send_job_failure(
    *,
    job_id: str,
    failed_stage: str,
    error_type: str,
    error_message: str,
    details: dict[str, Any] | None = None,
    config: EmailConfig | None = None,
    provider: EmailProvider | None = None,
) -> SendResult:
    """Send a failure notification for a failed job.

    Parameters
    ----------
    job_id:
        The job identifier.
    failed_stage:
        The pipeline stage that failed (e.g. "research", "upload").
    error_type:
        The exception class name (e.g. "UploadError").
    error_message:
        The error message.
    details:
        Optional additional diagnostic details.
    config:
        Email configuration. Loaded from environment if None.
    provider:
        Email provider. Uses SmtpEmailProvider if None.

    Returns
    -------
    SendResult
        Structured result of the send operation.
    """
    cfg = config or load_email_config()
    prov = provider or SmtpEmailProvider()
    subject = _build_failure_subject(job_id)

    try:
        body = _build_failure_body(
            job_id=job_id,
            failed_stage=failed_stage,
            error_type=error_type,
            error_message=error_message,
            details=details,
        )
        _send_with_retry(prov, subject=subject, body=body, config=cfg)
        return SendResult(success=True, subject=subject)
    except EmailError as exc:
        logger.error("Failed to send failure notification: %s", exc)
        return SendResult(success=False, subject=subject, error=str(exc))
    except Exception as exc:
        logger.error("Unexpected error sending failure notification: %s", exc)
        return SendResult(success=False, subject=subject, error=str(exc))


def send_partial_success(
    *,
    job_id: str,
    topic: str,
    video_id: str,
    video_url: str,
    upload_status: str,
    thumbnail_status: str,
    warnings: list[str],
    config: EmailConfig | None = None,
    provider: EmailProvider | None = None,
) -> SendResult:
    """Send a partial success notification (completed with warnings).

    Parameters
    ----------
    job_id:
        The job identifier.
    topic:
        The video topic/title.
    video_id:
        The YouTube video ID.
    video_url:
        The YouTube video URL.
    upload_status:
        Upload status.
    thumbnail_status:
        Thumbnail status.
    warnings:
        List of warning messages.
    config:
        Email configuration. Loaded from environment if None.
    provider:
        Email provider. Uses SmtpEmailProvider if None.

    Returns
    -------
    SendResult
        Structured result of the send operation.
    """
    cfg = config or load_email_config()
    prov = provider or SmtpEmailProvider()
    subject = _build_partial_subject(job_id)

    try:
        body = _build_partial_body(
            job_id=job_id,
            topic=topic,
            video_id=video_id,
            video_url=video_url,
            upload_status=upload_status,
            thumbnail_status=thumbnail_status,
            warnings=warnings,
        )
        _send_with_retry(prov, subject=subject, body=body, config=cfg)
        return SendResult(success=True, subject=subject)
    except EmailError as exc:
        logger.error("Failed to send partial success notification: %s", exc)
        return SendResult(success=False, subject=subject, error=str(exc))
    except Exception as exc:
        logger.error("Unexpected error sending partial success notification: %s", exc)
        return SendResult(success=False, subject=subject, error=str(exc))

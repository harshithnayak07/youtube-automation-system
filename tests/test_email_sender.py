"""Tests for the notifications.email_sender module — all use mocks, no real emails."""

from __future__ import annotations

import smtplib
from unittest.mock import MagicMock, patch, call

import pytest

from notifications.email_sender import (
    EmailConfig,
    EmailError,
    EmailConfigError,
    EmailAuthError,
    EmailProvider,
    SendResult,
    SmtpEmailProvider,
    load_email_config,
    send_job_success,
    send_job_failure,
    send_partial_success,
    _build_success_body,
    _build_failure_body,
    _build_partial_body,
    _build_success_subject,
    _build_failure_subject,
    _build_partial_subject,
    _is_transient,
    _send_with_retry,
    _MAX_RETRIES,
    _RETRY_DELAY_BASE,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> EmailConfig:
    defaults = dict(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="bot@example.com",
        smtp_password="secure_password_123",
        email_to="admin@example.com",
    )
    defaults.update(overrides)
    return EmailConfig(**defaults)


class StubProvider(EmailProvider):
    """In-memory email provider for testing."""

    def __init__(self):
        self.sent: list[dict] = []
        self.side_effect: Exception | None = None

    def send_email(self, *, subject: str, body: str, config: EmailConfig) -> None:
        if self.side_effect:
            raise self.side_effect
        self.sent.append({"subject": subject, "body": body})


# ---------------------------------------------------------------------------
# EmailConfig model
# ---------------------------------------------------------------------------

class TestEmailConfigModel:
    def test_fields(self):
        cfg = _make_config()
        assert cfg.smtp_host == "smtp.example.com"
        assert cfg.smtp_port == 587
        assert cfg.smtp_username == "bot@example.com"
        assert cfg.smtp_password == "secure_password_123"
        assert cfg.email_to == "admin@example.com"
        assert cfg.max_retries == _MAX_RETRIES

    def test_is_frozen(self):
        cfg = _make_config()
        with pytest.raises(AttributeError):
            cfg.smtp_host = "x"  # type: ignore[misc]

    def test_custom_retries(self):
        cfg = _make_config(max_retries=5)
        assert cfg.max_retries == 5


# ---------------------------------------------------------------------------
# load_email_config
# ---------------------------------------------------------------------------

class TestLoadEmailConfig:
    @patch("notifications.email_sender.settings")
    def test_loads_from_settings(self, mock_settings):
        mock_settings.SMTP_HOST = "smtp.test.com"
        mock_settings.SMTP_PORT = 465
        mock_settings.SMTP_USERNAME = "user@test.com"
        mock_settings.SMTP_PASSWORD = "pass123"
        mock_settings.NOTIFICATION_EMAIL_TO = "admin@test.com"
        cfg = load_email_config()
        assert cfg.smtp_host == "smtp.test.com"
        assert cfg.smtp_port == 465
        assert cfg.smtp_username == "user@test.com"
        assert cfg.smtp_password == "pass123"
        assert cfg.email_to == "admin@test.com"

    @patch("notifications.email_sender.settings")
    def test_raises_on_missing_host(self, mock_settings):
        mock_settings.SMTP_HOST = ""
        mock_settings.SMTP_USERNAME = "user"
        mock_settings.SMTP_PASSWORD = "pass"
        mock_settings.NOTIFICATION_EMAIL_TO = "admin@test.com"
        with pytest.raises(EmailConfigError, match="SMTP_HOST"):
            load_email_config()

    @patch("notifications.email_sender.settings")
    def test_raises_on_missing_all(self, mock_settings):
        mock_settings.SMTP_HOST = ""
        mock_settings.SMTP_USERNAME = ""
        mock_settings.SMTP_PASSWORD = ""
        mock_settings.NOTIFICATION_EMAIL_TO = ""
        with pytest.raises(EmailConfigError, match="Missing required"):
            load_email_config()


# ---------------------------------------------------------------------------
# _is_transient
# ---------------------------------------------------------------------------

class TestIsTransient:
    def test_connection_error_is_transient(self):
        assert _is_transient(ConnectionError("refused")) is True

    def test_timeout_error_is_transient(self):
        assert _is_transient(TimeoutError("timed out")) is True

    def test_os_error_is_transient(self):
        assert _is_transient(OSError("network unreachable")) is True

    def test_email_auth_error_not_transient(self):
        assert _is_transient(EmailAuthError("auth failed")) is False

    def test_generic_email_error_not_transient(self):
        assert _is_transient(EmailError("unknown")) is False

    def test_smtp_connection_error_is_transient(self):
        assert _is_transient(EmailError("SMTP connection refused")) is True

    def test_smtp_timeout_error_is_transient(self):
        assert _is_transient(EmailError("SMTP timeout")) is True

    def test_smtp_other_error_not_transient(self):
        assert _is_transient(EmailError("SMTP error: something else")) is False

    def test_value_error_not_transient(self):
        assert _is_transient(ValueError("bad value")) is False


# ---------------------------------------------------------------------------
# _build_success_body
# ---------------------------------------------------------------------------

class TestBuildSuccessBody:
    def test_contains_all_fields(self):
        body = _build_success_body(
            job_id="job-001",
            topic="AI Revolution",
            video_id="vid123",
            video_url="https://youtube.com/shorts/vid123",
            upload_status="uploaded",
            thumbnail_status="uploaded",
        )
        assert "job-001" in body
        assert "AI Revolution" in body
        assert "vid123" in body
        assert "https://youtube.com/shorts/vid123" in body
        assert "uploaded" in body
        assert "Completed at:" in body

    def test_no_secrets_in_body(self):
        body = _build_success_body(
            job_id="j",
            topic="t",
            video_id="v",
            video_url="u",
            upload_status="ok",
            thumbnail_status="ok",
        )
        assert "secure_password" not in body
        assert "client_secret" not in body
        assert "refresh_token" not in body


# ---------------------------------------------------------------------------
# _build_failure_body
# ---------------------------------------------------------------------------

class TestBuildFailureBody:
    def test_contains_all_fields(self):
        body = _build_failure_body(
            job_id="job-002",
            failed_stage="upload",
            error_type="UploadError",
            error_message="OAuth token expired",
        )
        assert "job-002" in body
        assert "upload" in body
        assert "UploadError" in body
        assert "OAuth token expired" in body
        assert "Failed at:" in body

    def test_includes_details(self):
        body = _build_failure_body(
            job_id="j",
            failed_stage="qa",
            error_type="QAFailure",
            error_message="Video too short",
            details={"duration": 5, "min_duration": 15},
        )
        assert "duration" in body
        assert "5" in body
        assert "min_duration" in body
        assert "15" in body

    def test_no_secrets_in_body(self):
        body = _build_failure_body(
            job_id="j",
            failed_stage="s",
            error_type="E",
            error_message="m",
        )
        assert "secure_password" not in body
        assert "client_secret" not in body


# ---------------------------------------------------------------------------
# _build_partial_body
# ---------------------------------------------------------------------------

class TestBuildPartialBody:
    def test_contains_all_fields(self):
        body = _build_partial_body(
            job_id="job-003",
            topic="AI News",
            video_id="vid456",
            video_url="https://youtube.com/shorts/vid456",
            upload_status="uploaded",
            thumbnail_status="failed",
            warnings=["Thumbnail upload failed", "Short narration"],
        )
        assert "job-003" in body
        assert "AI News" in body
        assert "vid456" in body
        assert "uploaded" in body
        assert "failed" in body
        assert "Thumbnail upload failed" in body
        assert "Short narration" in body
        assert "Completed at:" in body


# ---------------------------------------------------------------------------
# Subject lines
# ---------------------------------------------------------------------------

class TestSubjectLines:
    def test_success_subject(self):
        s = _build_success_subject("job-42")
        assert "job-42" in s
        assert "completed successfully" in s

    def test_failure_subject(self):
        s = _build_failure_subject("job-42")
        assert "job-42" in s
        assert "FAILED" in s

    def test_partial_subject(self):
        s = _build_partial_subject("job-42")
        assert "job-42" in s
        assert "warnings" in s


# ---------------------------------------------------------------------------
# _send_with_retry
# ---------------------------------------------------------------------------

class TestSendWithRetry:
    def test_succeeds_on_first_try(self):
        provider = StubProvider()
        cfg = _make_config()
        _send_with_retry(provider, subject="s", body="b", config=cfg)
        assert len(provider.sent) == 1

    def test_retries_on_transient_then_succeeds(self):
        provider = StubProvider()
        call_count = 0
        original_send = provider.send_email

        def flaky_send(*, subject, body, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise EmailError("SMTP connection refused")
            return original_send(subject=subject, body=body, config=config)

        provider.send_email = flaky_send  # type: ignore[assignment]
        with patch("notifications.email_sender.time.sleep"):
            _send_with_retry(provider, subject="s", body="b", config=_make_config())
        assert call_count == 2

    def test_fails_after_exhausted_retries(self):
        provider = StubProvider()
        provider.side_effect = EmailError("SMTP connection refused")
        cfg = _make_config(max_retries=2)
        with patch("notifications.email_sender.time.sleep"):
            with pytest.raises(EmailError, match="failed after"):
                _send_with_retry(provider, subject="s", body="b", config=cfg)

    def test_no_retry_on_auth_error(self):
        provider = StubProvider()
        provider.side_effect = EmailAuthError("auth failed")
        cfg = _make_config(max_retries=3)
        with pytest.raises(EmailAuthError):
            _send_with_retry(provider, subject="s", body="b", config=cfg)
        assert provider.sent == []

    def test_no_retry_on_permanent_smtp_error(self):
        provider = StubProvider()
        provider.side_effect = EmailError("SMTP 550 mailbox unavailable")
        cfg = _make_config(max_retries=3)
        with pytest.raises(EmailError, match="SMTP 550 mailbox unavailable"):
            _send_with_retry(provider, subject="s", body="b", config=cfg)


# ---------------------------------------------------------------------------
# SmtpEmailProvider
# ---------------------------------------------------------------------------

class TestSmtpEmailProvider:
    def test_creates_message_and_sends(self):
        provider = SmtpEmailProvider()
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        with patch("notifications.email_sender.smtplib.SMTP", return_value=mock_smtp):
            provider.send_email(
                subject="Test Subject",
                body="Test Body",
                config=_make_config(),
            )
        mock_smtp.sendmail.assert_called_once()
        args = mock_smtp.sendmail.call_args
        assert args[0][0] == "bot@example.com"  # from
        assert args[0][1] == ["admin@example.com"]  # to list
        msg_str = args[0][2]
        assert "Test Subject" in msg_str
        assert "Test Body" in msg_str

    def test_raises_auth_error_on_535(self):
        provider = SmtpEmailProvider()
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b"auth failed")
        with patch("notifications.email_sender.smtplib.SMTP", return_value=mock_smtp):
            with pytest.raises(EmailAuthError, match="authentication failed"):
                provider.send_email(
                    subject="s", body="b", config=_make_config(),
                )

    def test_raises_on_smtp_error(self):
        provider = SmtpEmailProvider()
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp.sendmail.side_effect = smtplib.SMTPException("send failed")
        with patch("notifications.email_sender.smtplib.SMTP", return_value=mock_smtp):
            with pytest.raises(EmailError, match="SMTP error"):
                provider.send_email(
                    subject="s", body="b", config=_make_config(),
                )

    def test_raises_on_connection_error(self):
        provider = SmtpEmailProvider()
        with patch("notifications.email_sender.smtplib.SMTP", side_effect=OSError("refused")):
            with pytest.raises(EmailError, match="connection error"):
                provider.send_email(
                    subject="s", body="b", config=_make_config(),
                )


# ---------------------------------------------------------------------------
# send_job_success
# ---------------------------------------------------------------------------

class TestSendJobSuccess:
    def test_successful_send(self):
        provider = StubProvider()
        result = send_job_success(
            job_id="job-1",
            topic="AI News",
            video_id="vid123",
            video_url="https://youtube.com/shorts/vid123",
            upload_status="uploaded",
            thumbnail_status="uploaded",
            config=_make_config(),
            provider=provider,
        )
        assert result.success is True
        assert len(provider.sent) == 1
        assert "job-1" in provider.sent[0]["subject"]
        assert "completed successfully" in provider.sent[0]["subject"]

    def test_failure_returns_send_result(self):
        provider = StubProvider()
        provider.side_effect = ConnectionError("refused")
        result = send_job_success(
            job_id="job-2",
            topic="t",
            video_id="v",
            video_url="u",
            upload_status="ok",
            thumbnail_status="ok",
            config=_make_config(),
            provider=provider,
        )
        assert result.success is False
        assert "refused" in result.error

    def test_body_contains_video_url(self):
        provider = StubProvider()
        send_job_success(
            job_id="job-3",
            topic="t",
            video_id="v",
            video_url="https://youtube.com/shorts/abc",
            upload_status="ok",
            thumbnail_status="ok",
            config=_make_config(),
            provider=provider,
        )
        assert "https://youtube.com/shorts/abc" in provider.sent[0]["body"]

    def test_body_includes_thumbnail_status(self):
        provider = StubProvider()
        send_job_success(
            job_id="j",
            topic="t",
            video_id="v",
            video_url="u",
            upload_status="uploaded",
            thumbnail_status="failed",
            config=_make_config(),
            provider=provider,
        )
        assert "failed" in provider.sent[0]["body"]

    def test_no_secrets_in_email(self):
        provider = StubProvider()
        send_job_success(
            job_id="j",
            topic="t",
            video_id="v",
            video_url="u",
            upload_status="ok",
            thumbnail_status="ok",
            config=_make_config(smtp_password="top_secret_pw"),
            provider=provider,
        )
        body = provider.sent[0]["body"]
        subject = provider.sent[0]["subject"]
        assert "top_secret_pw" not in body
        assert "top_secret_pw" not in subject


# ---------------------------------------------------------------------------
# send_job_failure
# ---------------------------------------------------------------------------

class TestSendJobFailure:
    def test_successful_send(self):
        provider = StubProvider()
        result = send_job_failure(
            job_id="job-f1",
            failed_stage="upload",
            error_type="UploadError",
            error_message="Token expired",
            config=_make_config(),
            provider=provider,
        )
        assert result.success is True
        assert len(provider.sent) == 1
        assert "FAILED" in provider.sent[0]["subject"]
        assert "job-f1" in provider.sent[0]["subject"]

    def test_body_contains_stage(self):
        provider = StubProvider()
        send_job_failure(
            job_id="j",
            failed_stage="research",
            error_type="ResearchError",
            error_message="No results",
            config=_make_config(),
            provider=provider,
        )
        assert "research" in provider.sent[0]["body"]

    def test_body_contains_error_details(self):
        provider = StubProvider()
        send_job_failure(
            job_id="j",
            failed_stage="qa",
            error_type="QAFailure",
            error_message="Video too short",
            details={"duration": 5, "min_duration": 15},
            config=_make_config(),
            provider=provider,
        )
        body = provider.sent[0]["body"]
        assert "duration" in body
        assert "QAFailure" in body
        assert "Video too short" in body

    def test_failure_returns_send_result(self):
        provider = StubProvider()
        provider.side_effect = EmailAuthError("bad creds")
        result = send_job_failure(
            job_id="j",
            failed_stage="s",
            error_type="E",
            error_message="m",
            config=_make_config(),
            provider=provider,
        )
        assert result.success is False
        assert "bad creds" in result.error

    def test_no_secrets_in_email(self):
        provider = StubProvider()
        send_job_failure(
            job_id="j",
            failed_stage="s",
            error_type="E",
            error_message="m",
            config=_make_config(smtp_password="secret_pw"),
            provider=provider,
        )
        body = provider.sent[0]["body"]
        assert "secret_pw" not in body


# ---------------------------------------------------------------------------
# send_partial_success
# ---------------------------------------------------------------------------

class TestSendPartialSuccess:
    def test_successful_send(self):
        provider = StubProvider()
        result = send_partial_success(
            job_id="job-p1",
            topic="AI News",
            video_id="vid789",
            video_url="https://youtube.com/shorts/vid789",
            upload_status="uploaded",
            thumbnail_status="failed",
            warnings=["Thumbnail upload failed"],
            config=_make_config(),
            provider=provider,
        )
        assert result.success is True
        assert len(provider.sent) == 1
        assert "warnings" in provider.sent[0]["subject"]
        assert "job-p1" in provider.sent[0]["subject"]

    def test_body_contains_warnings(self):
        provider = StubProvider()
        send_partial_success(
            job_id="j",
            topic="t",
            video_id="v",
            video_url="u",
            upload_status="ok",
            thumbnail_status="failed",
            warnings=["Thumb failed", "Short narration"],
            config=_make_config(),
            provider=provider,
        )
        body = provider.sent[0]["body"]
        assert "Thumb failed" in body
        assert "Short narration" in body

    def test_body_contains_video_url(self):
        provider = StubProvider()
        send_partial_success(
            job_id="j",
            topic="t",
            video_id="v",
            video_url="https://youtube.com/shorts/xyz",
            upload_status="ok",
            thumbnail_status="failed",
            warnings=[],
            config=_make_config(),
            provider=provider,
        )
        assert "https://youtube.com/shorts/xyz" in provider.sent[0]["body"]

    def test_failure_returns_send_result(self):
        provider = StubProvider()
        provider.side_effect = OSError("network down")
        result = send_partial_success(
            job_id="j",
            topic="t",
            video_id="v",
            video_url="u",
            upload_status="ok",
            thumbnail_status="fail",
            warnings=[],
            config=_make_config(),
            provider=provider,
        )
        assert result.success is False
        assert "network down" in result.error

    def test_no_secrets_in_email(self):
        provider = StubProvider()
        send_partial_success(
            job_id="j",
            topic="t",
            video_id="v",
            video_url="u",
            upload_status="ok",
            thumbnail_status="fail",
            warnings=[],
            config=_make_config(smtp_password="my_secret"),
            provider=provider,
        )
        body = provider.sent[0]["body"]
        assert "my_secret" not in body


# ---------------------------------------------------------------------------
# SendResult model
# ---------------------------------------------------------------------------

class TestSendResultModel:
    def test_fields(self):
        r = SendResult(success=True, subject="Test Subject")
        assert r.success is True
        assert r.subject == "Test Subject"
        assert r.error == ""
        assert r.details is None

    def test_with_error(self):
        r = SendResult(success=False, subject="s", error="SMTP failed")
        assert r.success is False
        assert r.error == "SMTP failed"

    def test_with_details(self):
        r = SendResult(success=True, subject="s", details={"key": "val"})
        assert r.details is not None
        assert r.details["key"] == "val"

    def test_is_frozen(self):
        r = SendResult(success=True, subject="s")
        with pytest.raises(AttributeError):
            r.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class TestProviderInterface:
    def test_stub_provider_implements_interface(self):
        provider = StubProvider()
        assert isinstance(provider, EmailProvider)

    def test_smtp_provider_implements_interface(self):
        provider = SmtpEmailProvider()
        assert isinstance(provider, EmailProvider)


# ---------------------------------------------------------------------------
# Integration: retry + provider interaction
# ---------------------------------------------------------------------------

class TestRetryIntegration:
    def test_transient_then_success_with_real_retry(self):
        provider = StubProvider()
        call_count = 0
        original_send = provider.send_email

        def flaky_send(*, subject, body, config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise EmailError("SMTP connection refused")
            return original_send(subject=subject, body=body, config=config)

        provider.send_email = flaky_send  # type: ignore[assignment]
        with patch("notifications.email_sender.time.sleep") as mock_sleep:
            _send_with_retry(provider, subject="s", body="b", config=_make_config(max_retries=3))
        assert call_count == 2
        mock_sleep.assert_called_once()

    def test_auth_error_bypasses_retry(self):
        provider = StubProvider()
        provider.side_effect = EmailAuthError("bad creds")
        with pytest.raises(EmailAuthError):
            _send_with_retry(provider, subject="s", body="b", config=_make_config(max_retries=3))
        assert len(provider.sent) == 0

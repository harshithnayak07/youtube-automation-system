"""
Local email notification verification — exercises the real email stack
against a local SMTP server that supports STARTTLS and AUTH LOGIN.

The server is a simple threaded socket server (no asyncore) that properly
handles the ehlo → starttls → ehlo → auth → sendmail flow that
SmtpEmailProvider expects.

Usage:
    python tools/test_email_local.py
"""

from __future__ import annotations

import base64
import os
import socket
import ssl
import sys
import tempfile
import threading

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from notifications.email_sender import (
    EmailConfig,
    SmtpEmailProvider,
    send_job_failure,
    send_job_success,
    send_partial_success,
)

# ---------------------------------------------------------------------------
# Self-signed certificate (generated once via cryptography or stdlib fallback)
# ---------------------------------------------------------------------------


def _build_ssl_ctx() -> ssl.SSLContext:
    cert_p = os.path.join(tempfile.gettempdir(), "email_test_cert.pem")
    key_p = os.path.join(tempfile.gettempdir(), "email_test_key.pem")

    if not (os.path.exists(cert_p) and os.path.exists(key_p)):
        try:
            _generate_cryptography_cert(cert_p, key_p)
        except ImportError:
            raise RuntimeError(
                "The 'cryptography' package is required for local email "
                "verification.  Install it with: pip install cryptography"
            )

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_p, key_p)
    return ctx


def _generate_cryptography_cert(cert_path: str, key_path: str) -> None:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))


# ---------------------------------------------------------------------------
# Threaded SMTP server — handles STARTTLS + AUTH LOGIN properly
# ---------------------------------------------------------------------------


class _SMTPSession(threading.Thread):
    """Handles one SMTP client connection in a dedicated thread."""

    def __init__(self, conn: socket.socket, addr, ssl_ctx: ssl.SSLContext,
                 auth_user: str, auth_pass: str, capture_ref: dict):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self.ssl_ctx = ssl_ctx
        self.auth_user = auth_user
        self.auth_pass = auth_pass
        self._capture = capture_ref  # shared dict to store captured DATA

    def run(self):
        try:
            self.conn.settimeout(10)
            self._send(b"220 localhost SMTP ready\r\n")
            self._handle_smtp()
        except Exception:
            pass
        finally:
            try:
                self.conn.close()
            except Exception:
                pass

    def _send(self, data: bytes):
        self.conn.sendall(data)

    def _recv_line(self) -> str:
        buf = b""
        while not buf.endswith(b"\r\n"):
            chunk = self.conn.recv(1)
            if not chunk:
                raise ConnectionError("client disconnected")
            buf += chunk
        return buf.decode("ascii", errors="replace").strip()

    def _handle_smtp(self):
        authenticated = False
        tls_active = False

        while True:
            line = self._recv_line()
            upper = line.upper()

            if upper.startswith("EHLO") or upper.startswith("HELO"):
                self._send(b"250-localhost\r\n")
                if not tls_active:
                    self._send(b"250-STARTTLS\r\n")
                    self._send(b"250-AUTH LOGIN\r\n")
                    self._send(b"250 8BITMIME\r\n")
                else:
                    self._send(b"250-AUTH LOGIN\r\n")
                    self._send(b"250 8BITMIME\r\n")

            elif upper == "STARTTLS":
                self._send(b"220 Ready to start TLS\r\n")
                try:
                    self.conn = self.ssl_ctx.wrap_socket(self.conn, server_side=True)
                    self.conn.settimeout(10)
                    tls_active = True
                except Exception:
                    self._send(b"454 TLS handshake failed\r\n")
                    return

            elif upper.startswith("AUTH"):
                parts = line.split(None, 1)
                mechanism = parts[1].strip().split()[0].upper() if len(parts) > 1 else ""
                rest = parts[1].strip()[len(mechanism):].strip() if len(parts) > 1 else ""
                if mechanism == "LOGIN":
                    username = ""
                    if rest:
                        username = base64.b64decode(rest).decode("utf-8", "replace")
                    else:
                        self._send(b"334 VXNlcm5hbWU6\r\n")
                        username_b64 = self._recv_line()
                        username = base64.b64decode(username_b64).decode("utf-8", "replace")
                    self._send(b"334 UGFzc3dvcmQ6\r\n")
                    password_b64 = self._recv_line()
                    password = base64.b64decode(password_b64).decode("utf-8", "replace")
                    if username == self.auth_user and password == self.auth_pass:
                        self._send(b"235 Authentication successful\r\n")
                        authenticated = True
                    else:
                        self._send(b"535 Authentication credentials invalid\r\n")
                        return
                elif mechanism == "PLAIN":
                    if rest:
                        decoded = base64.b64decode(rest).decode("utf-8", "replace")
                    else:
                        self._send(b"334\r\n")
                        decoded = base64.b64decode(self._recv_line()).decode("utf-8", "replace")
                    parts_auth = decoded.split("\x00")
                    username = parts_auth[1] if len(parts_auth) > 1 else ""
                    password = parts_auth[2] if len(parts_auth) > 2 else ""
                    if username == self.auth_user and password == self.auth_pass:
                        self._send(b"235 Authentication successful\r\n")
                        authenticated = True
                    else:
                        self._send(b"535 Authentication credentials invalid\r\n")
                        return
                else:
                    self._send(b"504 Unrecognized mechanism\r\n")

            elif upper.startswith("MAIL FROM"):
                self._send(b"250 OK\r\n")

            elif upper.startswith("RCPT TO"):
                self._send(b"250 OK\r\n")

            elif upper == "DATA":
                self._send(b"354 Start mail input, end with <CRLF>.<CRLF>\r\n")
                data_lines = []
                while True:
                    data_line = self._recv_line()
                    if data_line == ".":
                        break
                    data_lines.append(data_line)
                self._capture["data"] = "\n".join(data_lines)
                self._send(b"250 OK\r\n")

            elif upper == "QUIT":
                self._send(b"221 Bye\r\n")
                return

            elif upper == "NOOP":
                self._send(b"250 OK\r\n")

            else:
                self._send(b"500 Command not recognized\r\n")


class _DebugSMTPServer:
    """Simple threaded SMTP server with STARTTLS and AUTH LOGIN."""

    def __init__(self, host: str, port: int, ssl_ctx: ssl.SSLContext,
                 auth_user: str, auth_pass: str):
        self.host = host
        self.port = port
        self.ssl_ctx = ssl_ctx
        self.auth_user = auth_user
        self.auth_pass = auth_pass
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(5)
        self._running = True
        self.last_captured: dict = {}

    def serve_forever(self):
        while self._running:
            try:
                self._sock.settimeout(1.0)
                conn, addr = self._sock.accept()
                self.last_captured = {}
                session = _SMTPSession(
                    conn, addr, self.ssl_ctx, self.auth_user, self.auth_pass,
                    self.last_captured,
                )
                session.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def shutdown(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Configuration — local debug server, dummy credentials
# ---------------------------------------------------------------------------

LOCAL_PORT = 1025
_AUTH_USER = "test-sender@example.com"
_AUTH_PASS = "not-a-real-password"

_config = EmailConfig(
    smtp_host="127.0.0.1",
    smtp_port=LOCAL_PORT,
    smtp_username=_AUTH_USER,
    smtp_password=_AUTH_PASS,
    email_to="test-recipient@example.com",
    max_retries=1,
)
_provider = SmtpEmailProvider()


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_server_ref: _DebugSMTPServer | None = None


def _start_debug_server(port: int) -> threading.Thread:
    global _server_ref
    ssl_ctx = _build_ssl_ctx()
    _server_ref = _DebugSMTPServer(
        "127.0.0.1", port, ssl_ctx, _AUTH_USER, _AUTH_PASS,
    )
    t = threading.Thread(target=_server_ref.serve_forever, daemon=True)
    t.start()
    return t


def _wait_for_server(host: str, port: int, timeout: float = 3.0) -> None:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Local SMTP server on {host}:{port} not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_output(func, *args, **kwargs):
    """Run *func* while capturing the SMTP DATA received by the server.

    Returns (return_value, captured_text).
    """
    _server_ref.last_captured = {}
    result = func(*args, **kwargs)
    import time
    time.sleep(0.3)  # let server session thread finish writing
    return result, _server_ref.last_captured.get("data", "")


def _verify(label: str, captured: str, checks: dict[str, str],
            private_values: list[str]) -> list[str]:
    """Return a list of failure messages (empty == pass)."""
    errors: list[str] = []
    for key, needle in checks.items():
        if needle not in captured:
            errors.append(f"  {label}: '{key}' not found in SMTP output")
    for secret in private_values:
        if secret in captured:
            errors.append(f"  {label}: SECRET LEAKED — '{secret[:8]}...' in SMTP output")
    return errors


# ---------------------------------------------------------------------------
# Test cases — one per notification path
# ---------------------------------------------------------------------------


def check_success():
    """Exercise send_job_success() against the local server."""
    result, captured = _collect_output(
        send_job_success,
        job_id="local-test-001",
        topic="Local Verification Topic",
        video_id="test_video_123",
        video_url="https://youtube.com/shorts/test_video_123",
        upload_status="uploaded",
        thumbnail_status="uploaded",
        config=_config,
        provider=_provider,
    )
    errors: list[str] = []
    if not result.success:
        errors.append(f"  send_job_success returned success=False: {result.error}")
    errors.extend(_verify("SUCCESS", captured, {
        "subject": "[YouTube Automation] Job local-test-001 completed successfully",
        "job_id": "local-test-001",
        "topic": "Local Verification Topic",
        "video_id": "test_video_123",
        "video_url": "https://youtube.com/shorts/test_video_123",
    }, [_config.smtp_password, "client_secret", "refresh_token"]))
    return errors


def check_failure():
    """Exercise send_job_failure() against the local server."""
    result, captured = _collect_output(
        send_job_failure,
        job_id="local-test-002",
        failed_stage="images",
        error_type="ImageGenError",
        error_message="Pexels API returned 429",
        details={"scene": 3, "retries_exhausted": True},
        config=_config,
        provider=_provider,
    )
    errors: list[str] = []
    if not result.success:
        errors.append(f"  send_job_failure returned success=False: {result.error}")
    errors.extend(_verify("FAILURE", captured, {
        "subject": "[YouTube Automation] Job local-test-002 FAILED",
        "job_id": "local-test-002",
        "failed_stage": "images",
        "error_type": "ImageGenError",
        "error_message": "Pexels API returned 429",
        "scene": "scene",
    }, [_config.smtp_password, "client_secret", "refresh_token"]))
    return errors


def check_partial():
    """Exercise send_partial_success() against the local server."""
    result, captured = _collect_output(
        send_partial_success,
        job_id="local-test-003",
        topic="Partial Success Topic",
        video_id="partial_vid_456",
        video_url="https://youtube.com/shorts/partial_vid_456",
        upload_status="uploaded",
        thumbnail_status="failed",
        warnings=["Thumbnail upload failed", "Short narration detected"],
        config=_config,
        provider=_provider,
    )
    errors: list[str] = []
    if not result.success:
        errors.append(f"  send_partial_success returned success=False: {result.error}")
    errors.extend(_verify("PARTIAL", captured, {
        "subject": "[YouTube Automation] Job local-test-003 completed with warnings",
        "job_id": "local-test-003",
        "topic": "Partial Success Topic",
        "warning_1": "Thumbnail upload failed",
        "warning_2": "Short narration detected",
    }, [_config.smtp_password, "client_secret", "refresh_token"]))
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 60)
    print("  Email Notification — Local Verification")
    print("=" * 60)
    print()

    print(f"[1/4] Starting local SMTP server on 127.0.0.1:{LOCAL_PORT} ...")
    print("       (STARTTLS + AUTH LOGIN enabled)")
    _start_debug_server(LOCAL_PORT)
    _wait_for_server("127.0.0.1", LOCAL_PORT)
    print("       Server ready.")
    print()

    print("[2/4] Testing send_job_success() ...")
    e1 = check_success()
    print("       PASS" if not e1 else "       FAIL")
    for e in e1:
        print(e)

    print()
    print("[3/4] Testing send_job_failure() ...")
    e2 = check_failure()
    print("       PASS" if not e2 else "       FAIL")
    for e in e2:
        print(e)

    print()
    print("[4/4] Testing send_partial_success() ...")
    e3 = check_partial()
    print("       PASS" if not e3 else "       FAIL")
    for e in e3:
        print(e)

    all_errors = e1 + e2 + e3
    print()
    print("=" * 60)
    if not all_errors:
        print("  ALL 3 TESTS PASSED — email stack verified locally")
        print("  No real emails were sent.")
    else:
        print(f"  {len(all_errors)} ASSERTION(S) FAILED")
    print("=" * 60)

    if _server_ref:
        _server_ref.shutdown()

    return 0 if not all_errors else 1


if __name__ == "__main__":
    sys.exit(main())

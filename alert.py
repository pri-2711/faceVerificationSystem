# =============================================================
#  alert.py  —  Gmail email alert system
#
#  Uses Python's built-in smtplib — no pip install needed.
#  Face snapshots are attached as snapshot.jpg to the email.
#
#  One-time Gmail setup:
#    1. myaccount.google.com → Security → enable 2-Step Verification
#    2. Search "App Passwords" → generate one for Mail
#    3. Paste the 16-char password in config.py as SENDER_PASSWORD
# =============================================================

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage
from datetime             import datetime
import config


def _send(subject, body, image_bytes=None):
    """
    Internal sender. All public alert functions route through here.

    Args:
        subject     (str)         : email subject
        body        (str)         : plain-text body
        image_bytes (bytes | None): raw JPEG bytes attached as snapshot.jpg
    """
    try:
        msg              = MIMEMultipart()
        msg["From"]      = config.SENDER_EMAIL
        msg["To"]        = config.ADMIN_EMAIL
        msg["Subject"]   = subject
        msg.attach(MIMEText(body, "plain"))

        if image_bytes is not None:
            img = MIMEImage(image_bytes, name="snapshot.jpg")
            img.add_header("Content-Disposition", "attachment", filename="snapshot.jpg")
            msg.attach(img)

        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
            server.sendmail(config.SENDER_EMAIL, config.ADMIN_EMAIL, msg.as_string())

        print(f"[Alert] Email sent → {config.ADMIN_EMAIL}")
        return True

    except Exception as e:
        print(f"[Alert] Failed to send email: {e}")
        return False


# ── Public alert functions ─────────────────────────────────────

def alert_unknown_face(timestamp=None, image_bytes=None):
    """Single unknown face detected — attaches snapshot."""
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _send(
        subject     = f"[ALERT] Unknown Face Detected — {ts}",
        body        = (
            f"An unknown face was detected.\n\n"
            f"Timestamp : {ts}\n"
            f"Status    : Access Denied\n\n"
            f"Snapshot is attached. Please review."
        ),
        image_bytes = image_bytes
    )


def alert_multiple_unknowns(count, minutes, image_bytes=None):
    """Multiple unknowns in a short window — possible security event."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _send(
        subject     = f"[ALERT] Unknown Faces in {minutes} min — {ts}",
        body        = (
            f"Multiple unknown faces detected in a short window.\n\n"

            f"Window    : last {minutes} minutes\n"
            f"Timestamp : {ts}\n\n"
            f"This may indicate an unauthorised access attempt."
        ),
        image_bytes = image_bytes
    )


def alert_duplicate_entry(name, timestamp=None):
    """Same person detected again within the duplicate window."""
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _send(
        subject = f"[ALERT] Duplicate Entry — {name} — {ts}",
        body    = (
            f"A duplicate entry attempt was detected.\n\n"
            f"Name      : {name}\n"
            f"Timestamp : {ts}\n\n"
            f"This person already has a recent verified entry on record."
        )
    )


def alert_system_error(error_msg):
    """Critical system error or crash."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _send(
        subject = f"[ALERT] System Error — {ts}",
        body    = (
            f"Face Verification System encountered an error.\n\n"
            f"Error     : {error_msg}\n"
            f"Timestamp : {ts}\n\n"
            f"Please restart the system and check logs."
        )
    )


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Sending test email...")
    _send(
        subject = "Test — Face Verification System",
        body    = (
            "This is a test email from your Face Verification System.\n"
            "If you received this, Gmail alerts are configured correctly."
        )
    )
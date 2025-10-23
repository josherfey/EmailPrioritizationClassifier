# save as generate_mfa_emails.py and run: python generate_mfa_emails.py
import csv
import random
from datetime import datetime, timedelta

random.seed(42)  # remove or change for different outputs

SENDERS = [
    "security@appleid.com", "no-reply@chase.com", "login@google.com", "noreply@outlook.com",
    "verification@paypal.com", "no-reply@amazon.com", "security@bankofamerica.com",
    "verification@microsoft.com", "no-reply@facebookmail.com", "verify@coinbase.com",
    "no-reply@netflix.com", "alert@wellsfargo.com", "support@dropbox.com", "noreply@twitter.com",
    "auth@github.com", "no-reply@instagram.com", "security@venmo.com", "verify@airbnb.com",
    "security@uber.com", "no-reply@linkedin.com"
]

SUBJECT_TEMPLATES = [
    "Your verification code",
    "Verification required",
    "Confirm your login",
    "Sign-in attempt detected",
    "Security alert: new login",
    "Your one-time code",
    "Confirm your account access",
    "Use this code to continue",
    "Identity verification code",
    "2-step verification code"
]

BODY_TEMPLATES = [
    "Your verification code is {code}. Enter this code to continue signing in.",
    "Use {code} to verify your account. This code will expire in {mins} minutes.",
    "Enter {code} to confirm your login attempt from {device}. If this wasn’t you, please secure your account.",
    "Use security code {code} to finish signing in on your new device.",
    "Here’s your verification code: {code}. Keep this code private and do not share it.",
    "Use {code} to verify your identity and complete your sign-in. This link expires at {expiry}.",
    "Security code: {code}. You received this because of a recent login attempt.",
    "Enter {code} on the verification page to confirm it’s you.",
    "Your one-time verification code is {code}. It will expire shortly.",
    "Use {code} to access your account securely. If you didn't request this, ignore this email."
]

DEVICES = [
    "Chrome on Windows", "Safari on Mac", "Firefox on Linux", "Chrome on Android",
    "Safari on iPhone", "Edge on Windows", "Mobile App (iOS)", "Mobile App (Android)"
]

OUTPUT_CSV = "./data/mfa_verification_emails.csv"
N = 1000

rows = []
for _ in range(N):
    sender = random.choice(SENDERS)
    subject = random.choice(SUBJECT_TEMPLATES)
    code = random.randint(100000, 999999)
    mins = random.choice([5, 10, 15])
    device = random.choice(DEVICES)
    expiry_dt = (datetime.utcnow() + timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M UTC")
    body_template = random.choice(BODY_TEMPLATES)
    body = body_template.format(code=code, mins=mins, device=device, expiry=expiry_dt)
    # Assemble email text: From, Subject, then body on the next line (no "Body:" label)
    email_text = f"From: {sender}\nSubject: {subject}\n{body}"
    rows.append({"email": email_text, 'label': "Prioritize"})

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["email", 'label'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {N} MFA emails to {OUTPUT_CSV}")

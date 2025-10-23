#!/usr/bin/env python3
# generate_urgent_emails.py
# Generates time-sensitive/urgent email examples (default N=1000).
# Output: CSV with columns: email_text, Priority

import csv
import random
from datetime import datetime, timedelta

random.seed(42)  # change or remove for different outputs

# Configuration
N = 1000                           # number of emails to generate
OUTPUT_CSV = "./data/urgent_time_sensitive_emails.csv"
FROM_DOMAINS = ["company.com", "ops.company.com", "alerts.bank.com", "billing.example.com", "security.company.com"]
SENDER_NAMES = ["Operations", "Billing", "Finance", "Security Team", "Account Manager", "Support"]
RECIPIENT_FIRSTS = ["Josh", "Alex", "Sam", "Taylor", "Morgan", "Pat", "Jordan", "Casey", "Riley", "Dana"]
RECIPIENT_LASTS = ["Smith", "Johnson", "Lee", "Garcia", "Davis", "Martinez", "Brown", "Wilson", "Moore", "Clark"]

# Subject templates (all urgent / time-sensitive)
SUBJECT_TEMPLATES = [
    "Urgent: Action required by {deadline}",
    "Time-Sensitive: Response needed within {hours} hours",
    "Immediate Attention Required – {topic}",
    "Final Notice: {topic} due {deadline}",
    "Deadline Today: {topic}",
    "Critical: please respond regarding {topic}",
    "Payment Overdue: Invoice {invoice_id}",
    "Security Alert: Respond by {deadline}",
    "Escalation: {topic} needs your approval",
    "Immediate: Confirm {topic} by {deadline}"
]

# Short description/topics to insert in subject/body
TOPICS = [
    "Q3 budget approval", "vendor payment", "invoice reconciliation", "contract signature",
    "server outage ticket", "compliance acknowledgment", "account verification",
    "shipment hold release", "access provisioning", "refund authorization"
]

# Body sentence fragments / templates for urgency (mix and match)
BODY_TEMPLATES = [
    # Direct commands / short imperative
    "Please approve {topic} by {deadline}.",
    "We require your signature on {topic} before {deadline}.",
    "Pay invoice {invoice_id} immediately to avoid service interruption.",
    "This is a final reminder: {topic} must be completed by {deadline}.",
    "Confirm receipt and action on {topic} within {hours} hours.",
    # Context + consequence
    "If we do not receive confirmation by {deadline}, services will be suspended and late fees will apply.",
    "Failure to respond within {hours} hours may result in escalation to legal/compliance.",
    "This change impacts scheduled deliveries—please confirm to prevent delays.",
    # Clarifying / extra info
    "Reference: case #{case_id}. Contact {contact_name} at {contact_phone} for immediate assistance.",
    "Attached: the updated document and summary of required changes.",
    # Questions and polite phrasing mixed with urgency
    "Can you confirm you will complete this by {deadline}?",
    "Are you able to provide the requested information before {deadline}?",
    "We need a short confirmation (Yes/No) within {hours} hours so we can proceed.",
    # Apologetic + urgent
    "Apologies for the short notice—this must be handled today to meet regulatory timelines.",
    "Sorry for the urgency; this came up unexpectedly and requires immediate approval.",
    # Actionable instructions
    "Log in to the dashboard and click 'Approve' on request {request_id}.",
    "Please reply with 'APPROVED' and the authorization code, or call {contact_phone}.",
    "Complete the attached form and send it back by {deadline}.",
    # Emphasize priority and contact
    "This matter is high priority—reach out to {contact_name} if you have any blockers.",
    "If you did not authorize this, contact our Security Desk immediately at {contact_phone}.",
]

# Greetings & sign-offs
GREETINGS = ["Hi {recipient},", "Hello {recipient},", "Dear {recipient},", ""]
SIGN_OFFS = ["Thanks,", "Regards,", "Best,", "Sincerely,", "Thank you,"]

def random_sender():
    name = random.choice(SENDER_NAMES)
    domain = random.choice(FROM_DOMAINS)
    local = name.lower().replace(" ", ".")
    return f"{local}@{domain}"

def random_recipient():
    first = random.choice(RECIPIENT_FIRSTS)
    last = random.choice(RECIPIENT_LASTS)
    return f"{first} {last}", f"{first.lower()}.{last.lower()}@example.com"

def rand_deadline(days_min=0, days_max=5, within_day=False):
    # returns string like "2025-10-24 15:00" or "today by 5:00 PM" depending on within_day
    dt = datetime.utcnow() + timedelta(days=random.randint(days_min, days_max),
                                       hours=random.randint(0, 23),
                                       minutes=random.choice([0,15,30,45]))
    if within_day:
        return dt.strftime("%b %d at %I:%M %p UTC")
    return dt.strftime("%Y-%m-%d %H:%M UTC")

def random_invoice_id():
    return f"INV-{random.randint(100000, 999999)}"

def random_case_id():
    return f"{random.randint(1000000, 9999999)}"

def random_request_id():
    return f"REQ-{random.randint(10000,99999)}"

def random_contact():
    name = random.choice(RECIPIENT_FIRSTS) + " " + random.choice(RECIPIENT_LASTS)
    phone = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
    return name, phone

def build_body(recipient_name):
    # Build a multi-sentence, coherent urgent body
    n_sentences = random.choices([2,3,4], weights=[0.5,0.35,0.15])[0]
    parts = []
    topic = random.choice(TOPICS)
    hours = random.choice([2,4,6,12,24])
    deadline = rand_deadline(days_min=0, days_max=2, within_day=True)
    invoice_id = random_invoice_id()
    case_id = random_case_id()
    request_id = random_request_id()
    contact_name, contact_phone = random_contact()
    
    # start with a greeting sometimes
    greeting = random.choice(GREETINGS).format(recipient=recipient_name)
    if greeting:
        parts.append(greeting)
    
    # choose several templates and fill placeholders
    templates = random.sample(BODY_TEMPLATES, k=n_sentences)
    for t in templates:
        txt = t.format(topic=topic, deadline=deadline, hours=hours,
                       invoice_id=invoice_id, case_id=case_id,
                       request_id=request_id, contact_name=contact_name,
                       contact_phone=contact_phone)
        parts.append(txt)
    
    # add sign-off sometimes
    sign = random.choice(SIGN_OFFS)
    if random.random() < 0.9:  # mostly include sign-offs
        parts.append(sign)
        parts.append(random.choice(SENDER_NAMES))
    return " ".join(parts)

def build_subject():
    template = random.choice(SUBJECT_TEMPLATES)
    hours = random.choice([2,4,6,12,24])
    deadline = rand_deadline(days_min=0, days_max=2, within_day=False)
    topic = random.choice(TOPICS)
    invoice_id = random_invoice_id()
    subj = template.format(deadline=deadline, hours=hours, topic=topic, invoice_id=invoice_id)
    # sometimes prepend priority words explicitly
    if random.random() < 0.25:
        subj = "URGENT: " + subj
    return subj

def assemble_email_text(sender, subject, body):
    # single text column: From:, Subject:, then body on next line(s)
    return f"From: {sender}\nSubject: {subject}\n{body}"

# Generate rows
rows = []
for i in range(N):
    recipient_name, recipient_email = random_recipient()
    sender = random_sender()
    subject = build_subject()
    body = build_body(recipient_name)
    email_text = assemble_email_text(sender, subject, body)
    rows.append({"email": email_text, "label": "Prioritize"})

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["email", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {N} urgent/time-sensitive emails to {OUTPUT_CSV}")

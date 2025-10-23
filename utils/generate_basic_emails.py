#!/usr/bin/env python3
# generate_non_urgent_emails.py
# Generates 5,000 non-urgent (low-priority) emails in CSV form.
# Output columns: email_text, Priority

import csv
import random
from datetime import datetime, timedelta

random.seed(123)

N = 5000
OUTPUT_CSV = "./data/non_urgent_basic_emails.csv"

# Possible senders and recipients
SENDER_NAMES = ["Alex", "Taylor", "Jordan", "Riley", "Casey", "Morgan", "Sam", "Jamie", "Avery", "Cameron"]
SENDER_DOMAINS = ["company.com", "mail.com", "teamhub.io", "project.io", "example.org"]
RECIPIENT_NAMES = ["Josh", "Pat", "Dana", "Chris", "Jamie", "Robin", "Kelly", "Drew", "Skyler", "Jess"]

# Subject templates (non-urgent)
SUBJECT_TEMPLATES = [
    "Quick question about {topic}",
    "Following up on {topic}",
    "Notes from our last meeting",
    "Checking in about {topic}",
    "Small update on {topic}",
    "Thanks for your help with {topic}",
    "Friendly reminder about {topic}",
    "Let’s discuss {topic} next week",
    "Ideas for improving {topic}",
    "Feedback on {topic}"
]

# Topics
TOPICS = [
    "the Q3 project", "marketing materials", "website redesign", "data report",
    "product roadmap", "budget review", "team offsite", "client proposal",
    "presentation slides", "weekly sync", "testing plan", "API update"
]

# Body sentence templates (calm, conversational)
BODY_TEMPLATES = [
    "I just wanted to check in and see how things are going with {topic}.",
    "Hope you’ve had a good week so far.",
    "No rush at all — just wanted to share a quick thought on {topic}.",
    "Here are a few ideas we could explore when you have a chance.",
    "Let’s plan to catch up later this week or early next.",
    "Please take a look whenever it’s convenient.",
    "I appreciate your input on this — thanks again!",
    "It’s not urgent, just keeping you in the loop.",
    "We made some small updates based on your feedback.",
    "I’ll follow up again once we have more info.",
    "Just wanted to see if you had any updates on {topic}.",
    "Everything on my end looks good so far.",
    "We can finalize the details next week if that works for you.",
    "I’ll add this to our next team agenda.",
    "No need to reply right away.",
    "Sharing this in case it’s helpful for your reference.",
    "Hope this makes sense — happy to clarify if needed.",
    "It’s been a busy week, hope things are going smoothly for you.",
    "We can revisit this when timing works better.",
    "Just wanted to confirm you received my last message.",
    "Here’s a quick summary of where we left things.",
    "Thanks again for reviewing those documents.",
    "Feel free to make any changes you think are needed.",
    "Looking forward to hearing your thoughts on this.",
    "Not a big deal, just thought I’d check in.",
    "We can discuss this during our next sync.",
    "Everything is on track from my perspective.",
    "I’ll keep you posted as things progress.",
    "It’s been a while since we last talked about {topic}.",
    "Adding this note so we don’t lose track of it.",
    "I’ll send a reminder closer to the date.",
    "Appreciate your patience as we work through this.",
    "Thanks for catching that detail — great eye.",
    "I reviewed your comments and they make sense.",
    "Hope the transition has been going smoothly.",
    "This one isn’t time-sensitive, just a quick thought.",
    "I went ahead and made those minor edits.",
    "We should be able to finalize things soon.",
    "I wanted to make sure you saw the latest update.",
    "Thanks for looping me in.",
    "Let’s revisit this after the weekend.",
    "All looks good from my side — thanks for checking.",
    "No changes needed at this point, I think we’re good.",
    "Sharing this so it’s on your radar.",
    "Hope you’re having a good start to the week.",
    "Following up from our last chat on {topic}.",
    "I think we’re aligned, but open to your feedback.",
    "Nothing urgent, just wanted to share this note.",
    "Let’s touch base once everyone’s back from vacation.",
    "Appreciate your quick turnaround last time.",
    "I’ll take care of that and update you once done.",
    "Adding you here for visibility.",
    "If you have a moment, could you take a quick look?",
    "Feel free to hold off until next week.",
    "I’ll coordinate with the rest of the team and circle back.",
    "Hope this helps move things forward a bit.",
    "Please let me know if you see anything off.",
    "Not expecting a reply right away — just wanted to share.",
    "Thanks for pulling that together so quickly.",
    "I’ve attached the notes from our discussion.",
    "This should cover what we talked about earlier.",
    "I’ll reach out again once we have final numbers.",
    "We’re making good progress overall.",
    "Appreciate your thoughts on the latest draft.",
    "I’ll double-check and confirm tomorrow.",
    "Adding this item for future consideration.",
    "We’ll handle the rest internally for now.",
    "I think we’re heading in the right direction.",
    "Here’s a brief recap from today’s meeting.",
    "No major updates, just sharing where things stand.",
    "Thanks for the clarification earlier.",
    "We should be set for now — I’ll let you know if that changes.",
    "Let’s plan to regroup next month.",
    "I went through the document and left a few notes.",
    "Your suggestions were really helpful — thanks.",
    "I’ll let you know once I’ve spoken with the others.",
    "We’re waiting on one more piece of info before proceeding.",
    "Just keeping you updated on where things are.",
    "Hope all’s well with you and the team.",
    "It’s been nice collaborating on this.",
    "We’ve made good headway, thanks to your input.",
    "This aligns with what we discussed previously.",
    "I’ve incorporated your latest feedback.",
    "Feel free to ignore this until you have time.",
    "This might be useful for reference later.",
    "I’ll update the spreadsheet accordingly.",
    "I added a few more notes to the shared doc.",
    "No action needed, just FYI.",
    "We’ll monitor and adjust if needed.",
    "The schedule looks fine from my end.",
    "Please let me know if you’d prefer a different approach.",
    "Just documenting this for future reference.",
    "I’ll keep an eye on the responses as they come in.",
    "We should be able to wrap this up soon.",
    "Everything seems to be on track at the moment.",
    "Thanks again for taking the time to review.",
    "We can pick this back up after the break.",
    "Adding a quick summary so it’s easy to follow.",
    "Let me know if anything changes.",
    "It’s not critical, but I thought it was worth mentioning.",
    "Happy to go over this in more detail later.",
    "All good on my side — hope the same for you."
]


# Greetings and sign-offs
GREETINGS = ["Hi {recipient},", "Hello {recipient},", "Hey {recipient},", "", "Good morning {recipient},"]
SIGN_OFFS = ["Thanks,", "Best,", "Cheers,", "Talk soon,", "Take care,", "Appreciate it,"]

def random_sender():
    name = random.choice(SENDER_NAMES)
    domain = random.choice(SENDER_DOMAINS)
    return f"{name.lower()}@{domain}"

def random_recipient():
    return random.choice(RECIPIENT_NAMES)

def random_topic():
    return random.choice(TOPICS)

def random_subject(topic):
    return random.choice(SUBJECT_TEMPLATES).format(topic=topic)

def build_body(recipient, topic):
    n_sentences = random.choice([2,3,4])
    sentences = []
    greeting = random.choice(GREETINGS).format(recipient=recipient)
    if greeting:
        sentences.append(greeting)
    for _ in range(n_sentences):
        sentences.append(random.choice(BODY_TEMPLATES).format(topic=topic))
    signoff = random.choice(SIGN_OFFS)
    sender_name = random.choice(SENDER_NAMES)
    sentences.append(signoff)
    sentences.append(sender_name)
    return " ".join(sentences)

def assemble_email_text(sender, subject, body):
    return f"From: {sender}\nSubject: {subject}\n{body}"

# Generate emails
rows = []
for i in range(N):
    recipient = random_recipient()
    sender = random_sender()
    topic = random_topic()
    subject = random_subject(topic)
    body = build_body(recipient, topic)
    email_text = assemble_email_text(sender, subject, body)
    rows.append({"email": email_text, "label": "Default"})

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["email", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {N} non-urgent emails to {OUTPUT_CSV}")

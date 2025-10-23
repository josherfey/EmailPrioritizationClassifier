# generate_promotional_emails.py
#!/usr/bin/env python3
# generate_promotional_emails.py
# Generates 1000 promotional emails (marketing-style) and saves to CSV.

import csv
import random

random.seed(42)  # For reproducibility

N = 3000
OUTPUT_CSV = "./data/promotional_emails.csv"

# Possible senders
SENDERS = [
    "offers@amazon.com", "promotions@nike.com", "newsletter@apple.com",
    "deals@bestbuy.com", "sales@target.com", "rewards@starbucks.com",
    "updates@spotify.com", "no-reply@airbnb.com", "hello@etsy.com",
    "news@wholefoods.com", "offers@delta.com", "deals@expedia.com",
    "rewards@uber.com", "newsletter@costco.com", "info@rei.com",
    "promos@patagonia.com", "news@lyft.com", "offers@doordash.com",
    "no-reply@sephora.com", "updates@netflix.com"
]

# Subject templates
SUBJECT_TEMPLATES = SUBJECT_TEMPLATES = [
    "Save {discount}% on your next order!",
    "Exclusive offer just for you — {discount}% off today!",
    "Don’t miss our {discount}% off sale this weekend!",
    "New arrivals just dropped — shop now!",
    "Special savings inside: {discount}% off everything!",
    "You’ve earned {discount}% off — use it before it’s gone!",
    "Spring sale is here! {discount}% off select items!",
    "Flash deal: {discount}% off for the next 24 hours!",
    "Member exclusive: {discount}% off your next purchase!",
    "Time to refresh your favorites — {discount}% off sitewide!",
    "Limited-time deal: get {discount}% off now!",
    "Your favorite items are waiting — {discount}% off today!",
    "Weekend special: save {discount}% on selected products!",
    "Shop more, save more — up to {discount}% off!",
    "Don’t wait! {discount}% off ends soon!",
    "Exclusive online offer: {discount}% off everything!",
    "Special promotion: {discount}% off your cart total!",
    "Get ready for savings — {discount}% off top picks!",
    "Hot deal: {discount}% off just for you!",
    "Today only: {discount}% off your order!",
    "Back in stock! Get {discount}% off now!",
    "Early access: {discount}% off new arrivals!",
    "Your deal is here: {discount}% off selected items!",
    "Limited supply — {discount}% off while supplies last!",
    "Seasonal sale: {discount}% off your favorites!",
    "Flash sale alert: {discount}% off for 24 hours only!",
    "Exclusive: {discount}% off just for our subscribers!",
    "Don’t miss out on {discount}% off our bestsellers!",
    "Hot picks: {discount}% off today only!",
    "Shop today and enjoy {discount}% off everything!"
]

# Body templates
BODY_TEMPLATES = [
    "We’ve got something special for you. For a limited time, enjoy {discount}% off our most popular items.",
    "Shop new arrivals and get {discount}% off your first purchase. Don’t miss out!",
    "Celebrate with us — take {discount}% off select products. Offer ends soon.",
    "Your favorites are waiting! Save {discount}% today and treat yourself.",
    "Exclusive deal: {discount}% off everything in your cart. Act fast!",
    "Ready to save? {discount}% off for a limited time only. Shop now!",
    "This weekend only: get {discount}% off sitewide. Happy shopping!",
    "We thought you’d like this: {discount}% off our top picks for you.",
    "Special offer inside: {discount}% off your order. Don’t miss it!",
    "It’s time to refresh your wardrobe — enjoy {discount}% off today!",
    "Your personalized deal: {discount}% off selected items just for you.",
    "Grab your favorites while supplies last — {discount}% off today.",
    "Treat yourself: {discount}% off our newest collection.",
    "Don’t wait! {discount}% off ends at midnight tonight.",
    "Our biggest sale of the season — save {discount}% now!",
    "Flash deal: {discount}% off limited stock items.",
    "Only a few hours left — claim your {discount}% discount now!",
    "Spring into savings with {discount}% off select products.",
    "Your cart is calling! Get {discount}% off before it’s too late.",
    "Hot picks for you: {discount}% off top-rated products.",
    "Enjoy {discount}% off on our most popular categories.",
    "Special savings event: {discount}% off everything online.",
    "Shop now and save {discount}% on your next purchase.",
    "This week only: {discount}% off sitewide.",
    "Unlock {discount}% off your order today — limited time offer!",
    "Our gift to you: {discount}% off for loyal customers.",
    "Take advantage of {discount}% off our latest arrivals.",
    "Your favorites are discounted! {discount}% off for a short time.",
    "Get the look you love with {discount}% off today.",
    "Upgrade your collection with {discount}% off selected items.",
    "Check out our bestsellers — now {discount}% off!",
    "Don’t miss these deals: {discount}% off today only.",
    "Limited stock alert: {discount}% off your favorite items.",
    "Your exclusive deal: {discount}% off before it expires.",
    "Special online savings: {discount}% off everything you love.",
    "Refresh your wardrobe with {discount}% off new styles.",
    "Shop and save {discount}% on handpicked favorites.",
    "Enjoy this special promotion: {discount}% off selected products.",
    "It’s your lucky day! {discount}% off your next order.",
    "We’re celebrating with {discount}% off just for you.",
    "Your next purchase is discounted: save {discount}% now.",
    "Seasonal savings: {discount}% off popular items.",
    "Hot deals inside — {discount}% off for a limited time.",
    "Exclusive online offer: take {discount}% off today.",
    "Today only: save {discount}% on your favorite products.",
    "Check out our latest deals — {discount}% off selected items.",
    "Special promotion: enjoy {discount}% off your cart total.",
    "Shop our newest collection with {discount}% off today.",
    "Your personal offer: {discount}% off just for subscribers.",
    "Don’t miss out — {discount}% off while supplies last.",
    "Your favorites, now discounted: {discount}% off.",
    "Limited time: save {discount}% on top picks.",
    "Take {discount}% off and enjoy free shipping on select items."
]


# Function to create one email
def generate_email():
    sender = random.choice(SENDERS)
    discount = random.choice([10, 15, 20, 25, 30, 35, 40, 50])
    subject = random.choice(SUBJECT_TEMPLATES).format(discount=discount)
    body = random.choice(BODY_TEMPLATES).format(discount=discount)
    email_text = f"From: {sender}\nSubject: {subject}\n{body}"
    return {"email": email_text, "label": "Slow"}

# Generate emails
emails = [generate_email() for _ in range(N)]

# Write to CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["email", "label"])
    writer.writeheader()
    writer.writerows(emails)

print(f"Wrote {N} promotional emails to {OUTPUT_CSV}")

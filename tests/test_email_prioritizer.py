import pytest
import pandas as pd

from email_prioritizer.email_prioritizer import (
    EmailSplitter,
    NLPTransformer,
    EmailPrioritizer
)

# ------------------------------------------------------
# Fixtures
# ------------------------------------------------------
@pytest.fixture
def sample_emails():
    # Minimal valid RFC822-style emails
    raw0 = b"From: support@legitcompany.com\nSubject: Regarding Your Recent Inquiry\n\nThank you for reaching out regarding [your inquiry]. We have reviewed your request and will get back to you within 24 hours with a detailed response. Sincerely, Customer Service Team"
    raw4 = b"From: support@legitcompany.com\nSubject: Regarding Your Recent Inquiry\n\nThank you for reaching out regarding [your inquiry]. We have reviewed your request and will get back to you within 24 hours with a detailed response. Sincerely, Customer Service Team"
    raw1 = b"From: noreply@softwareupdates.com\nSubject: Weekly Newsletter - Latest Updates\n\nPlease find attached your invoice for the services rendered in June. The total amount due is $X.XX. Payment is due by [Date]. If you have any questions, please reply to this email. Thank you, Accounts Department"
    raw3 = b"From: noreply@softwareupdates.com\nSubject: Weekly Newsletter - Latest Updates\n\nPlease find attached your invoice for the services rendered in June. The total amount due is $X.XX. Payment is due by [Date]. If you have any questions, please reply to this email. Thank you, Accounts Department"
    raw2 = b"From: noreply@softwareupdates.com\nSubject: Weekly Newsletter - Latest Updates\n\nPlease find attached your invoice for the services rendered in June. The total amount due is $X.XX. Payment is due by [Date]. If you have any questions, please reply to this email. Thank you, Accounts Department"
    raw5 = b"From: support@legitcompany.com\nSubject: Regarding Your Recent Inquiry\n\nThank you for reaching out regarding [your inquiry]. We have reviewed your request and will get back to you within 24 hours with a detailed response. Sincerely, Customer Service Team"
    
    return pd.DataFrame({"email":[raw0, raw1, raw2, raw3, raw4, raw5]}).email

@pytest.fixture
def sample_labels():
    # Binary labels
    return pd.DataFrame({"label":["Prioritize", "Default", "Slow", "Slow","Prioritize", "Default",]}).label


# ------------------------------------------------------
# EmailSplitter Tests
# ------------------------------------------------------
def test_email_splitter(sample_emails):
    splitter = EmailSplitter()
    df = splitter.transform(sample_emails)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["sender", "subject", "body"]
    assert df.iloc[0]["sender"] == "support@legitcompany.com"
    assert df.iloc[1]["subject"] == "Weekly Newsletter - Latest Updates"
    assert "thank you" in df.iloc[0]["body"].lower()


# ------------------------------------------------------
# NLPTransformer Tests
# ------------------------------------------------------
def test_nlp_transformer_body_tokenization():
    transformer = NLPTransformer()
    out = transformer.transform(["This is SOME words!!!"])
    # Stemmed / stopword filtered result will vary slightly
    assert isinstance(out[0], str)
    assert "some" not in out[0].split()  # should be stemmed
    assert "thi" not in out[0].split()   # stopwords removed


def test_nlp_transformer_sender_mode():
    transformer = NLPTransformer(keep_sender=True)
    out = transformer.transform(["John.Doe@Example.COM"])
    assert out[0] == "john.doe@example.com"


# ------------------------------------------------------
# EmailPrioritizer End-to-End Tests
# ------------------------------------------------------
def test_prioritizer_fit_predict(sample_emails, sample_labels):
    model = EmailPrioritizer(model_type="logistic")
    model.fit(sample_emails, sample_labels, cv=2)  # small CV for speed

    preds = model.predict(sample_emails)
    assert len(preds) == len(sample_labels)
    assert set(preds).issubset({'Prioritize', 'Default', 'Slow'})


def test_predict_without_fit_raises(sample_emails):
    model = EmailPrioritizer(model_type="logistic")
    with pytest.raises(ValueError):
        model.predict(sample_emails)


def test_predict_proba_returns_probabilities(sample_emails, sample_labels):
    model = EmailPrioritizer(model_type="logistic")
    model.fit(sample_emails, sample_labels, cv=2)
    
    proba = model.predict_proba(sample_emails)
    assert proba.shape == (len(sample_emails), 3) # assuming 3 classes
    assert (proba >= 0).all() and (proba <= 1).all()


# ------------------------------------------------------
# Model-Type Behavior Tests
# ------------------------------------------------------
def test_random_forest_supports_predict(sample_emails, sample_labels):
    model = EmailPrioritizer(model_type="random_forest")
    model.fit(sample_emails, sample_labels, cv=2)
    preds = model.predict(sample_emails)
    assert len(preds) == len(sample_labels)


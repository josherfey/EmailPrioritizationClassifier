"""
email_prioritizer.py
-----------------------
Email classification module:
- Splits raw emails into sender, subject, and body
- NLP preprocessing on each field
- TF-IDF vectorization
- Machine learning models (Logistic Regression, Random Forest, CatBoost)
- Hyperparameter tuning and cross-validation
- Predict probabilities
"""

import re
from typing import List, Union

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from email import policy
from email.parser import BytesParser

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import FunctionTransformer

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

nltk.download('stopwords', quiet=True)

# --------------------------
# Custom transformers
# --------------------------
class EmailSplitter(BaseEstimator, TransformerMixin):
    """
    Split raw email text into sender, subject, and body.
    Returns a DataFrame with columns ['sender', 'subject', 'body'].
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = [self._parse_email(email_text) for email_text in X]
        df = pd.DataFrame(data, columns=['sender', 'subject', 'body'])
        return df.fillna('')

    @staticmethod
    def _parse_email(raw_email: Union[str, bytes]):
        if isinstance(raw_email, str):
            raw_email = raw_email.encode('utf-8')
        msg = BytesParser(policy=policy.default).parsebytes(raw_email)
        sender = msg['From'] or ''
        subject = msg['Subject'] or ''
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content()
        else:
            if msg.get_body(preferencelist=('plain')):
                body = msg.get_body(preferencelist=('plain')).get_content()
        return sender, subject, body

# --------------------------
class NLPTransformer(BaseEstimator, TransformerMixin):
    """
    NLP preprocessing: tokenization, stopwords removal, stemming.
    For sender, keep email addresses intact.
    """
    def __init__(self, keep_sender=False):
        self.keep_sender = keep_sender
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = []
        for text in X:
            tokens = self._preprocess_text(text)
            if not tokens:
                tokens = ["emptyplaceholder"]
            processed.append(" ".join(tokens))
        return processed

    def _preprocess_text(self, text: str):
        if not isinstance(text, str):
            text = str(text)
        if self.keep_sender:
            # Keep letters, digits, @, .
            text = re.sub(r"[^a-zA-Z0-9@.]", " ", text)
            tokens = text.lower().split()
            return tokens
        else:
            # Normal body/subject processing
            text = re.sub(r"[^a-zA-Z]", " ", text)
            tokens = text.lower().split()
            return [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]

# --------------------------
class EmailPrioritizer:
    """Pipeline for email priority classification."""
    def __init__(self, model_type: str = "logistic"):
        self.model_type = model_type.lower()
        self.pipeline = None
        # vectorizers can be customized
        self.vectorizer_sender = TfidfVectorizer(max_features=1000)
        self.vectorizer_subject = TfidfVectorizer(max_features=1000)
        self.vectorizer_body = TfidfVectorizer(max_features=5000)

    def _get_model_and_params(self):
        if self.model_type == "logistic":
            model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
            params = {"model__C": [0.01, 0.1, 1, 10]}
        elif self.model_type == "random_forest":
            model = RandomForestClassifier(random_state=42)
            params = {"model__n_estimators": [100, 200],
                      "model__max_depth": [None, 10, 20],
                      "model__min_samples_split": [2, 5]}
        elif self.model_type == "catboost":
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not installed. Run: pip install catboost")
            model = CatBoostClassifier(verbose=False)
            params = {"model__depth": [4, 6],
                      "model__learning_rate": [0.05, 0.1],
                      "model__iterations": [200, 500]}
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        return model, params

    def fit(self, X: List[str], y: List[int], cv: int = 5, sample_weight=None):
        """
        Train model using email text and labels.
        Optionally apply sample weights to emphasize reviewed data.
        """
        sender_pipeline = Pipeline([
            ('select', FunctionTransformer(lambda X: X['sender'].fillna('').values, validate=False)),
            ('nlp', NLPTransformer(keep_sender=True)),
            ('tfidf', self.vectorizer_sender)
        ])
        subject_pipeline = Pipeline([
            ('select', FunctionTransformer(lambda X: X['subject'].fillna('').values, validate=False)),
            ('nlp', NLPTransformer()),
            ('tfidf', self.vectorizer_subject)
        ])
        body_pipeline = Pipeline([
            ('select', FunctionTransformer(lambda X: X['body'].fillna('').values, validate=False)),
            ('nlp', NLPTransformer()),
            ('tfidf', self.vectorizer_body)
        ])

        features = FeatureUnion([
            ('sender_features', sender_pipeline),
            ('subject_features', subject_pipeline),
            ('body_features', body_pipeline)
        ])

        model, params = self._get_model_and_params()

        pipeline = Pipeline([
            ('splitter', EmailSplitter()),
            ('features', features),
            ('model', model)
        ])

        print(f"Running GridSearchCV for {self.model_type}...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid=params,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
        )

        # Note: sample_weight is not supported inside GridSearchCV
        grid_search.fit(X, y)
        self.pipeline = grid_search.best_estimator_

        print(f"\nBest cross-val f1 score: {grid_search.best_score_:.3f}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Retrain the best pipeline on the full dataset with sample weights (if provided)
        if sample_weight is not None:
            print("\nRetraining best model on full data with sample weights...")
            # Handle cases where model is wrapped inside pipeline
            try:
                self.pipeline.fit(X, y, model__sample_weight=sample_weight)
            except TypeError:
                # fallback if pipeline or model doesn't support sample_weight
                print("⚠️ sample_weight not supported for this model type. Training without weights.")
                self.pipeline.fit(X, y)
        else:
            self.pipeline.fit(X, y)


    def predict(self, X: List[str]) -> List[int]:
        if not self.pipeline:
            raise ValueError("Model not trained. Run .fit() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]):
        """
        Predict class probabilities for each email.
        Returns a 2D array: shape (n_samples, n_classes)
        """
        if not self.pipeline:
            raise ValueError("Model not trained. Run .fit() first.")
        if not hasattr(self.pipeline.named_steps['model'], 'predict_proba'):
            raise ValueError(f"The selected model ({self.model_type}) does not support predict_proba.")
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test: List[str], y_test: List[int]):
        if not self.pipeline:
            raise ValueError("Model not trained. Run .fit() first.")
        y_pred = self.pipeline.predict(X_test)
        print("\nEvaluation Report:")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))


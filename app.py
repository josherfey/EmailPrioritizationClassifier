import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import dill  # updated import
import sys
import os

module_path = os.path.abspath(os.path.join(os.getcwd(), "./email_prioritizer"))
sys.path.append(module_path)

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    # replace with your saved model path
    with open("./finalized_model_lr.dill", "rb") as f:
        model = dill.load(f)
    return model

model = load_model()
class_names = model.classes_  # get class labels

# ------------------------------
# UI Layout
# ------------------------------
st.title("Email Prioriziation Model")

st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input mode:", ["Single text input", "CSV upload"])

# ------------------------------
# Single Text Prediction
# ------------------------------
if mode == "Single text input":
    input_text = st.text_area("Enter text to classify. Please format like this:\n\n"
                              "From: operations@security.company.com\n"
                              "Subject: URGENT: Escalation: access provisioning needs your approval\n\n"
                            "Hello Sam Lee, This matter is high priority—reach out to Jordan Clark if you "
                            "have any blockers. If you did not authorize this, contact our Security Desk "
                            "immediately at +1-403-313-4841. Thank you, Operations\n\n\n"
                              
                              , height=200)
    if st.button("Predict Single Text"):
        if input_text.strip():
            try:
                pred_class = model.predict([input_text])[0]
                pred_proba = model.predict_proba([input_text])[0]

                st.subheader("🔮 Model Prediction")
                st.write(f"**Predicted Class:** {pred_class}")

                st.write("**Class Probabilities:**")
                st.dataframe(pd.DataFrame({
                    "Class": class_names,
                    "Probability": np.round(pred_proba, 3)
                }))
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning("Please enter some text.")

# ------------------------------
# CSV Batch Prediction
# ------------------------------
else:
    uploaded = st.file_uploader(
        "Upload CSV with text and optional labels. Must be in the format:\n\n"
        """
email,label\n\n
"From: alex@mail.com
Subject: Quick question about marketing materials
Hey Jess, Here’s a quick summary of where we left things. No major updates, just sharing where things stand. Thanks again for taking the time to review. Best, Taylor",Default\n\n
"From: operations@ops.company.com
Subject: URGENT: Payment Overdue: Invoice INV-907918
Hello Casey Garcia, This is a final reminder: account verification must be completed by Oct 25 at 06:44 AM UTC. Pay invoice INV-780261 immediately to avoid service interruption. Sincerely, Billing",Prioritize\n\n
"""
        , type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        text_col = st.text_input("Email", value="email")
        label_col = st.text_input("Column with true labels (optional)", value="label")

        if st.button("Predict CSV"):
            if text_col not in df.columns:
                st.error(f"CSV must have a column named '{text_col}'")
            else:
                try:
                    # Predict
                    predictions = model.predict(df[text_col])
                    pred_proba = model.predict_proba(df[text_col])
                    df["predicted_label"] = predictions
                    for i, cls in enumerate(class_names):
                        df[f"prob_{cls}"] = pred_proba[:, i]

                    st.subheader("Predictions")
                    st.dataframe(df)

                    st.download_button(
                        "Download predictions as CSV",
                        df.to_csv(index=False),
                        "predictions.csv",
                        "text/csv"
                    )

                    # Show performance if labels exist
                    if label_col in df.columns:
                        y_true = df[label_col]
                        y_pred = df["predicted_label"]

                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred, labels=class_names)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                    xticklabels=class_names, yticklabels=class_names, ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)

                        st.subheader("Classification Report")
                        st.text(classification_report(y_true, y_pred, target_names=class_names))

                except Exception as e:
                    st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Built by Josh Sherfey")

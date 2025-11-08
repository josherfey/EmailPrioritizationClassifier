import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import dill  # updated import
import sys
import os
from email_prioritizer.email_prioritizer import EmailPrioritizer    


module_path = os.path.abspath(os.path.join(os.getcwd(), "./src/email_prioritizer"))
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
st.title("Email Prioritization Model")

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

                st.subheader("Model Prediction")
                st.write(f"**Predicted Class:** {pred_class}")

                # Show class probabilities
                st.write("**Class Probabilities:**")
                st.dataframe(pd.DataFrame({
                    "Class": class_names,
                    "Probability": np.round(pred_proba, 3)
                }))

                # 👉 NEW: Explanation Section
                try:
                    pred_class, pos_words, neg_words, total_contrib = EmailPrioritizer.explain_single_prediction(email_text=input_text, pipeline=model)

                    st.subheader("Why This Prediction? (Model Explanation)")
                    st.write(f"**Total Contribution Weight:** `{total_contrib:.3f}` (higher magnitude = stronger confidence)")

                    st.write("### 🔼 Top words increasing support")
                    st.dataframe(pos_words.style.format({"contribution": "{:.3f}", "weight": "{:.3f}", "value": "{:.3f}"}))

                    st.write("### 🔽 Top words pushing against the prediction")
                    st.dataframe(neg_words.style.format({"contribution": "{:.3f}", "weight": "{:.3f}", "value": "{:.3f}"}))

                except Exception as e:
                    st.warning(f"Explanation unavailable: {e}")


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

# ------------------------------
# Fast Active Learning Review UI (uses 'email' column)
# ------------------------------
st.header("Active Learning: Batch Label Review")

with st.expander("Open Review Tool"):
    unlabeled_file = st.file_uploader("Upload unlabeled CSV (must contain an 'email' column)", type=["csv"], key="active_fast_csv")
    if unlabeled_file:
        df_unlabeled = pd.read_csv(unlabeled_file)

        if "email" not in df_unlabeled.columns:
            st.error("CSV must contain an 'email' column.")
        else:
            st.write("Predicting labels for uploaded data...")
            try:
                # Model prediction + uncertainty
                probs = model.predict_proba(df_unlabeled["email"])
                preds = model.predict(df_unlabeled["email"])
                uncertainties = 1 - np.max(probs, axis=1)

                df_unlabeled["predicted_label"] = preds
                df_unlabeled["uncertainty"] = uncertainties
                df_unlabeled = df_unlabeled.sort_values(by="uncertainty", ascending=False).reset_index(drop=True)

                # Select how many rows to review
                num_to_review = st.slider("Rows to review", 5, 100, 20)
                review_df = df_unlabeled.head(num_to_review)

                st.caption("Select the correct label for each email below:")
                reviewed_labels = []

                # Build fast table-like layout
                for idx, row in review_df.iterrows():
                    with st.container():
                        cols = st.columns([6, 2, 4])
                        with cols[0]:
                            st.markdown(f"**Email {idx+1}:** {row['email']}")
                        with cols[1]:
                            st.write(f"Pred: `{row['predicted_label']}`")
                            st.write(f"Unc: `{row['uncertainty']:.3f}`")
                        with cols[2]:
                            chosen = st.radio(
                                "Label",
                                class_names,
                                index=list(class_names).index(row["predicted_label"]) if row["predicted_label"] in class_names else 0,
                                key=f"radio_{idx}",
                                horizontal=True
                            )
                            reviewed_labels.append(chosen)

                if st.button("Save Reviewed Labels"):
                    review_df["reviewed_label"] = reviewed_labels
                    st.session_state["reviewed_data"] = review_df
                    st.success("✅ Labels reviewed and stored.")
                    st.dataframe(review_df[["email", "predicted_label", "reviewed_label", "uncertainty"]])

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Export reviewed data
    if "reviewed_data" in st.session_state:
        reviewed_data = st.session_state["reviewed_data"]
        st.download_button(
            "⬇️ Download reviewed data as CSV",
            reviewed_data.to_csv(index=False),
            "reviewed_labels.csv",
            "text/csv"
        )


st.markdown("---")
st.caption("Built by Josh Sherfey")

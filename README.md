# Prioritize Email Model

## Goal
The goal of this notebook is to demonstrate a basic model to prioritize emails into three buckets: slow, default, priority.

## Outcome
- Built a logistic regression model to predict email prioritization. 
- This model performance very well on synthetic data (n=500) with near perfect accuracy, indicating that synthetic data may be too cookie cutter and may not reflect real world varition.
- Model was applied on two real world unlabeld email datasets (Resend spam example) and Enron dataset with visualization below.

## Methodology

This model processes simulated **email events** and applies a prioritization label based on subject, from, and body of the email.
  1. `Prioritize`: Critical messages like MFA codes and email verification.
  2. `Default`: The default label for generic messages.
  3. `Slow`: The label for non-urgent, promotional messages

### Datasets
Thee datasets were used:
- synthetic labeled dat
  - used to train / evaluate model performance. 
  - Generated using the scripts in utils folder and used to catch several use cases: promotional emails, mfa verfication, time sensitive emails, and non urget emails.
- unlabeled spam detection data 
  - provided by Resend
- unlabled enron email data
  -  More details [here](https://technocrat.github.io/_book/the-enron-email-corpus.html)

## Model training
Model was trained using the synthetic data. Even though there were 10k emails in dataset, only 5k were used to train the model to avoid over training and since synthetic data was pretty simple. Even with limited training data, model performance was greater than 99% accuracy, indicating that synthetic data likely doesn't generalize well to real world. This considered, it will give a good baseline to improve upon as more real data comes in.

## Model evaluation
Three models were considered: logistic regression, random forest, and catboost. Given the simple nature of training data, logistic regression performed well enough and was chosen for its simplicity. We can evaluate later to swicth to more complex model as we get more features and labeled real world data

## Interacting with model
Included is a streamlet UI to test new email strings or bulk evaluate emails.
Notebook training and evaluating model can be found in notebook folder (prioritize_emails.ipynb)


#### Single email prediciton
<img width="847" height="837" alt="Screenshot 2025-10-23 at 4 19 32 PM" src="https://github.com/user-attachments/assets/dfc8a51e-3ed6-4d04-9aa8-3ef632032882" />


##### Batch prediction and evaluation
<img width="734" height="835" alt="Screenshot 2025-10-23 at 4 20 03 PM" src="https://github.com/user-attachments/assets/748b567c-0b03-4768-a2ad-fa98f2002931" />

<img width="454" height="618" alt="Screenshot 2025-10-23 at 4 20 14 PM" src="https://github.com/user-attachments/assets/575cca75-cb82-47cf-9c15-dca9b3a425be" />

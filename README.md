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

## Improving Performance with Active Learning
To improve model, active learning can be used to hand label data that the model is most uncertain about. My hand labeling the model, we are able to train the model to new and real datasets. 

This active learning is demonstrated in the notebook (prioritize_email.ipynb)

### Active Learning Workflow
1. Upload unlabeld data to active learning feature in streamlet
2. Hand label the most uncertain cases
3. Download the csv data of hand labeled data
4. Retrain the model using the hand labeled data
5. Publish the trained model to streamlet app


#### Single email prediciton
<img width="872" height="870" alt="Screenshot 2025-10-23 at 4 43 54 PM" src="https://github.com/user-attachments/assets/cd3a0445-c88d-4f03-a7a3-11103537f70f" />

##### Batch prediction and evaluation
<img width="706" height="695" alt="Screenshot 2025-10-23 at 4 46 07 PM" src="https://github.com/user-attachments/assets/6e206d79-68eb-4777-a74c-f5a269fbbeae" />

<img width="609" height="593" alt="Screenshot 2025-10-23 at 4 46 18 PM" src="https://github.com/user-attachments/assets/87052649-af4f-4c4f-a998-ea53da520c4e" />

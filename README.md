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
Included is a Streamlit UI to test new email strings or bulk evaluate emails.
Notebook training and evaluating model can be found in notebook folder (prioritize_emails.ipynb)

## Testing
Added testing using pytest. Todo will be to add more coverage of testing: including model performance testing on larger datasets.

## Improving Performance with Active Learning
To improve model, active learning can be used to hand label data that the model is most uncertain about. My hand labeling the model, we are able to train the model to new and real datasets. 

This active learning is demonstrated in the notebook (prioritize_email.ipynb)

### Active Learning Workflow
1. Upload unlabeld data to active learning feature in Streamlit
2. Hand label the most uncertain cases
3. Download the csv data of hand labeled data
4. Retrain the model using the hand labeled data
5. Publish the trained model to Streamlit app

## Streamlit Application Overview
#### Single email prediciton
UI supports classification of single email input.
<img width="872" height="870" alt="Screenshot 2025-10-23 at 4 43 54 PM" src="https://github.com/user-attachments/assets/cd3a0445-c88d-4f03-a7a3-11103537f70f" />

##### Batch prediction and evaluation
UI supports batch prediction and evaluation.
<img width="706" height="695" alt="Screenshot 2025-10-23 at 4 46 07 PM" src="https://github.com/user-attachments/assets/6e206d79-68eb-4777-a74c-f5a269fbbeae" />

<img width="609" height="593" alt="Screenshot 2025-10-23 at 4 46 18 PM" src="https://github.com/user-attachments/assets/87052649-af4f-4c4f-a998-ea53da520c4e" />

#### Improving Model with Active Learning
UI supports hand labeling difficult to classify instances to quickly improve model performance.
<img width="688" height="829" alt="Screenshot 2025-10-23 at 9 55 06 PM" src="https://github.com/user-attachments/assets/babbea7e-c582-4cfd-bb81-808072443ddd" />


## Next Steps
1. To improve the performance of this model, the most important step would be collect more ground truth data. If we have access to event logs in emails, then we could create a frame work to label data, where we determine email priority based on normal user actions.
2. Expand feature set to improve model performance

#### Automated Scoring Model to generate labeled data
We track the following features and create scoring system:
- Whether the user opened the email (1 pts)
- Whether the user replied (3 pts)
- Whether the user forwarded (4 pts)
- Whether the email was marked important / starred (5 pts)
- Whether the email was deleted without being opened (-2 pts)
- Whether the email was marked as spam (-5 pts)
- Time-to-open (shorter time is more important) (something like (1/x hours) pts)

Translate to score
- score >= 3: Priorititze
- score <= 0: Slow
- otherwise: Default

#### Add new features
Create new features and add to improve model performance. Potential list of features given below
- Sender-based features:
  - Sender domain reputation score (e.g., personal vs corporate vs unknown)
  - Whether sender is in user's contacts
  - Whether sender has emailed the user before
  - Frequency of past communication with sender
  - Average response time to this sender historically
  - Whether sender is in same organization / same domain
  - Sender email alias complexity (spam senders often use random strings)

- Email content features:
  - Email length (characters or words)
  - Subject line length
  - Presence of urgency keywords (e.g., "urgent", "ASAP", "action required")
  - Presence of question marks (may indicate requests that matter)
  - Uppercase emphasis count (e.g., words in ALL CAPS)
  - Link count (higher could signal marketing/spam)
  - Attachment count (often increases importance)

- Thread / conversation features:
  - Whether the email is part of an existing conversation thread
  - Email is reply vs forward vs new thread
  - Thread depth (number of messages in the conversation)
  - Whether others replied quickly in the thread (group urgency signal)

- User interaction history features:
  - Past open rate for this sender
  - Past reply rate for this sender
  - Average read time of emails from this sender
  - Whether the user previously marked similar emails as important
  - Whether the user archived or ignored similar emails

- Behavioral / temporal features:
  - Time of day email was received (work hours vs off hours)
  - Day of week patterns (e.g., newsletters on weekends)
  - Time-to-open (faster opens may imply higher importance)

- Categorical metadata features:
  - Email category (e.g., work, personal, finance, notification, newsletter)
  - Whether the email includes a calendar invite
  - Whether the email relates to a package delivery or notification
  - Industry or category of sender domain (e.g., SaaS, retail, finance)

- Message format & tracking features:
  - Presence of `List-Unsubscribe` header (strong newsletter signal)
  - Whether the email is plain text vs HTML
  - Presence of tracking pixel (indicative of marketing)
  - Presence of email signature block (often human-composed mail)

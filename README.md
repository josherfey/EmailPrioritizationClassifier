% Email Prioritization Classifier

This model processes simulated **email events** and applies a prioritization label based on subject, from, and body of the email.
  1. `Prioritize`: Critical messages like MFA codes and email verification.
  2. `Default`: The default label for generic messages.
  3. `Slow`: The label for non-urgent, promotional messages


Included is a streamlet UI to test new email strings or bulk evaluate emails.
Notebook training and evaluating model can be found in notebook folder (prioritize_emails.ipynb)

#### Single email prediciton
<img width="847" height="837" alt="Screenshot 2025-10-23 at 4 19 32 PM" src="https://github.com/user-attachments/assets/dfc8a51e-3ed6-4d04-9aa8-3ef632032882" />


##### Batch prediction and evaluation
<img width="734" height="835" alt="Screenshot 2025-10-23 at 4 20 03 PM" src="https://github.com/user-attachments/assets/748b567c-0b03-4768-a2ad-fa98f2002931" />

<img width="454" height="618" alt="Screenshot 2025-10-23 at 4 20 14 PM" src="https://github.com/user-attachments/assets/575cca75-cb82-47cf-9c15-dca9b3a425be" />

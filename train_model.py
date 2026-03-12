import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import string
import os

# 1. Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer() # Advanced: Reduces words to root form

if not os.path.exists('model'):
    os.makedirs('model')

# 2. Load Data
print("Loading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

# --- MASSIVE EXPANDED TRAINING DATA ---
# We are injecting real-world email structures to force the model to learn them
deep_spam_data = [
    # --- Marketing / Promotions ---
    ('spam', 'Up to 65% OFF on Home Appliances. Refrigerators, Air Conditioners & More. Shop NOW.'),
    ('spam', 'Get started with Adobe Acrobat Pro. Edit directly in your PDF. Fix it fast.'),
    ('spam', 'Introducing the BMW X7. A new dimension in mobility with outright performance.'),
    ('spam', 'Final call - registration for Twilio Forge: Messaging Reimagined is closing soon!'),
    ('spam', 'Exclusive Offer: 50% off on all electronics. Limited time deal.'),
    ('spam', 'Flash Sale! Get the latest smartphone at half price. Buy now.'),
    ('spam', 'Your free trial is expiring. Upgrade now to keep your premium access.'),
    ('spam', 'Congratulations! You have been selected for a special reward. Claim now.'),
    ('spam', 'Don not miss out. Huge discounts on laptops and gadgets this weekend.'),
    ('spam', 'Subscribe now and get 3 months free streaming service.'),
    ('spam', 'Special promotion for our loyal customers. Save big on your next order.'),

    # --- Webinar / Event / Professional Spam ---
    ('spam', 'You are invited to the virtual summit on AI and Machine Learning.'),
    ('spam', 'Join our practical session to learn coding. Register today.'),
    ('spam', 'Reminder: Webinar starting in 1 hour. Click here to join.'),
    ('spam', 'Network with industry leaders at the Global Tech Conference.'),
    ('spam', 'Prototype a branded messaging flow with RCS. Jumpstart omnichannel projects.'),

    # --- Phishing / Scam style ---
    ('spam', 'URGENT: Your bank account is locked. Verify your details immediately.'),
    ('spam', 'You have won a lottery of $1,000,000. Send your bank details.'),
    ('spam', 'Security Alert: Unusual login attempt detected. Reset your password.'),
    ('spam', 'Hi, I am a prince and I want to transfer money to you.'),

    # --- Hard Ham (Tricky legit emails) ---
    ('ham', 'Hey, can you send me the project files? I need them for the meeting.'),
    ('ham', 'Mom called, she wants you to come over for dinner tonight.'),
    ('ham', 'The meeting has been rescheduled to 5 PM. Please be on time.'),
    ('ham', 'I am running late, will be there in 10 minutes.'),
    ('ham', 'Did you watch the game last night? It was amazing.'),
    ('ham', 'Please find the attached report for this month sales.'),
    ('ham', 'Can we reschedule our lunch? Something came up.'),
    ('ham', 'Happy Birthday! Hope you have a great day.'),
    ('ham', 'Let us grab a beer after work.'),
    ('ham', 'I finished the assignment. Review it when you can.')
]

print(f"Adding {len(deep_spam_data)} custom training examples...")
df_extra = pd.DataFrame(deep_spam_data, columns=['label', 'message'])
df = pd.concat([df, df_extra], ignore_index=True)

# 3. Advanced Preprocessing Function
def clean_text(text):
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Tokenize and remove stopwords
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    # Apply Stemming (Deep Learning technique)
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

print("Cleaning text data (Deep Cleaning with Stemming)...")
df['clean_message'] = df['message'].apply(clean_text)

# 4. Feature Extraction (TF-IDF)
# ngram_range=(1,2) means it looks at single words AND pairs of words (better context)
print("Vectorizing data...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df['clean_message'])
y = df['label']

# 5. Train the Model
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adjust alpha for smoothing
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Trained Successfully with Accuracy: {accuracy*100:.2f}%")

# 6. Save
joblib.dump(model, 'model/spam_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
print("Model saved in 'model/' folder.")
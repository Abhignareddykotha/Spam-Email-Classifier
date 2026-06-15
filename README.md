# 📧 Spam Email Classifier

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Python and TF-IDF Vectorization.

---

# 🚀 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to detect whether an SMS message is spam or not.

The model:
- Reads SMS data
- Cleans and preprocesses text
- Converts text into numerical features using TF-IDF
- Trains a Naive Bayes classifier
- Predicts whether a new message is Spam or Not Spam

---

# ✨ Features

✅ SMS Spam Detection

✅ Text Preprocessing

✅ Lowercase Conversion

✅ Punctuation Removal

✅ TF-IDF Vectorization

✅ Naive Bayes Classification

✅ Custom User Input Prediction

---

# 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes

---

# 📂 Dataset

Dataset Used:
SMSSpamCollection

Dataset Size:
- Total Messages: 5572
- Ham Messages: 4825
- Spam Messages: 747

---

# ⚙️ Machine Learning Workflow

1. Load Dataset
2. Convert Labels to Numbers
3. Convert Text to Lowercase
4. Remove Punctuation
5. TF-IDF Vectorization
6. Split Training and Testing Data
7. Train Naive Bayes Model
8. Evaluate Accuracy
9. Predict New Messages

---

# 📊 Model Performance

Model Accuracy:

**95.70%**

---

# 🧪 Example Predictions

### Spam Message

Input:
Congratulations! You have won a free iPhone. Click here to claim.

Output:
❌ This is SPAM!

---

### Normal Message

Input:
Hey! Are you coming to college tomorrow?

Output:
✅ This is NOT SPAM!

---

# ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/Abhignareddykotha/Spam-Email-Classifier.git
```

Go to the project folder:

```bash
cd Spam-Email-Classifier
```

Install required libraries:

```bash
pip install pandas scikit-learn
```

Run the project:

```bash
python spam_classifier.py
```

---

# 📁 Project Structure

```
Spam-Email-Classifier/
│
├── README.md
├── spam_classifier.py
└── SMSSpamCollection
```

---

# 🎯 Future Improvements

- Add GUI
- Build a Web Application
- Email Spam Detection
- Improve Model Accuracy
- Deploy using Streamlit

---
# 📸 Sample Output

### Spam Detection
[Insert screenshot here]

### Normal Message Detection
[Insert screenshot here]

# 👩‍💻 Author

**Abhigna Reddy**

Computer Science Engineering Student

Interested in:
- Machine Learning
- AI
- UI/UX Design
- Web Development

---

⭐ If you like this project, consider giving it a star!

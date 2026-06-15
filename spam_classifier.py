import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Read dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

# Show first 5 rows
print("First 5 Messages:")
print(data.head())

# Dataset size
print("\nDataset Shape:")
print(data.shape)

# Spam and Ham count
print("\nSpam vs Ham:")
print(data["label"].value_counts())
# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

print("\nConverted Labels:")
print(data.head())
# Convert messages to lowercase
data["message"] = data["message"].str.lower()

print("\nLowercase Messages:")
print(data.head())
# Remove punctuation
data["message"] = data["message"].apply(
    lambda x: x.translate(str.maketrans('', '', string.punctuation))
)

print("\nMessages Without Punctuation:")
print(data.head())
# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["message"])

print("\nTF-IDF Matrix Shape:")
print(X.shape)
# Target values
y = data["label"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)
# Train the model
model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Training Completed!")
# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy*100:.2f}%")
# Test with user input

message = input("\nEnter a message: ")

# Convert to lowercase
message = message.lower()

# Remove punctuation
import string
message = message.translate(str.maketrans('', '', string.punctuation))

# Convert into TF-IDF features
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

# Display result
if prediction[0] == 1:
    print("\n🚫 This is SPAM!")
else:
    print("\n✅ This is NOT SPAM!")
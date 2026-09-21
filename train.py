import json
import nltk
import numpy as np
from nltk.stem import PorterStemmer

# -----------------------------
# 1. Initialize NLP tools
# -----------------------------

stemmer = PorterStemmer()


# -----------------------------
# 2. Load dataset
# -----------------------------

with open("intents.json", "r") as file:
    data = json.load(file)


# -----------------------------
# 3. Create lists
# -----------------------------

words = []
classes = []
documents = []


# -----------------------------
# 4. Process dataset
# -----------------------------

for intent in data["intents"]:

    tag = intent["tag"]

    if tag not in classes:
        classes.append(tag)

    for pattern in intent["patterns"]:

        # Tokenize sentence
        word_list = nltk.word_tokenize(pattern)

        # Store words
        words.extend(word_list)

        # Store sentence and intent
        documents.append((word_list, tag))


# -----------------------------
# 5. Stem words
# -----------------------------

words = [
    stemmer.stem(word.lower())
    for word in words
]


# -----------------------------
# 6. Remove duplicate words
# -----------------------------

words = sorted(set(words))

classes = sorted(classes)


# -----------------------------
# 7. Create training data
# -----------------------------

training_data = []


for document in documents:

    pattern_words = [
        stemmer.stem(word.lower())
        for word in document[0]
    ]

    # Create Bag of Words
    bag = []

    for word in words:

        if word in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    # Create output row
    output = [0] * len(classes)

    output[classes.index(document[1])] = 1

    training_data.append([bag, output])


# -----------------------------
# 8. Convert to NumPy arrays
# -----------------------------

X = np.array([item[0] for item in training_data])

Y = np.array([item[1] for item in training_data])


# -----------------------------
# 9. Display information
# -----------------------------

print("Training completed successfully!")

print()
print("Number of words:", len(words))
print("Number of classes:", len(classes))
print("Number of training sentences:", len(documents))

print()
print("Classes:")
print(classes)

print()
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

print()
print("Example Bag of Words:")
print(X[0])

print()
print("Example Output:")
print(Y[0])
# -----------------------------
# 10. Train the model
# -----------------------------

# Calculate weights using NumPy
weights = np.linalg.pinv(X) @ Y


# -----------------------------
# 11. Save the trained model
# -----------------------------

np.save("chatbot_model.npy", weights)

print()
print("Model trained successfully!")
print("Model saved as chatbot_model.npy")
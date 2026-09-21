import json
import random
import nltk
import numpy as np
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Load dataset
with open("intents.json", "r") as file:
    data = json.load(file)

# Load trained model
weights = np.load("chatbot_model.npy")

print("Chatbot is ready!")
# Create vocabulary
words = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words.extend(nltk.word_tokenize(pattern))

words = [
    stemmer.stem(word.lower())
    for word in words
]

words = sorted(set(words))


# Create classes
classes = sorted([
    intent["tag"]
    for intent in data["intents"]
])


# Convert user sentence into Bag of Words
def bag_of_words(sentence):

    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [
        stemmer.stem(word.lower())
        for word in sentence_words
    ]

    bag = []

    for word in words:

        if word in sentence_words:
            bag.append(1)
        else:
            bag.append(0)

    return np.array(bag)
def predict_intent(sentence):
    
    # Convert sentence to numbers
    bow = bag_of_words(sentence)

    # Calculate prediction scores
    scores = np.dot(bow, weights)

    # Find the class with the highest score
    predicted_index = np.argmax(scores)

    # Return the predicted intent
    return classes[predicted_index]
print()
print("================================")
print("      College Enquiry Bot")
print("================================")
print("Type 'quit' to exit.")
print()

while True:

    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye! Have a great day.")
        break

    intent = predict_intent(user_input)

    for item in data["intents"]:

        if item["tag"] == intent:

            response = random.choice(item["responses"])

            print("Bot:", response)

            break
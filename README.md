# AI-Chatbot-Using-NLP-and-NumPy# College Enquiry Chatbot Using NLP and NumPy

## Project Overview

This project is a simple **College Enquiry Chatbot** developed using **Python, Natural Language Processing (NLP), NLTK, and NumPy**.

The chatbot can understand basic college-related questions and provide suitable responses. It uses a small dataset of questions and intents to classify the user's input.

This project demonstrates the basic working of an NLP-based chatbot without using advanced AI models or external APIs.

---

## Features

* Answers basic college-related questions
* Uses Natural Language Processing
* Uses NLTK for tokenization and stemming
* Uses NumPy for numerical calculations
* Uses a custom training dataset
* Classifies questions into different intents
* Provides random responses for each intent
* Works through the command-line interface

---

## Technologies Used

* **Python**
* **NumPy**
* **NLTK**
* **JSON**

---

## Project Structure

```text
AI_Chatbot_Project/
│
├── chatbot.py
├── train.py
├── intents.json
├── chatbot_model.npy
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `chatbot.py`        | Runs the chatbot and responds to user questions              |
| `train.py`          | Processes the dataset and creates the chatbot model          |
| `intents.json`      | Contains training patterns, intents, and responses           |
| `chatbot_model.npy` | Stores the numerical model weights generated during training |
| `requirements.txt`  | Contains the required Python libraries                       |
| `README.md`         | Project documentation                                        |
| `.gitignore`        | Specifies files that Git should ignore                       |

---

## Dataset

The chatbot uses a custom dataset stored in `intents.json`.

The dataset contains different intents such as:

* Greeting
* Goodbye
* Courses
* Admission
* Fees
* Placements
* College Timings
* Thanks

Each intent contains:

* **Tag** – identifies the intent
* **Patterns** – example questions from the user
* **Responses** – possible chatbot responses

Example:

```json
{
  "tag": "fees",
  "patterns": [
    "What is the fee structure?",
    "How much is the college fee?",
    "Tell me about fees"
  ],
  "responses": [
    "The fee structure varies depending on the course."
  ]
}
```

---

## How the Chatbot Works

The chatbot follows these basic steps:

```text
User Input
    ↓
NLP Preprocessing
    ↓
Tokenization and Stemming
    ↓
Bag of Words
    ↓
NumPy Model
    ↓
Intent Prediction
    ↓
Select Response
    ↓
Chatbot Response
```

### 1. Tokenization

The user's sentence is divided into individual words using NLTK.

For example:

```text
"What courses are available?"
```

becomes approximately:

```text
["What", "courses", "are", "available", "?"]
```

### 2. Stemming

Words are converted to their basic stem form using the NLTK `PorterStemmer`.

### 3. Bag of Words

The words are converted into numerical values such as `0` and `1`.

For example:

```text
[0, 1, 0, 1, 0, ...]
```

This allows the text to be processed mathematically.

### 4. Model

NumPy is used to calculate numerical weights from the training data.

The model is created using:

```python
weights = np.linalg.pinv(X) @ Y
```

The resulting numerical weights are saved in:

```text
chatbot_model.npy
```

### 5. Intent Prediction

When the user asks a question, the chatbot converts the question into a Bag-of-Words vector.

NumPy calculates scores using the trained weights:

```python
scores = np.dot(bow, weights)
```

The chatbot then selects the intent with the highest score.

### 6. Response

After identifying the intent, the chatbot selects a suitable response from `intents.json`.

---

## Installation

### Step 1: Install Python

Make sure Python is installed on your computer.

Check the installation using:

```cmd
python --version
```

### Step 2: Install Required Libraries

Run:

```cmd
python -m pip install numpy nltk
```

### Step 3: Download NLTK Resources

Run:

```cmd
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Training the Chatbot

Before running the chatbot, train the model using:

```cmd
python train.py
```

The training process reads `intents.json`, processes the text, creates the numerical training data, and generates:

```text
chatbot_model.npy
```

---

## Running the Chatbot

After training, run:

```cmd
python chatbot.py
```

You should see:

```text
================================
      College Enquiry Bot
================================
Type 'quit' to exit.
```

You can then ask questions such as:

```text
You: What courses are available?
Bot: Our college offers various undergraduate and postgraduate programs.
```

Another example:

```text
You: How can I get admission?
Bot: The admission process depends on the course. Please contact the college admission office for current requirements.
```

To exit:

```text
You: quit
Bot: Goodbye! Have a great day.
```

---

## Example Questions

The chatbot can handle questions such as:

```text
Hello
What courses are available?
How can I get admission?
What is the fee structure?
Tell me about placements
What are the college timings?
Thank you
Bye
```

---

## Requirements

The required Python libraries are listed in `requirements.txt`:

```text
numpy
nltk
```

---

## Limitations

This is a simple educational chatbot and has some limitations:

* It supports only the intents included in the dataset.
* It does not use a large language model.
* It may not understand questions that are very different from the training examples.
* The information provided is based on the responses defined in the dataset.
* It does not connect to a live college database or website.

---

## Future Improvements

The project can be improved by:

* Adding more training questions
* Adding more college-related intents
* Adding confidence-based unknown-question handling
* Adding a graphical user interface
* Connecting the chatbot to a college database
* Adding voice input and output
* Improving the NLP preprocessing
* Using a more advanced machine-learning model

---

## Learning Outcomes

Through this project, I learned:

* Basics of Natural Language Processing
* Text tokenization
* Stemming
* Bag-of-Words representation
* Dataset creation
* Intent classification
* NumPy matrix operations
* Saving and loading NumPy model files
* Building a simple chatbot using Python
* Using Git and GitHub for project version control

---

## Author

**Amshavarthen**

GitHub Repository:

**AI Chatbot Using NLP and NumPy**

---

## Conclusion

This project demonstrates how a simple chatbot can be created using Python, NLP techniques, and NumPy.

The chatbot takes a user's question, processes the text, converts it into numerical form, predicts the appropriate intent, and provides a predefined response.

It is a beginner-friendly project for understanding the basic concepts behind NLP-based chatbot systems.

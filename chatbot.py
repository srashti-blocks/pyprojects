import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download('punkt_tab')
nltk.download('wordnet')

# Chatbot intents data
data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["HELLO", "Hi there", "Hi", "what's up"],
            "responses": ["hello",  "hi", " HI !Long time no see!", "yo"]
        },
        {
            "tag": "date",
            "patterns": ["what are you doing this weekend", "do you wanna hangout some time", "so, are you free on sunday we can have a cup of tea"],
            "responses": ["I'm not sure", "I am available this week", "I don't have any plans"]
        },
        {
            "tag": "name",
            "patterns": ["what's your name?", "what are you called", "who are you"],
            "responses": ["my name is Kiko", "I'm Kiko", "Kiko"]
        },
        {
            "tag": "goodbye",
            "patterns": ["cya", "g2g", "adios", "bye"],
            "responses": ["see you later", "speak soon", "It was nice speaking to you"]
        }
    ]
}

lemmatizer = WordNetLemmatizer()
words = []
classes = []
doc_x = []
doc_y = []



for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))


training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(doc_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for w in words:
        bow.append(1 if w in text else 0)
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    training.append([bow, output_row])
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))








model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=500, verbose=1)




def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    return tokens
def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    res = model.predict(np.array([bow]))[0]
    thresh = 0.3
    y_pred = np.argmax(res)
    if res[y_pred] > thresh:
        return [labels[y_pred]]
    else:
        return ["noanswer"]


def get_response(intents_list, intents_json):
    tag = intents_list[0]
    if tag == "noanswer":
        return "Sorry, I didn't understand that. Can you rephrase?"
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])



while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Chatbot: Goodbye!")
        break
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print("Chatbot:",result)
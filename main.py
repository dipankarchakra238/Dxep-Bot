import datetime
import requests
from urllib import request
import webbrowser
import os
import json
import random

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

nltk.download('punkt_tab')
nltk.download('wordnet')


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path

        self .documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        words = nltk.word_tokenize(text)
        lemmatizer = nltk.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r', encoding='utf-8') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump(
                {'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(
            dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(
            torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        # Confidence threshold
        probs = torch.softmax(predictions, dim=1)
        confidence = probs[0][predicted_class_index].item()

        # Fuzzy matching: check if user input closely matches any pattern
        max_similarity = 0
        best_intent = None
        for intent in self.intents:
            for pattern in self.intents_responses.get(intent, []):
                similarity = self._similarity(
                    input_message.lower(), pattern.lower())
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_intent = intent

        # If similarity is high, use best intent
        if max_similarity > 0.7 and best_intent:
            predicted_intent = best_intent

        # Only respond with predicted intent if confidence is high
        if confidence > 0.7:
            if self.function_mappings:
                if predicted_intent in self.function_mappings:
                    func = self.function_mappings[predicted_intent]
                    try:
                        return func(input_message)
                    except TypeError:
                        return func()
            if self.intents_responses.get(predicted_intent):
                return random.choice(self.intents_responses[predicted_intent])
        # Otherwise, fallback
        try:
            from log_unknown import log_unknown_question
            log_unknown_question(input_message)
        except Exception:
            pass
        return "I'm not sure how to answer that yet, but I'm learning more every day!"

    def _similarity(self, a, b):
        # Simple similarity: ratio of common words
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0
        return len(set_a & set_b) / max(len(set_a), len(set_b))


def open_google():
    webbrowser.open('https://www.google.com')
    return "Redirecting to Google..."


def get_weather(user_message=None):
    # Try to extract location from user_message
    location = "London"
    if user_message:
        # Simple extraction: look for 'in <location>' or after 'weather'
        import re
        match = re.search(r'in ([A-Za-z ]+)', user_message)
        if match:
            location = match.group(1).strip()
        else:
            # Try to find a word after 'weather'
            parts = user_message.lower().split('weather')
            if len(parts) > 1:
                possible = parts[1].strip()
                if possible:
                    location = possible.split()[0].capitalize()
    try:
        response = requests.get(f'https://wttr.in/{location}?format=3')
        if response.status_code == 200:
            return response.text
        else:
            return f"Unable to fetch weather for {location} right now."
    except Exception:
        return f"Unable to fetch weather for {location} right now."


def get_datetime(user_message=None):
    date_str = f"Today's date is {datetime.date.today().strftime('%A, %d %B %Y')}"
    time_str = f"Current time is {datetime.datetime.now().strftime('%H:%M:%S')}"
    msg = user_message.lower() if user_message else ''
    if 'date' in msg and 'time' in msg:
        return f"{date_str}\n{time_str}"
    elif 'date' in msg:
        return date_str
    elif 'time' in msg:
        return time_str
    else:
        return date_str


def get_stocks():

    stocks = ['AAPL', 'GOOGL', 'MSFT']

    print(random.sample(stocks, 3))


if __name__ == '__main__':
    assistant = ChatbotAssistant(
        'intents.json', function_mappings={
            'stocks': get_stocks,
            'google': open_google,
            'weather': lambda msg=None: get_weather(msg),
            'datetime': lambda msg=None: get_datetime(msg)
        })
    # assistant.parse_intents()
    # assistant.prepare_data()
    # assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    # assistant.save_model('chatbot_model.pth', 'dimensions.json')

    # assistant.parse_intents()
    # assistant.load_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message: ')

        if message.lower() in ['exit', 'quit']:
            print("Exiting the chat. Goodbye!")
            break

        print(assistant.process_message(message))


# chatbot = ChatbotAssistant('intents.json')
# print(chatbot.tokenize_and_lemmatize("Hello, how are you?"))


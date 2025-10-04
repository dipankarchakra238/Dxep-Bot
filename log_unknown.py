import json
from datetime import datetime

def log_unknown_question(question, log_path='unknown_questions.json'):
    entry = {
        'question': question,
        'timestamp': datetime.now().isoformat()
    }
    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(entry)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage in your ChatbotAssistant.process_message method:
# if fallback response is returned, call log_unknown_question(input_message)

from log_unknown import log_unknown_question
from flask import Flask, render_template, request, jsonify
from main import get_stocks, open_google, get_weather, get_datetime
from main import ChatbotAssistant
import os
import re
from markupsafe import Markup


def linkify(text):
    url_pattern = r'(https?://[^\s]+)'
    return re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)


app = Flask(__name__)


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    question = data.get('question')
    if question:
        log_unknown_question(question)
    return jsonify({'status': 'logged'})


assistant = ChatbotAssistant(
    'intents.json', function_mappings={
        'stocks': get_stocks,
        'google': open_google,
        'weather': lambda msg=None: get_weather(msg),
        'datetime': lambda msg=None: get_datetime(msg)
    })

assistant.parse_intents()
assistant.prepare_data()
assistant.train_model(batch_size=8, lr=0.001, epochs=100)
assistant.save_model('chatbot_model.pth', 'dimensions.json')


# assistant.parse_intents()
# assistant.prepare_data()
# assistant.load_model('chatbot_model.pth', 'dimensions.json')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = assistant.process_message(user_message)
    response = linkify(response)
    return jsonify({'response': Markup(response)})


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


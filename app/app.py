import os
import joblib

import nltk
from tasks.tokenize_input import tokenize

from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading punkt_tab")
    nltk.download('punkt_tab')

models = dict()

for filename in os.listdir('models/word2vec'):
    if 'pipeline' in filename:
        modelName = filename.split('_pipeline')[0]
        models[modelName] = os.path.join('models/word2vec/', filename)

for model in models:
    models[model] = joblib.load(models[model])

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    content = request.get_json(force=True)

    text = content['text']

    answer_prediction = dict()

    for _model in models:
        tokens = tokenize(text)
        prediction = models[_model].predict(tokens)
        answer_prediction[_model] = int(prediction[0])
        if _model != 'SVC':
            answer_prediction[_model+'_proba'] = models[_model].predict_proba(tokens)[0].tolist()

    response = jsonify(answer_prediction)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

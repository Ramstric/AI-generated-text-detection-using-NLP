from scripts import  model_prediction

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    content = request.get_json(force=True)

    text = [content['text']]
    #text = [''.join(request.form['text'])]

    predictions = model_prediction.predict(text)

    response = jsonify(predictions)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

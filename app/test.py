import os
import joblib
import nltk
from tasks.tokenize_input import tokenize

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

for filename in os.listdir('models/word2vec/'):
    modelName = filename.split('_pipeline')[0]
    models[modelName] = os.path.join('models/word2vec/', filename)

for model in models:
    models[model] = joblib.load(models[model])

def predict(text, model):
    tokens = tokenize(text)
    prediction = model.predict(tokens)
    return prediction[0]

print(predict("This is a test", models['LogisticRegression']))
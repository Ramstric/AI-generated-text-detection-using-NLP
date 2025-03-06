import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import pickle

available_models = os.listdir('./models/')

models = list()

for model in available_models:
    with open(f'models/{model}', 'rb') as f:
        models.append(pickle.load(f))

def predict(text):
    predictions = dict()

    for i in range(len(models)):
        predictions[available_models[i].split('.')[0]] = int(models[i].predict(text)[0])

    return predictions
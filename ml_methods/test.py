import pickle

# Load the models
with open('models/naive_bayes_model.pkl', 'rb') as f:
    nb = pickle.load(f)

with open('models/random_forest_model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('models/logistic_regression_model.pkl', 'rb') as f:
    lr = pickle.load(f)

with open('models/support_vector_model.pkl', 'rb') as f:
    svc = pickle.load(f)


test_abstract = ["This study investigates the efficacy of transformer-based models in generating coherent and contextually relevant research abstracts. We analyze a corpus of AI-generated abstracts against a benchmark dataset of human-authored abstracts, focusing on metrics such as lexical diversity, semantic coherence, and adherence to established academic writing conventions. Preliminary findings indicate that while AI models can produce syntactically sound abstracts, challenges remain in capturing the nuanced argumentation and critical insights characteristic of human scholarship. We explore potential avenues for refining these models to bridge this gap, including the integration of domain-specific knowledge and improved contextual understanding."]

print("\n\tWhere 0 is Human Generated and 1 is AI Generated...")
print("\n\tRandom Forest Classifier prediction:", rf.predict(test_abstract))
print("\n\tNaive Bayes Classifier prediction:", nb.predict(test_abstract))
print("\n\tLogistic Regression Classifier prediction:", lr.predict(test_abstract))
print("\n\tSupport Vector Classifier prediction:", svc.predict(test_abstract))




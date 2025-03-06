import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the data
data = pd.read_csv('dataset/data_set.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['abstract'], data['is_ai_generated'], random_state=0)

rf = Pipeline([('vectorizer', CountVectorizer()), ('classifier', RandomForestClassifier())])  # Create a Random Forest Classifier
nb = Pipeline([('vectorizer', CountVectorizer()), ('classifier', MultinomialNB())])  # Create a Naive Bayes Classifier
lr = Pipeline([('vectorizer', CountVectorizer()), ('classifier', LogisticRegression())])  # Create a Logistic Regression Classifier
svc = Pipeline([('vectorizer', CountVectorizer()), ('classifier', SVC())])  # Create a Support Vector Classifier

# Train the models
rf.fit(X_train, y_train)
nb.fit(X_train, y_train)
lr.fit(X_train, y_train)
svc.fit(X_train, y_train)

# Save the models
with open('models/naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/support_vector_model.pkl', 'wb') as f:
    pickle.dump(svc, f, protocol=pickle.HIGHEST_PROTOCOL)


# Predict the labels
predictions_rf = rf.predict(X_test)
predictions_nb = nb.predict(X_test)
predictions_lr = lr.predict(X_test)
predictions_svc = svc.predict(X_test)

# Print the classification report
print('\n\tRandom Forest Classifier:')
print(classification_report(y_test, predictions_rf))
print('\n\tNaive Bayes Classifier:')
print(classification_report(y_test, predictions_nb))
print('\n\tLogistic Regression Classifier:')
print(classification_report(y_test, predictions_lr))
print('\n\tSupport Vector Classifier:')
print(classification_report(y_test, predictions_svc))

# Confusion Matrix plot
cm_rf = confusion_matrix(y_test, predictions_rf)
cm_nb = confusion_matrix(y_test, predictions_nb)
cm_lr = confusion_matrix(y_test, predictions_lr)
cm_svc = confusion_matrix(y_test, predictions_svc)

fig_rf = px.imshow(cm_rf, labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['AI Generated', 'Human Generated'], y=['AI Generated', 'Human Generated'],
                   title='Random Forest Classifier', text_auto=True,
                   color_continuous_scale='Burgyl')
fig_rf.show()

fig_nb = px.imshow(cm_nb, labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['AI Generated', 'Human Generated'], y=['AI Generated', 'Human Generated'],
                   title='Naive Bayes Classifier', text_auto=True,
                   color_continuous_scale='Burgyl')
fig_nb.show()

fig_lr = px.imshow(cm_lr, labels=dict(x="Predicted", y="Actual", color="Count"),
                     x=['AI Generated', 'Human Generated'], y=['AI Generated', 'Human Generated'],
                     title='Logistic Regression Classifier', text_auto=True,
                     color_continuous_scale='Burgyl')
fig_lr.show()

fig_svc = px.imshow(cm_svc, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['AI Generated', 'Human Generated'], y=['AI Generated', 'Human Generated'],
                        title='Support Vector Classifier', text_auto=True,
                        color_continuous_scale='Burgyl')
fig_svc.show()

test_abstract = ["This study investigates the efficacy of transformer-based models in generating coherent and contextually relevant research abstracts. We analyze a corpus of AI-generated abstracts against a benchmark dataset of human-authored abstracts, focusing on metrics such as lexical diversity, semantic coherence, and adherence to established academic writing conventions. Preliminary findings indicate that while AI models can produce syntactically sound abstracts, challenges remain in capturing the nuanced argumentation and critical insights characteristic of human scholarship. We explore potential avenues for refining these models to bridge this gap, including the integration of domain-specific knowledge and improved contextual understanding."]

print("\n\tWhere 0 is Human Generated and 1 is AI Generated...")
print("\n\tRandom Forest Classifier prediction:", rf.predict(test_abstract))
print("\n\tNaive Bayes Classifier prediction:", nb.predict(test_abstract))
print("\n\tLogistic Regression Classifier prediction:", lr.predict(test_abstract))
print("\n\tSupport Vector Classifier prediction:", svc.predict(test_abstract))




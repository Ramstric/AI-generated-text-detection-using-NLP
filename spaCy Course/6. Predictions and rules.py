#               10 June 2024
#           spacY Course exercises

import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_md")

# In spaCy the statistical models are used for:
# Applications that need to generalize based on a few examples
# Such as product names, person names and relationships between subjects and objects
# Which invoves the entity recognizer, dependency parser and part-of-speech tagger

# On the other hand, the rule-based systems are used for:
# Applications that need a dictionary of finite examples
# Such as countries, cities, drug names and dog breeds
# Which involves the tokenizer, matcher and phrase matcher

# Initialize with the shared vocab

matcher = Matcher(nlp.vocab)

# Patterns are lists of dictionaries describing the tokens
pattern = [{"LEMMA": "love", "POS": "VERB"}, {"LOWER": "cats"}]
matcher.add("LOVE_CATS", [pattern])

# Operators can specify how often a token should be matched
pattern = [{"TEXT": "very", "OP": "+"}, {"TEXT": "happy"}]
matcher.add("VERY_HAPPY", [pattern])

# Inlining the pattern
matcher.add("DOG", [[{"LOWER": "golden"}, {"LOWER": "retriever"}]])

# Calling matcher on doc returns list of (match_id, start, end) tuples
doc = nlp("I love cats and I'm very very happy. Also, I have a Golden Retriever.")
matches = matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print("Matched span:", span.text)
    # Get the span's root token and root head token
    print("Root token:", span.root.text)
    print("Root head token:", span.root.head.text)
    # Get the previous token and its POS tag
    print("Previous token:", doc[start - 1].text, doc[start - 1].pos_, "\n")

# For a more efficient version of the matcher, use the PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)

pattern = nlp("Golden Retriever")
matcher.add("DOG", [pattern])
doc = nlp("I have a Golden Retriever")

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Get the matched span
    span = doc[start:end]
    print("Matched span:", span.text)
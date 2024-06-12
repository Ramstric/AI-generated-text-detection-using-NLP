#               10 June 2024
#           spacY Course exercises

import spacy

nlp = spacy.load("en_core_web_md")

# ---------------------------- Document similarity ----------------------------

doc1 = nlp("It's a warm summer day")
doc2 = nlp("It's sunny outside")

# Get the similarity of doc1 and doc2
similarity = doc1.similarity(doc2)
print("Similarity between doc1 and doc2: ", similarity)

# ---------------------------- Token similarity ----------------------------

doc = nlp("TV and books")
token1, token2 = doc[0], doc[2]

# Get the similarity of the tokens "TV" and "books"
similarity = token1.similarity(token2)
print("Similarity between token1 and token2: ", similarity)

# ---------------------------- Span similarity ----------------------------

doc = nlp("This was a great restaurant. Afterwards, we went to a really nice bar.")

# Create spans for "great restaurant" and "really nice bar"
span1 = doc[3:5]
span2 = doc[12:15]

# Get the similarity of the spans
similarity = span1.similarity(span2)
print("Similarity between span1 and span2: ", similarity)

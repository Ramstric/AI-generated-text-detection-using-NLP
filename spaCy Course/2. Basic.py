#               10 June 2024
#           spacY Course exercises

import spacy

nlp = spacy.blank("en")             # English NLP object

doc = nlp("This is a sentence.")    # Process the text

span = doc[1:3]                     # Slice the Doc to get a Span object

# Print the text of the tokenized document
print("----[ Basic document tokenization ]----")
for token in doc:
    print(token.text)

# Print the text of the span
print("\n----[ Span from token 1 to token 3 ]----")
print(span.text)


doc = nlp("This example contains percentages like 55% and 2%.")  # Process the text

# Finding percentages in the text
print("\n----[ Finding percentages in the text ]----")
for token in doc:
    if token.text == "%":
        print("Percentage found: " + doc[token.i - 1].text)


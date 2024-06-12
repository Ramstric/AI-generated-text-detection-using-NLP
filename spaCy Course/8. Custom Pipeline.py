#               10 June 2024
#           spacY Course exercises

import spacy
from spacy.language import Language

# Create the nlp object
nlp = spacy.load("en_core_web_sm")


# Define a custom component
@Language.component("custom_component")
def custom_component_function(doc):
    # Print the doc's length
    print("Doc length:", len(doc))
    # Return the doc object
    return doc


# Add the component first in the pipeline
nlp.add_pipe("custom_component", first=True)

# Process a text
doc = nlp("Hello world!")

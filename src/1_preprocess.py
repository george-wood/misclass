import pandas as pd
import spacy
from spacy.language import Language
import numpy as np
import en_core_web_sm
import wordninja # for splitting tokens lacking whitespace
import jamspell
import contextualSpellCheck
import re

# spelling correction
corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel("aux/en.bin")

# jamspell (https://github.com/bakwc/JamSpell/) works OK but fails on "onicer"
# can we improve this?
corrector.FixFragment(
  "alleged that the accusea onicer nandcuffed the victim too tightly."
)

# the pipeline (see: https://spacy.io/usage/processing-pipelines)
# tok2vec > stopwords > tagger > parser > ner > attribute_ruler > lemmatizer
# what else do we need?

# read data
narratives = pd.read_csv("../data/narratives.csv")
intake = narratives.column_name.str.contains('take')
narratives = narratives.loc[intake, ["cr_id", "column_name", "text"]]
narratives = narratives.drop_duplicates()
df = narratives[0:50].copy()

# preprocessing - removing repeating substrings
# source: https://stackoverflow.com/questions/29481088/how-can-i-tell-if-a-string-repeats-itself-in-python
@Language.component("repeats")
def repeats(doc):
  s = doc.text.lower()
  i = (s+" "+s).find(s, 1, -1)
  if i == -1:
    doc = doc
  else:
    doc = nlp.make_doc(s[:i-1])
  return(doc)

# component for removing stop words
@Language.component("stopwords")
def stopwords(doc):
  doc = [t.text for t in doc if not t.is_stop]
  doc = nlp.make_doc(' '.join(map(str, doc)))
  return(doc)

# component for removing punctuation
@Language.component("punctuation")
def punctuation(doc):
  doc = [t.text for t in doc if (not t.is_punct and not t.is_space)]
  doc = nlp.make_doc(' '.join(map(str, doc)))
  return(doc)

# set up NLP
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("stopwords", name = "stopwords", before = "tagger") # add stopword remover to pipeline
# add punctuation remover to pipeline
nlp.add_pipe("punctuation", name = "punctuation", before = "tagger")
# add repeats remover to pipeline
nlp.add_pipe("repeats", name = "repeats", before = "tagger")

# add contextual spellchecker to pipeline
# note: putting this at the end of the pipeline because that's what the docs do,
# could see if there is a better place to put it
nlp.add_pipe("contextual spellchecker")

print(nlp.pipe_names)

# run pipeline
docs = list(nlp.pipe(df["text"]))

# compare
print(df["text"][0:5])
print(docs[0:5])

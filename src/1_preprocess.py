import pandas as pd
import spacy
from spacy.language import Language
import numpy as np
import en_core_web_sm
import wordninja # for splitting tokens lacking whitespace
import jamspell

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
narratives = pd.read_csv("data/narratives.csv")
intake = narratives.column_name.str.contains('take')
narratives = narratives.loc[intake, ["cr_id", "column_name", "text"]]
narratives = narratives.drop_duplicates()
df = narratives[0:50].copy()

# component for removing stop words
@Language.component("stopwords")
def stopwords(doc):
  doc = [t.text for t in doc if not t.is_stop]
  doc = nlp.make_doc(' '.join(map(str, doc)))
  return(doc)

# set up NLP
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("stopwords", name = "stopwords", before = "tagger") # add stopword remover to pipeline
nlp.pipe_names

# run pipeline
docs = list(nlp.pipe(df["text"]))

# compare
df["text"][1]
docs[1]

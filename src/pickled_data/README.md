# Pickeled data

This folder contains pickled data produced by notebooks in the model development folder as we as an example for how to extract the data frame and gensim objects from each pickle.  Each picked item contains a tuple containing the associated **lemmatized text dataframe, gesim corpus, and gensim gensim [dictionary](https://radimrehurek.com/gensim/corpora/dictionary.html)**.

The df_lemmatized_texts.pkl contains all the preprocessed data in a pythonic format for third-party usage.
- **_columns:_**
    - _Cr_id:_ unique identifier from raw dataframe.
    - _Column_name:_ complaint type from raw dataframe.
    - _Text:_ raw text from raw dataframe.
    - _Bag_of_lemmas:_ list objects containing extracted lemmas for each document.
    - _BoL_length_ length of bag of lemmas.
    - _Row_number:_ row entry.
    - _Gensim_nogram:_ dictionary object representation of lemmatized text processed with no-gram implementation.
    - _Gensim_bigram_ dictionary object representation of lemmatized text processed with bi-gram implementation.
    - _Gensim_trigram_ dictionary object representation of lemmatized text processed with tri-gram implementation.

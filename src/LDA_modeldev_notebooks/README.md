# LDA Model Development Notebooks

This folder contains four jupyter notebooks used preprocess and model raw textual data from intake allegations.
- **LDA_combo_processing_only.ipynb**: 
    - preprocesses dataframe of raw textual data into lemmatized bag of words format.  Final product is the original  dataframe with appended columns containing dictionary objects representing vectorized representations of the text.  There are implementations for bigram, trigram, and nogram methods.
- **LDA_combo_modeling_only_nogram.ipynb**:
    - takes preprocessed dataframe dictionary objects from the "gensim_nogram" column and applies an LDA model.
- **LDA_combo_modeling_only_bigram.ipynb**:
    - takes preprocessed dataframe dictionary objects from the "gensim_nogram" column and applies an LDA model.
- **LDA_combo_modeling_only_trigram.ipynb**: 
    - takes preprocessed dataframe dictionary objects from the "gensim_nogram" column and applies an LDA model.
    
### LDA_combo_processing_only.ipynb:

- #### Preprocessing Pipeline:
    - **Input normalization**: normalizes raw textual data before tokenization while conserving as much information as possible.
		- **_Text Lowering_**: removes capitalization of all text
		- **_Accent Striping_**: replaces accented non-ASCII characters with thier non-accented ASCII character.
		- **_Whitespace Stripping_**:  strips non-space white spaces (tabs, paragraph-breaks, etc.)
        - **_Remove Identical Repeats_**: if a document has two idential pharases, we remove one of the duplicates.  Does not work if two phrases are not identical or if there is more that one repeat.
	- **Tokenization**: Takes normalized text and implements various methods for tokenization.
		- _**Spacy Tokenization:**_
            - _Filter word length:_ remove words with less than three characters.
			- _Remove stop words:_ remove spacy defined stop words
			- _Remove punctuation:_
			- _Filter Part of Speech Tags:_ retains tokens with 'NOUN', 'ADJ', 'VERB', and 'ADV' POS tags only.
			- _Extract lemmas:_ Lemma remaining tokens and store in a list.
        - _**Document Length Filtering**_:
            - removes document inputs with less than 10 lemmas.
		- _**Genism Processing**_:
			- _N-Gram construction_: construct bigram and trigram corpora from origional. 
			- _Extreme filtering_: create gensim corpus and corpus-dictionary objects.  filter our items that appear in less than 20 documents or more than 50% of documents.
- #### Output:
    - **Dataframe:** 
        - **_path_**: 
            - ~/src/pickled_data/df_lemmatized_texts.pkl
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

        
        
### LDA_combo_modeling_only_*.ipynb:

no-gram, bi-gram, and tri-gram have idential implemntations and evaluation methods.

- #### Modeling
    - implements [gensims lda model](https://radimrehurek.com/gensim/models/ldamodel.html) for 5, 10, and 15 topics hyper parameters
- #### Evaluation Frameworks
    - **Coherence Score:** Uses [gensims topic coherence pipeline](https://radimrehurek.com/gensim/models/coherencemodel.html) for different topic number parameters.
    - **Topic Component Distribution:** Displays each topic as a collection of word contribution in descending order.  A "good" topic will likely have its highest contributors be from a similar topic.
    - **pyLDAvis pricipal component visualization:** Visualization of each topic on a two dimentional principal component space using [pyLDAvis](https://github.com/bmabey/pyLDAvis) package.  A "good" model should have distributed topics with little clustering or overlap.
    - **Most Representative Documents per Topic:** Determined which documents have the highest contribution from a specific topic and lists raw texual data in decending order.  Good for developing intution for what the model is incentivized to develop topics around.
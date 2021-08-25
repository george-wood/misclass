#### LDA Model Development Notebooks

This folder contains the jupyter node books used to construct lemmatatized data set for LDA modeling.

- Preprocessing Pipeline:
    - Input normalization:
		- Text lowering
		- Accent Striping 
		- Whitespace stripping
        - Remove identical repeats
	- Tokenization:
		- Spacy Tokenization:
			- Remove stop words
			- Remove punctuation
			- Extract lemmas
			- Produce bag of lemmas
			- Filter bag lengths
		- Genism Processing:
			- N-Gram construction
			- Extreme filtering
	- Composite Data Frame Columns:
		- Cr_id
		- Column_name
		- Text
		- Bag_of_lemmas
		- BoL_length
		- Row_number
		- Gensim_nogram
		- Gensim_bigram
        - Gensim_trigram
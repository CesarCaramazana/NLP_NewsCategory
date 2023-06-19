# A comparative analysis of classical Machine Learning methods and a Transformer architecture for the classification of News articles



In this final project of Natural Language Processing we will be carrying a supervised classification task over a News dataset after applying three different text vectorization techniques –TF-IDF, Word Embeddings (W2V) and LDA Topic representation– and comparing the performance of classical Machine Learning methods versus a fine-tuned Transformer from the Hugging Face library.


## The dataset

The source of the data is the News Category Dataset, a public dataset containing 210000 news headlines from 2012 to 2018 from the digital newspaper HuffPost, as well as a brief description, the author, the URL, the date and the label on the category of the piece of news. All the text is written in English. We generated our final set as a subpartition of 16500 samples with the body of the articles retrieved from the original webpage. Each sample was processed with the following steps:
1) Valid Part-of-Speech (nouns, verbs, pronouns and adjectives)
and alpha-numeric filtering.
2) Generic stopword removal.
3) Specific stopword removal, to eliminate a disclaimer
that appeared in every article, ”By entering your email
and clicking Sign Up, you’re agreeing to let us send
you customized marketing messages about us and our
advertising partners”, and other undesirable terms
from the HTML that spoilt the topic modeling task.
4) Tokenization: with lemmatization and to lower case.
5) Bi-grams detection: to merge words that commonly
appear together in the data, such as ”New York”,
”North Korea” or ”White House”.

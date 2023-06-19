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


## Text vectorization

1) TF-IDF: The TF-IDF matrix was computed automatically using Gensim’s implementation. This method returns a very sparse representation of each document, for a vocabulary size of 21184 words.
2) Word embeddings: We decided to train a Fast Text model with vector size 200 and window size 5 to be robust against out-of-vocabulary words. The document
embedding V is then calculated averaging the embeddings of the words in the document weighted by their tf-idf factor. With this strategy we control the contribution of common versus uncommon words in the embedding space.

3) LDA Topic representation: The LDA topic representation was carried out using Mallet [4] and in two steps. First, an exploratory phase to search for the number of topics with the highest coherence score and to inspect the resulting words per topic. This phase was iterated over several times, preprocessing the text after inspecting the outcome to obtain better coherence (mainly stopword removal, common-words filtering and bi-grams detection). Second, a re-training of the best model (n = 40) for a greater number of iterations. 


## Machine Learning methods

### Classical methods
Three classification paradigms were chosen from the Scikit-learn library to serve as the baseline of classical Machine Learning methods. These are: Support Vector Machines (SVM), Random Forest classifiers (RF) and Multilayer Perceptrons (MLP).
- SVM: with a RBF kernel. The parameter C was crossvalidated.
- RF: the number of trees and their depths were crossvalidated.
- MLP: with 3 hidden layers of sizes 128, 256 and 128. The Adam algorithm was used to optimize the weights.


### Transformer
Regarding the Transformer architecture, a pretrained DistilBERT model [6] was fine-tuned for 5 epochs. Some minor tweaks had to be done to accommodate for our multi-label classification task. The appropriate data preprocessing was also carried out to fit the required format of the model (such as one-hot encoding the labels and tokenizing the inputs).






## Summary of the discussion

Out of all the classical Machine Learning models, the Random Forest classifiers provided the worst performance overall, particularly for the TF-IDF input, due to the lack of expressiveness. The Support Vector Machines and the MLPs yielded similar results, and scored a bit lower than the DistilBERT fine-tuned Transformer, with which the best test accuracy of 0.829 was achieved. We analyzed the short-comings of each vector representation, such as the large dimensionality of the TF-IDF input matrix or the need to set beforehand the number of topics for the LDA model. A comparison of the computational cost was carried out in terms of average inference time: the Transformer was the most costly method due to the complexity of the architecture while the classical ML classifiers were much more light-weighted. We discovered that many errors in the classification are related to the ambiguity and the class imbalance of the data. Finally, we found out that the Machine Learning method had a greater impact in performance than the text vectorization technique, although it is the latter that influenced the most on the computational cost of the inference process, as the input dimensionality is very heterogeneous for the different representations.

Some possible lines of improvement involve: the crossvalidation of the number of topics in LDA as a hyperparamenter of the classification pipeline, a finer search of the threshold for the detection of N-grams or the deepening in other Transformer architectures, as they outperform the classical approaches we took in this project.


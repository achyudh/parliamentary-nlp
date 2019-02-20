# Parliamentary NLP
> Identifying the trends in Indian Legislative Debates over the last decade

This repository contains the code pertaining to various NLP techniques that were used in the textual analysis of the Indian parliamentary debates. We collected data from Lok Sabha and Rajya Sabha since 2007, making up 10 years' worth of debates. The data collection was made using a web crawler and is not featured in this repository.

## Approach:
* Identified the latent structures within the synopses of the parliamentary debates
* Used multidimensional scaling to reduce dimensionality within the corpus and used K-means clustering to identify similarities
* Performed hierarchical clustering on the corpus (structured according to the year and house) using Ward clustering
* Implemented the Rapid Automatic Keyword Extraction (RAKE) algorithm for key-phrase extraction
* Used collocation finders to extract n-grams from the corpus containing the parliamentary debates
* Conducted Vector Space Modelling and Latent Dirichlet Allocation for topic modelling and compared the results

## Results:
I'll include only the K-Means clustering result here, due to space constrains, and include the rest in a report with all the analyses that I have done and plan to do with this dataset. Both kinds of clustering indicate that there is little correlation, if at all, between the debates at both the parliamentary houses. Further, all the budget sessions have a very similar content and this is reflected in the document clustering also as there is significant clustering overlap between the budget sessions.
![K-Means Clustering](https://github.com/achyudhk/Parliamentary-Debate-NLP/blob/master/data/KMeans_Clustering.png)

## Prerequisites:
The project makes use of algorithms implemented in the Scikit-learn library for multi-dimensional scaling, evaluation metrics, K-Means and hierarchial clustering and Matplitlib for plotting the results. Stopword removal, Snowball Stemmer and WordNet Lemmatizer from the NLTK library were used to preprocess the corpora. Other dependencies include Pandas, Numpy, Scipy and Gensim.

## Contributing:
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change. Ensure any install or build dependencies are removed before the end of the layer when doing a build. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.

## License:
This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments:
I thank Dr. Anoop Kumar for his guidance and providing some of the ideas implemented in this project.

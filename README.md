# stackoverflow_kaggle

This repository contains a jupyter notebook describing my experiments on the public stackoverflow questions/answers dataset.

My solution is based on a tf-idf on question titles and contents on a vocabulary of 18 000 words followed by dimensionality reduction (size 100). Cosine similarity is computed between test and train questions and respondents to the 200 closest train questions are assigned to each test question, weighted on the similarity scores. It is sufficient to assign 20 users by question.

It is enhanced with a collaborative filtering algorithm (ALS from pyspark MLlib) trained on the train questions to create continuous scores on all users for each train question, instead of only taking the few respondents (often 1) to those train questions.

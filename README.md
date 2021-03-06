# stackoverflow_kaggle

This repository contains a jupyter notebook describing my experiments on the public stackoverflow questions/answers dataset.

My "simple" solution is based on a tf-idf on questions' title and content on a vocabulary of 18 000 words followed by dimensionality reduction (size 100). Cosine similarity is computed between test and train questions and respondents to the 200 closest train questions are assigned to each test question, weighted on the similarity scores, after filtering by date and the asker's user id. It is sufficient to assign 20 users by question. I also suggested to try to replace tf-idf by a sentence transformers model, and mentioned some potential advantages and drawbacks of it. The crossval metrics are the following:

<b>precision@1</b>: 0.040229111210084205 (approx. 4% of train questions have found a respondent in the top 1 predictions)</br>
<b>precision@5</b>: 0.01950345427615565 (approx. 10% of train questions have found a respondent in the top 5 predictions)</br>
<b>precision@20</b>: 0.007589360622287591 (approx. 15% of train questions have found a respondent in the top 20 predictions)</br>
<b>mean_reciprocal_rank</b> (approximated): 0.06493295795350325

I tried to enhance it with a collaborative filtering algorithm (ALS from pyspark MLlib) trained on the train questions to create continuous scores for all users on each train question, instead of only taking binary 0-1 scores on the few respondents (often 1 or 2) to those train questions. The crossval metrics were disappointing and clearly worse, giving the impression that always the same few users respond to the same kinds of questions.

<b>precision@1</b>: 0.0018353755165877758</br>
<b>precision@5</b>: 0.0011811863651900111</br>
<b>precision@20</b>: 0.0005237863424031819</br>
<b>mean_reciprocal_rank</b> (approximated): 0.003661054235073217

Besides, what could explain this disappointing result is I also struggled with pyspark cross-validation on this ALS model, the training time being very long, and chose a model with no regularization and 30 hidden features because I had the impression the model underfit the train data but I did not want the model to train for too long if too many hidden features were set.

I suggested further experiments, such as taking user features into account (About me section, reputation, up votes, down votes) and training a linear merger (like Logistic Regression) on those features coupled with features from the "simple" solution, or replace tf-idf by a sentence transformers model. I hope to have time to try those improvement ideas soon.

<b>Environment</b>: Python 3.8

<b>Required packages</b>:</br>
- pandas==1.3.3
- numpy==1.20.3
- tqdm==4.62.3
- nltk==3.6.3
- bs4==0.0.1
- sklearn==0.0
- sentence_transformers==2.1.0
- scipy==1.7.1
- hnswlib==0.5.2
- findspark==1.4.2
- pyspark==3.1.2

The notebook also needs to set up spark on the device.

The [notebook](https://drive.google.com/file/d/1RuAohnx4X7vx6-K7rIUlR5cKO064B3iz/view?usp=sharing) and the [submissions](https://drive.google.com/file/d/1GSBXA5DXhTJokBFHDE2dnI2NyPo07DOI/view?usp=sharing) are also available here for download.

# spam_detection
CISC-3440
Project-Spam Detection

Introduction: 
We are trying to implement an end to end ML model to solve a real world issue, specifically spam email detection.The objective of ML is to determine which model has the best performance rate to categories spam and ham email. The reason to do this is simple: by detecting unsolicited and unwanted emails, we can prevent spam messages from creeping into the user’s inbox, thereby improving user experience. Sometimes spam emails contain phishing and malware attacks. People can easily fall victim to it.it is of no importance but can cause damages like financial fraud to identity theft.It prevents users from productivity and also has destructive effects on storage capacity of email server and network bandwidth .

Dataset:
The dataset is Lingspam.csv found on kaggle.(as suggested).It contains 3 columns namely, body,label and unnamed.the body contains the text of the email.the label consist of two values 0 for spam and 1 for not spam emails,and unnamed is mostly numbering of the emails which is not very useful for our training a ML model.The shape of the dataset is (2605,3). Through our initial analysis of the dataset we see that there are no null values and duplicate values.Column “body” is of object type and “label” is a categorical column of int.

Data cleaning and preprocessing steps:
When we observe these huge amounts of text in our dataset, we have to clean those texts before converting them to vectors.Since we are dealing with text ,we need to implement natural language processing techniques.We  used the NLTK library and import all the required classes.Many text preprocessing operations that we have performed on our dataset are as follow:
Removed the word ”subject”, tabs , new line and links
Removed special character/punctuations and digits
Converted words into lower case
Remove stopwords-frequent words such as ”the”, ”is”, etc. that do not have specific semantic
Lemmatization of words-the process in which all similar words are reduced  to their root form by removing their suffix as it appears in the dictionary. We do this to achieve uniformity as there will be many words which will have the same root in our data. words like studies, studying gets converted to study
Tokenization-converts sentences to words.it is the process of splitting text into smaller chunks, called tokens. Each token is an input to the machine learning algorithm as a feature.


Feature extraction:
In text processing, words of the text represent discrete, categorical features.The mapping from textual data to real valued vectors is called feature extraction.it can be done in several different method .we used the TF-IDF technique.TF-IDF stands for Term Frequency Inverse Document Frequency of records. It can be defined as the calculation of how relevant a word in a series or corpus is to a text. The meaning increases proportionally to the number of times in the text a word appears but is compensated by the word frequency in the corpus (data-set).Tf-idf is one of the best metrics to determine how significant a term is to a text in a series or a corpus. tf-idf is a weighting system that assigns a weight to each word in a document based on its term frequency (tf) and the reciprocal document frequency (tf) (idf). The words with higher scores of weight are deemed to be more significant.
Majority of the text pre-processing models like Bag of words, TF-IDF pre-process each and every word at a token level by converting them to vectors and then feeding it to the model. That is why we convert large texts into tokens.

Machine learning methods: 
In the field of artificial intelligence, NLP is one of the most complex areas of research due to the fact that text data is contextual. It needs modification to make it machine-interpretable and requires multiple stages of processing for feature extraction.
There are many types of NLP problems, and one of the most common types is the classification of emails into spam or not spam.It is a binary classification problem which can be implemented either by supervised learning model or neural network . 
There are various ML methods that can be used in our dataset.word embedding is one of the NLP techniques  used to  convert formatted text data into numerical values/vectors which is machine interpretable.
Dimensionality reduction can also be used to remove the least important information (sometimes redundant columns) from a data set.we removed the column “unnamed” from our dataset which is the numbering of the emails as it has no contribution to the outcome of the result.Thus the shape of our dataset becomes  (2605,2)

Visualisation:
Word Cloud is a data visualisation technique used for representing text data in which the size of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word cloud. If a specific word appears multiple times, word cloud follows the trend of that word (or of those words) and gets insights on the trend and patterns. Let's take a spam email for example, if most spam emails talk about car extended warranty. Word cloud will keep track of frequent words found in that email. Anytime it encounters those words, it creates a pattern and makes the word appear larger which will help signify the category the email falls in. The image below is how word cloud classifies the frequency of certain words

This is the visualisation for words in spam email.
Splitting Training and Testing Data: Splitting the data into training and test datasets, where training data contains 70 percent and test data contains 30 percent.
Experimenting with Models:
We will use the following algorithms one by one: Naïve Bayes, Support Vector Machine, Decision Trees, Random Forest, KNN, and AdaBoost Classifier,XGB,Logistic regression. We trained all the different models to give accuracy and F1 scores.
Naive Bayes classification is a simple probability algorithm based on the fact that all features of the model are independent. In the context of the spam filter, we suppose that every word in the message is independent of all other words and we count them with the ignorance of the context.Our classification algorithm produces probabilities of the message to be spam or not spam by the condition of the current set of words. Calculation of the probability is based on the Bayes formula and the components of the formula are calculated based on the frequencies of the words in the whole set of messages.
SVM algorithm is based on the hyperplane that separates the two classes, the greater the margin, the better the classification (also called margin maximization).SVC also helps us classify the category of the type of emails (whether spam or ham) based on the data we trained.
Decision Trees are a non-parametric supervised learning method used for classification.Data is classified stepwise on each node using some decision rules inferred from the data features.Using a decision tree classifier would have been better if we had more categories to classify. We used it regardless to classify the group specific email belongs to. If we had more categories, the decision tree may have performed the best.Random forests are an ensemble Supervised Learning algorithm built on Decision trees.
K Nearest Neighbor is a Supervised Machine Learning algorithm that may be used for both classification and regression predictive problems. KNN is a lazy learner. It relies on distance for classification, so normalizing the training data can improve its accuracy dramatically
Ada-boost or Adaptive Boosting is also an ensemble boosting classifier. It is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
And of course, logistic regression helped us classify by grouping into two. Since the emails are either spam or ham, logistic regression was a useful classifier to use in this scenario since its a binary classifier (either yes or no)

Performance Metrics:
The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.Using F1-score as a performance metric for spam detection problems is a good choice.

The model with an F1 score of 0.945148 is SVC.It can be a good-to-go model.However, these results are based on the training data we used. When applying a model like this to real-world data, we still need to actively monitor the model’s performance over time. We can also continue to improve the model by responding to results and feedback by doing things like adding features and removing misspelt words.

Conclusion: 
Our goal for the project was to implement an end to end ML model to solve spam email detection. Spam emails are one of scammer’s greatest tools. They have been used to steal valuable information from individuals and organisations over the years. We utilised different models and we were able to discover the best model with the highest performance rate. Using natural language processing techniques, we were able to clean the texts in our dataset before converting them to vectors. We applied different models, while our performance metrics were focused on the f1 score. But due to time constraints, we are unable to measure the models performance on real world data over a long period of time. Which is why the result was solely based on our training data.


# Restaurant-Reviews-Classification-_NLP
This project aims to employ Natural Language Processing (NLP) and Machine Learning to develop a classification model that predicts whether a restaurant review is positive (1) or negative (0).

## Importing Libraries
The initial step involves importing essential libraries for data manipulation, analysis, and visualization. This includes NumPy, Pandas, Matplotlib, and Seaborn.

## Data Exploration
The dataset is loaded into a Pandas DataFrame, and an initial exploration is conducted to understand its structure and characteristics. Descriptive statistics, data types, and missing values are assessed. Additionally, the distribution of positive and negative reviews is visualized using Seaborn.

## Data Preprocessing

Checkpoint

A checkpoint is established by creating a copy of the original dataset. Further preprocessing involves adding a new column representing the length of each review, exploring the distribution of review lengths, and identifying reviews with extreme lengths.

## Cleaning the Text'

Text data is cleaned using the NLTK library, involving the removal of non-alphabetic characters, conversion to lowercase, tokenization, stemming, and removing stopwords.

Creating a Bag of Words Model
A Bag of Words model is constructed using the CountVectorizer from scikit-learn. This model converts the text data into a matrix of token counts, capturing the frequency of words in each review.

Splitting the Dataset
The dataset is divided into training and testing sets using the train_test_split function from scikit-learn.

## Building the Model

Naive_bayes Classifier
A Naive Bayes classifier is trained using the GaussianNB algorithm. 
The model is evaluated using an accuracy score and a confusion matrix, providing insights into its predictive performance.

XGBOOST Classifier
An XGBoost classifier is employed for model building. Similar to the Naive Bayes model, the XGBoost model is evaluated using an accuracy score and a confusion matrix.

Final Model - XGBOOST CLASSIFIER
After comparing the performance of the Naive Bayes and XGBoost models, the XGBoost classifier is selected as the final model. The model is trained on the training set and evaluated on the testing set. The accuracy score and confusion matrix are reported, indicating the model's effectiveness in predicting positive and negative sentiments.

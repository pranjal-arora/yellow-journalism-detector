PROJECT NAME: Fake Journalism Detector

MADE BY: Pranjal Arora

TECH-STACK USED: 
Programming Language- Python
Libraries and Modules- Pandas, Numpy, Sklearn, Seaborn
Platform- Jupyter Notebook (ipynb)

REQUIREMENTS THAT NEEDS TO BE INSTALLED:
Numpy, Pandas, Sklearn, Seaborn via pip install
Jupyter Lab

ABOUT PROJECT:
Fake news existed way before the introduction of social media, but it multifolded when social media came into being. With the current usage of social media platforms, consumers are creating and sharing more information than ever before, some of which are misleading with no relevance to reality. For this, we propose to use a machine learning approach for automated classification of news articles and returns its accuracy score.

SEQUENTIAL APPROACH:
This project aims to develop a method for detecting and classifying fake news stories using TF-IDF approach. TF-IDF stands for Term Frequency - Inverse Document Frequency and is a text vectorization algorithm that converts text into vectors and uses the frequency of words to determine how relevant those words are to a given document. 
Sklearn (Scikit-learn) library was used, that offers various features for data processing that can be used for classification, regression, and model selection.
We split the data into training and testing dataset using train_test_split function. It is a function in Sklearn model selection for randomly splitting data arrays into two subsets: for training data and for testing data.

Then, using Sklearn library in Python, we build a TfidfVectorizer on our dataset(news.csv file).

After that,a PassiveAggressive Classifier was initialized. Passive aggressive algorithm is an online-learning algorithm in which the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once. Passive-Aggressive classifier basically decides to keep the model as it is if the prediction is true, else it makes changes to the model if the prediction is incorrect.
Then the model waa fitted in for training and after that predictions are made on testing data using predict() method.

In the end, the accuracy score and the confusion matrix tells us how well our model performs. Accuracy represents the number of correctly classified data instances over the total number of data instances. And a confusion matrix is a technique for summarizing the performance of a classification algorithm. It has entries as True Positive(TP), True Negative(TN), False Positive(FP), and False Negative(FN). The accuracy should be over 88%. In our case, we recieved an accuracy of approximately 92%. 

We also used Seaborn library to display our final conclusions. Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.





 

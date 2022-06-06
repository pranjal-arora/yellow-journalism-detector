# this file is just a copy of the main.ipynb file in the folder. 
# please run only the main.ipynb file only, cell-wise
# the code must be run only in jupyter notebook environment

#cell 1
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#cell 2
df=pd.read_csv('news.csv')

df.shape
df.head()

#cell 3
print(df.info())

#cell 4
labels=df.label
labels.head()

#cell 5
sns.displot(labels)

#cell 6
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#cell 7
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#cell 8
pac=PassiveAggressiveClassifier(max_iter=50) #model goes through passiveaggressive classfier and is allowed to make maximum 50 iterations over the training dataset
pac.fit(tfidf_train,y_train) #fitting the model

y_pred=pac.predict(tfidf_test) #using predict method on the model after fitting the model.
score=accuracy_score(y_test,y_pred)

#cell 9
print(f'Accuracy: {round(score*100,2)}%')
print('\nSummary of perofrmance by Confusion matrix with no. of TP FN FP TN : ')

cm=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']) 
cm

#cell 10
sns.heatmap(cm, annot=True)

#cell 11
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
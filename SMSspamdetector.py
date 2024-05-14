import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("C:/Users/DELL/Downloads/archive/spam.csv", encoding= 'latin-1')
print(data.head())
print(data.isna().sum())
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1,inplace=True)
data.columns = ['Category','Message']
print(data.head(10))
print(data.info())
data['Category'].value_counts()
data['Category'].value_counts().plot(kind='bar')
data['Spam']=data['Category'].apply(lambda x:1 if x=='spam'else 0)
print(data.columns)
x = np.array(data["Message"])
y = np.array(data["Spam"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) 
clf = MultinomialNB()
clf.fit(X_train,y_train)
sample = input("Enter a message:")
data = cv.transform([sample]).toarray()
print(clf.predict(data))

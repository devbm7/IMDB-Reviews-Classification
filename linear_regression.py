import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report  
import pickle

df = pd.read_csv('IMDB Dataset.csv')  

print(df.head())  

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})  
print(df.isnull())

X = df['review']  
y = df['sentiment']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

tfidf_vectorizer = TfidfVectorizer()  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression()  
model.fit(X_train_tfidf, y_train)  

y_pred = model.predict(X_test_tfidf)  

print("Accuracy:", accuracy_score(y_test, y_pred))  
print(classification_report(y_test, y_pred))

filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as model_file:
    pickle.dump(model, model_file)

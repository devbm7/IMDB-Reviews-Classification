import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pickle

st.title(":blue[IMDB Dataset of 50k reviews]")


# Dataset
st.header("Dataset")
df = pd.read_csv('IMDB Dataset.csv')
with st.expander("Show Data"):
    st.write(df)
df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})
X = df['review']
y = df['sentiment']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=41)

tfidf_vectorizer = TfidfVectorizer()  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
X_test_tfidf = tfidf_vectorizer.transform(X_test)  

# Linear Regression
st.header('Linear Regression',divider='orange')
model = LogisticRegression()  
model.fit(X_train_tfidf, y_train)  

y_pred = model.predict(X_test_tfidf)  

print("Accuracy:", accuracy_score(y_test, y_pred))  
print(classification_report(y_test, y_pred))

filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as model_file:
    pickle.dump(model, model_file)

st.write("Accuracy:", accuracy_score(y_test, y_pred))  
st.markdown(body=classification_report(y_test, y_pred),unsafe_allow_html=True)  

# Naive Bayes
st.header("Naive Bayes",divider='orange')
model_nb = MultinomialNB()  
model_nb.fit(X_train_tfidf, y_train)  

# Evaluate the model  
y_pred = model_nb.predict(X_test_tfidf)  
st.write("Accuracy:", accuracy_score(y_test, y_pred))  
st.markdown(body=classification_report(y_test, y_pred),unsafe_allow_html=True)  

# SVM
st.header("Support Vector Machine")
st.caption("Kernal type is linear.")
model = SVC(kernel='linear')  # You can also try 'rbf', 'poly', etc.  
model.fit(X_train_tfidf, y_train)  

y_pred = model.predict(X_test_tfidf)  
st.write("Accuracy:", accuracy_score(y_test, y_pred))  
st.markdown(body=classification_report(y_test, y_pred),unsafe_allow_html=True)  

import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.metrics import accuracy_score, classification_report  
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch  

st.set_page_config(layout='wide')
@st.cache_data  
def load_data():  
    return pd.read_csv('IMDB Dataset.csv')  

if 'models' not in st.session_state:  
    st.session_state.models = {}  
if 'reports' not in st.session_state:  
    st.session_state.reports = {}  
if 'accuracy' not in st.session_state:  
    st.session_state.accuracy = {}  
code = '''
import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.metrics import accuracy_score, classification_report  
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch  

@st.cache_data  
def load_data():  
    return pd.read_csv('IMDB Dataset.csv')  

if 'models' not in st.session_state:  
    st.session_state.models = {}  
if 'reports' not in st.session_state:  
    st.session_state.reports = {}  
if 'accuracy' not in st.session_state:  
    st.session_state.accuracy = {}  

st.title(":blue[IMDB 50k Reviews Dataset]")

df = load_data()  
with st.expander("View Data"):
    st.write(df)
df = df[:100]
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})  

X = df['review']  
y = df['sentiment']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
vectorizer = TfidfVectorizer() 
X_train_tfidf = vectorizer.fit_transform(X_train)

if not st.session_state.models:  
    # models
    models = {  
        "SVM": SVC(kernel='linear'),  
        "Logistic Regression": LogisticRegression(max_iter=1000),  
        "Random Forest": RandomForestClassifier(n_estimators=10),  
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": MultinomialNB()  
    }  

    for name, model in models.items():  
        model.fit(X_train_tfidf, y_train)  
        st.session_state.models[name] = model  
        X_test_tfidf = vectorizer.transform(X_test)  
        y_pred = model.predict(X_test_tfidf)  
        st.session_state.accuracy[name] = accuracy_score(y_test, y_pred)  
        report = classification_report(y_test, y_pred, output_dict=True)  
        st.session_state.reports[name] = pd.DataFrame(report).transpose()  

if st.session_state.accuracy:  
    
    plt.figure(figsize=(10, 5))  
    plt.barh(st.session_state.accuracy.keys(), st.session_state.accuracy.values(), color=['blue', 'orange', 'green','red', 'purple'])  
    plt.ylabel('Accuracy')  
    plt.title('Model Accuracy Comparison')  
    st.pyplot(plt)  

    st.link_button("Reference for the methods", "https://scikit-learn.org/stable/supervised_learning.html")
    for name, report_df in st.session_state.reports.items(): 
        if name == 'SVM':
            st.header("Support Vector Machine",divider='orange') 
            st.link_button("Reference",'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC')
        elif name == "Random Forest":
            st.header("Random Forest Classifier",divider='orange')
            st.link_button("Reference","https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html")
        elif name == "Logistic Regression":
            st.header("Logistic Regression",divider='orange')
            st.link_button("Reference","https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html")
        elif name == "Gradient Boosting":
            st.header("Gradient Boosting Classifier",divider='orange')
            st.link_button("Reference", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html")
        else:
            st.header(f"{name}",divider='orange')  
        st.dataframe(report_df)  

st.caption("For Convience only first 100/1000 rows were considered for modeling.")
st.caption("In addition to these methods, manual tryout also uses DistilBERT for classification.")
st.header("Manual Tryouts",divider='green')
user_input = st.text_area("Review", "")  

if st.button("Predict"):  
    if user_input:
        user_input_tfidf = vectorizer.transform([user_input])  

        predictions = {}  
        for name, model in st.session_state.models.items():  
            prediction = model.predict(user_input_tfidf)  
            predictions[name] = "Positive" if prediction[0] == 1 else "Negative"  


        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model_dbrt = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        inputs = tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            logits = model_dbrt(**inputs).logits

        DistilBERT_prediction = logits.argmax().item()
        predictions["DistBERT"] = "Positive" if DistilBERT_prediction == 1 else "Negative"

        st.write("Predicted Sentiment:") 
        i = 0
        col1, col2, col3 = st.columns(3) 
        for name in predictions:  
            if i%3==0:
                col1.metric(label=name, value=predictions[name])
            elif i%3==1:
                col2.metric(label=name, value=predictions[name])
            else:
                col3.metric(label=name, value=predictions[name])
            i+=1
    else:  
        st.write("Please enter a review.")  
'''
with st.expander("View Code",expanded=True):
    st.code(code,language='python')

st.title(":blue[IMDB 50k Reviews Dataset]")

df = load_data()  
with st.expander("View Data"):
    st.write(df)
df = df[:100]
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})  

X = df['review']  
y = df['sentiment']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
vectorizer = TfidfVectorizer() 
X_train_tfidf = vectorizer.fit_transform(X_train)

if not st.session_state.models:  
    # models
    models = {  
        "SVM": SVC(kernel='linear'),  
        "Logistic Regression": LogisticRegression(max_iter=1000),  
        "Random Forest": RandomForestClassifier(n_estimators=10),  
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": MultinomialNB()  
    }  

    for name, model in models.items():  
        model.fit(X_train_tfidf, y_train)  
        st.session_state.models[name] = model  
        X_test_tfidf = vectorizer.transform(X_test)  
        y_pred = model.predict(X_test_tfidf)  
        st.session_state.accuracy[name] = accuracy_score(y_test, y_pred)  
        report = classification_report(y_test, y_pred, output_dict=True)  
        st.session_state.reports[name] = pd.DataFrame(report).transpose()  

if st.session_state.accuracy:  
    
    plt.figure(figsize=(10, 5))  
    plt.barh(st.session_state.accuracy.keys(), st.session_state.accuracy.values(), color=['blue', 'orange', 'green','red', 'purple'])  
    plt.ylabel('Accuracy')  
    plt.title('Model Accuracy Comparison')  
    st.pyplot(plt)  

    st.link_button("Reference for the methods", "https://scikit-learn.org/stable/supervised_learning.html")
    for name, report_df in st.session_state.reports.items(): 
        if name == 'SVM':
            st.header("Support Vector Machine",divider='orange') 
            st.link_button("Reference",'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC')
        elif name == "Random Forest":
            st.header("Random Forest Classifier",divider='orange')
            st.link_button("Reference","https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html")
        elif name == "Logistic Regression":
            st.header("Logistic Regression",divider='orange')
            st.link_button("Reference","https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html")
        elif name == "Gradient Boosting":
            st.header("Gradient Boosting Classifier",divider='orange')
            st.link_button("Reference", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html")
        else:
            st.header(f"{name}",divider='orange')  
        st.dataframe(report_df)  

st.caption("For Convience only first 100/1000 rows were considered for modeling.")
st.caption("In addition to these methods, manual tryout also uses DistilBERT for classification.")
st.header("Manual Tryouts",divider='green')
user_input = st.text_area("Review", "")  

if st.button("Predict"):  
    if user_input:
        user_input_tfidf = vectorizer.transform([user_input])  

        predictions = {}  
        for name, model in st.session_state.models.items():  
            prediction = model.predict(user_input_tfidf)  
            predictions[name] = "Positive" if prediction[0] == 1 else "Negative"  


        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model_dbrt = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        inputs = tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            logits = model_dbrt(**inputs).logits

        DistilBERT_prediction = logits.argmax().item()
        predictions["DistBERT"] = "Positive" if DistilBERT_prediction == 1 else "Negative"

        st.write("Predicted Sentiment:") 
        i = 0
        col1, col2, col3 = st.columns(3) 
        for name in predictions:  
            if i%3==0:
                col1.metric(label=name, value=predictions[name])
            elif i%3==1:
                col2.metric(label=name, value=predictions[name])
            else:
                col3.metric(label=name, value=predictions[name])
            i+=1
    else:  
        st.write("Please enter a review.")
    
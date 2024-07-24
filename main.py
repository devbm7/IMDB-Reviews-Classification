import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt

st.title(":blue[IMDB Dataset of 50k reviews]")


@st.cache_data
def load_data():
    return pd.read_csv('IMDB Dataset.csv')
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = {}
if 'report' not in st.session_state:
    st.session_state.report = {}

# Dataset
st.header("Dataset")
df = load_data()
with st.expander("Show Data"):
    st.write(df)
df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})
X = df['review']
y = df['sentiment']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=41)

tfidf_vectorizer = TfidfVectorizer()  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  
X_test_tfidf = tfidf_vectorizer.transform(X_test)  


if not st.session_state.models:  
    # Vectorization using TF-IDF  
    st.session_state.vectorizer = TfidfVectorizer()  
    X_train_tfidf = st.session_state.vectorizer.fit_transform(X_train)  

    # Initialize and train models  
    models = {  
        "SVM": SVC(kernel='linear'),  
        "Logistic Regression": LogisticRegression(max_iter=1000),  
        "Random Forest": RandomForestClassifier(n_estimators=100)  
    }  

    for name, model in models.items():  
        model.fit(X_train_tfidf, y_train)  
        st.session_state.models[name] = model  
        X_test_tfidf = st.session_state.vectorizer.transform(X_test)  
        y_pred = model.predict(X_test_tfidf)  
        st.session_state.accuracy[name] = accuracy_score(y_test, y_pred)  
        st.session_state.report[name] = classification_report(y_test, y_pred)  

# Streamlit app UI  
st.title("Sentiment Analysis App with Multiple Models")  
st.write("Enter a review to predict its sentiment using SVM, Logistic Regression, and Random Forest:")  

# Display accuracy and classification reports  
if st.session_state.accuracy:  
    st.write("Model trained successfully!")  
    
    # Visualizing accuracy  
    plt.figure(figsize=(10, 5))  
    plt.bar(st.session_state.accuracy.keys(), st.session_state.accuracy.values(), color=['blue', 'orange', 'green'])  
    plt.ylabel('Accuracy')  
    plt.title('Model Accuracy Comparison')  
    st.pyplot(plt)  

    # Display classification reports  
    for name in st.session_state.report:  
        st.write(f"### Classification Report for {name}:")  
        st.text(st.session_state.report[name])  

# Input text from the user  
user_input = st.text_area("Review", "")  

if st.button("Predict"):  
    if user_input:  
        # Vectorize user input for all models  
        user_input_tfidf = st.session_state.vectorizer.transform([user_input])  

        # Predict using all models  
        predictions = {}  
        for name, model in st.session_state.models.items():  
            prediction = model.predict(user_input_tfidf)  
            predictions[name] = "Positive" if prediction[0] == 1 else "Negative"  
        
        # Display predictions for each model  
        st.write("Predicted Sentiment:")  
        for name in predictions:  
            st.write(f"{name}: **{predictions[name]}**")  
    else:  
        st.write("Please enter a review.")
# # Linear Regression
# st.header('Linear Regression',divider='orange')
# model = LogisticRegression()  
# model.fit(X_train_tfidf, y_train)  

# y_pred = model.predict(X_test_tfidf)  

# print("Accuracy:", accuracy_score(y_test, y_pred))  
# print(classification_report(y_test, y_pred))

# filename = 'linear_regression_model.pkl'
# with open(filename, 'wb') as model_file:
#     pickle.dump(model, model_file)

# st.write("Accuracy:", accuracy_score(y_test, y_pred))  
# st.markdown(body=classification_report(y_test, y_pred),unsafe_allow_html=True)  

# # Naive Bayes
# st.header("Naive Bayes",divider='orange')
# model_nb = MultinomialNB()  
# model_nb.fit(X_train_tfidf, y_train)  

# # Evaluate the model  
# y_pred = model_nb.predict(X_test_tfidf)  
# st.write("Accuracy:", accuracy_score(y_test, y_pred))  
# st.markdown(body=classification_report(y_test, y_pred),unsafe_allow_html=True)  

# # SVM
# st.header("Support Vector Machine")
# st.caption("Kernal type is linear.")
# model = SVC(kernel='linear')  # You can also try 'rbf', 'poly', etc.  
# model.fit(X_train_tfidf, y_train)  

# y_pred = model.predict(X_test_tfidf)  
# st.write("Accuracy:", accuracy_score(y_test, y_pred))  
# st.markdown(body=classification_report(y_test, y_pred),unsafe_allow_html=True)  

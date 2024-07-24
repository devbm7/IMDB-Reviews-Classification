import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.metrics import accuracy_score, classification_report  
from transformers import BertTokenizer, BertForSequenceClassification  
import torch  

@st.cache_data  
def load_data():  
    return pd.read_csv('reviews.csv')  

if 'models' not in st.session_state:  
    st.session_state.models = {}  
if 'reports' not in st.session_state:  
    st.session_state.reports = {}  
if 'accuracy' not in st.session_state:  
    st.session_state.accuracy = {}  

df = load_data()  

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})  

X = df['review']  
y = df['sentiment']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

if not st.session_state.models:  
    vectorizer = TfidfVectorizer()  
    X_train_tfidf = vectorizer.fit_transform(X_train)  

    # models
    models = {  
        "SVM": SVC(kernel='linear'),  
        "Logistic Regression": LogisticRegression(max_iter=1000),  
        # "Random Forest": RandomForestClassifier(n_estimators=100),  
        "Gradient Boosting": GradientBoostingClassifier()  
    }  

    for name, model in models.items():  
        model.fit(X_train_tfidf, y_train)  
        st.session_state.models[name] = model  
        X_test_tfidf = vectorizer.transform(X_test)  
        y_pred = model.predict(X_test_tfidf)  
        st.session_state.accuracy[name] = accuracy_score(y_test, y_pred)  
        report = classification_report(y_test, y_pred, output_dict=True)  
        st.session_state.reports[name] = pd.DataFrame(report).transpose()  

    st.session_state.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
    st.session_state.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  

    train_encodings = st.session_state.bert_tokenizer(list(X_train), truncation=True, padding=True, return_tensors='pt')  
    train_labels = torch.tensor(y_train.values)  

    train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)  

    training_args = torch.optim.AdamW(st.session_state.bert_model.parameters(), lr=1e-5)  
    st.session_state.bert_model.train()  

    for epoch in range(1):  
        for batch in train_dataset:  
            inputs = batch[0], batch[1]  
            labels = batch[2]  
            outputs = st.session_state.bert_model(*inputs, labels=labels)  
            loss = outputs.loss  
            loss.backward()  
            training_args.step()  
            training_args.zero_grad()  

    st.session_state.bert_model.eval()  
    test_encodings = st.session_state.bert_tokenizer(list(X_test), truncation=True, padding=True, return_tensors='pt')  
    with torch.no_grad():  
        outputs = st.session_state.bert_model(test_encodings['input_ids'], test_encodings['attention_mask'])  
    predictions = torch.argmax(outputs.logits, dim=1).numpy()  
    st.session_state.accuracy["BERT"] = accuracy_score(y_test, predictions)  
    report = classification_report(y_test, predictions, output_dict=True)  
    st.session_state.reports["BERT"] = pd.DataFrame(report).transpose()  

if st.session_state.accuracy:  
    
    plt.figure(figsize=(10, 5))  
    plt.bar(st.session_state.accuracy.keys(), st.session_state.accuracy.values(), color=['blue', 'orange', 'green', 'purple'])  
    plt.ylabel('Accuracy')  
    plt.title('Model Accuracy Comparison')  
    st.pyplot(plt)  

    for name, report_df in st.session_state.reports.items():  
        st.header(f"{name}",divider='orange')  
        st.dataframe(report_df)  

st.header("Manual Tryouts")
user_input = st.text_area("Review", "")  

if st.button("Predict"):  
    if user_input:  
        user_input_tfidf = vectorizer.transform([user_input])  

        predictions = {}  
        for name, model in st.session_state.models.items():  
            prediction = model.predict(user_input_tfidf)  
            predictions[name] = "Positive" if prediction[0] == 1 else "Negative"  

        inputs = st.session_state.bert_tokenizer(user_input, return_tensors='pt', truncation=True, padding=True)  
        with torch.no_grad():  
            output = st.session_state.bert_model(inputs['input_ids'], inputs['attention_mask'])  
        bert_prediction = torch.argmax(output.logits, dim=1).item()  
        predictions["BERT"] = "Positive" if bert_prediction == 1 else "Negative"  

        st.write("Predicted Sentiment:")  
        for name in predictions:  
            st.write(f"{name}: **{predictions[name]}**")  
    else:  
        st.write("Please enter a review.")
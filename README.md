# 🧠 Student Depression Prediction using Machine Learning

This project predicts the mental health condition (Depressed / Not Depressed) of students using unsupervised learning (K-Means Clustering). The dataset includes various features such as sleep duration, social activities, academic performance, and lifestyle habits.

---

## 💡 Project Overview

- 📊 **Machine Learning Algorithm**: K-Means Clustering
- 🌐 **Frontend**: Streamlit
- 📁 **Dataset**: Cleaned dataset with behavioral, academic, and social parameters
- 🎯 **Goal**: Classify students into two clusters: `Depressed` and `Not Depressed`

---

## 🛠 Features

- Upload or manually enter student data
- Preprocessing with label encoding & scaling
- Clustering using KMeans with `k=2`
- Interactive UI using Streamlit
- Cluster results shown as **"Depressed"** or **"Not Depressed"**
- Graph visualizations like Boxplot, Cluster Distribution

---

## 📦 Requirements

Install required libraries using:

``bash
pip install -r requirements.txt

---

## 🔗 Live Demo
- [Frontend Streamlit App](https://your-streamlit-url.streamlit.app/)
- [Backend Flask API (Render)](https://your-api-name.onrender.com)

## 📌 How It Works
1. User enters student data into the Streamlit form
2. The data is sent via POST to the Flask API
3. Flask loads the model and returns if the student is depressed or not
4. Streamlit shows the result


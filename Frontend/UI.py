import streamlit as st
import pandas as pd
import requests
import os
import altair as alt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

st.set_page_config(page_title="Student Depression Prediction", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f7f9fa; }
    .title { font-size: 36px; color: #2c3e50; font-weight: 700; text-align: center; }
    .subtitle { font-size: 22px; color: #34495e; text-align: center; margin-bottom: 30px; }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 12px;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🧠 Student Depression Predictor</h1>', unsafe_allow_html=True)
st.markdown('<h5 class="subtitle">Enter student details to predict their cluster and view the visualization.</h5>', unsafe_allow_html=True)

@st.cache_data
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "cleaned_student_depression.csv")
    if not os.path.exists(path):
        st.error(f"❌ File not found at: {path}")
    return pd.read_csv(path)

df_train = load_data()

left_col, right_col = st.columns([1, 2])

with left_col:
    with st.form("depression_form"):
        age = st.slider("🧃 Age", 15, 40, 22)
        academic_pressure = st.slider("📚 Academic Pressure (0-5)", 0, 5, 2)
        work_pressure = st.slider("💼 Work Pressure (0-5)", 0, 5, 1)
        cgpa = st.number_input("🎓 CGPA (0.0 - 10.0)", 0.0, 10.0, 7.5)

        sleep_duration = st.slider("🛌 Sleep Duration (Hours)", 0.0, 12.0, 6.0)
        dietary = st.selectbox("🥗 Dietary Habits", ["Unhealthy", "Healthy"])
        suicidal_thoughts = st.selectbox("💭 Suicidal Thoughts", ["No", "Yes"])
        study_hours = st.slider("📖 Work/Study Hours", 0.0, 16.0, 4.0)
        financial_stress = st.slider("💰 Financial Stress (0-5)", 0, 5, 2)

        submitted = st.form_submit_button("Predict")

with right_col:
    # Add separate checkboxes for each graph
    show_kmeans = st.checkbox("📉 Show K-Means Cluster Visualization", value=True)
    show_elbow = st.checkbox("📈 Show Elbow Curve for Optimal Clusters", value=True)
    show_boxplots = st.checkbox("📦 Show Feature Distributions per Cluster (Box Plots)", value=True)

    if submitted:
        dietary_map = {"Unhealthy": 0, "Healthy": 1}
        suicidal_map = {"No": 0, "Yes": 1}

        input_data = {
            "Age": age,
            "Academic Pressure": academic_pressure,
            "Work Pressure": work_pressure,
            "CGPA": cgpa,
            "Sleep Duration": sleep_duration,
            "Dietary Habits": dietary_map[dietary],
            "Have you ever had suicidal thoughts ?": suicidal_map[suicidal_thoughts],
            "Work/Study Hours": study_hours,
            "Financial Stress": financial_stress
        }

        try:
            res = requests.post("https://student-depression-predictor-6hk7.onrender.com/predict", ...)
            result = res.json().get("prediction", "Error")
            prediction_result = f"🗞️ Prediction Result: **{result}**"
        except Exception as e:
            st.error(f"❌ Error during prediction or visualization: {e}")
            prediction_result = None

        if prediction_result:
            st.success(prediction_result)

        features = ['Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
                    'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration',
                    'Dietary Habits', 'Have you ever had suicidal thoughts ?',
                    'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 'Degree']

        df_clean = df_train[features].copy().fillna(0)

        if show_kmeans:
            try:
                st.subheader("📊 K-Means Clusters with Your Input")
                features = ['Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
                            'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration',
                            'Dietary Habits', 'Have you ever had suicidal thoughts ?',
                            'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 'Degree']

                df_pca = df_train[features].copy().fillna(0)

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(df_pca)
                kmeans = KMeans(n_clusters=2, random_state=42)
                clusters = kmeans.fit_predict(df_pca)
                df_train_copy = df_train.copy()
                df_train_copy['Cluster'] = clusters
                pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
                pca_df["Cluster"] = df_train_copy["Cluster"].astype(str)

                import numpy as np
                user_point = np.array([[
                    0,  # Gender (assuming 0 as default, since not input)
                    age,
                    academic_pressure,
                    work_pressure,
                    cgpa,
                    0,  # Study Satisfaction (default 0)
                    0,  # Job Satisfaction (default 0)
                    sleep_duration,
                    dietary_map[dietary],
                    suicidal_map[suicidal_thoughts],
                    study_hours,
                    financial_stress,
                    0,  # Family History of Mental Illness (default 0)
                    0   # Degree (default 0)
                ]])

                user_pca = pca.transform(user_point)[0]

                pca_df = pd.concat([pca_df, pd.DataFrame([{
                    "PC1": user_pca[0],
                    "PC2": user_pca[1],
                    "Cluster": "You"
                }])], ignore_index=True)

                chart = alt.Chart(pca_df).mark_circle(size=80).encode(
                    x='PC1',
                    y='PC2',
                    color=alt.Color('Cluster:N', scale=alt.Scale(scheme='category10')),
                    tooltip=['Cluster']
                ).properties(title="K-Means Clusters with Your Input", width=700, height=400)

                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying K-Means cluster visualization: {e}")

        if show_elbow:
            # Elbow Curve
            st.subheader("📈 Elbow Curve for Optimal Clusters")
            distortions = []
            K = range(1, 10)
            for k in K:
                kmeanModel = KMeans(n_clusters=k, random_state=42)
                kmeanModel.fit(df_clean)
                distortions.append(kmeanModel.inertia_)

            fig, ax = plt.subplots()
            ax.plot(K, distortions, 'bx-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('The Elbow Method showing optimal k')
            st.pyplot(fig)

        if show_boxplots:
            # Box Plots
            st.subheader("📦 Feature Distributions per Cluster (Box Plots)")
            selected_features = ['Academic Pressure', 'Sleep Duration', 'Financial Stress']

            for feature in selected_features:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x="Cluster", y=feature, data=df_train, palette="Set2", ax=ax)
                ax.set_title(f"{feature} by Cluster")
                st.pyplot(fig)

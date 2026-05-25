from dotenv import load_dotenv
import os
load_dotenv()
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import re
import nltk
import random
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
from groq import Groq

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("vader_lexicon")

FILE_NAME = "user_data.csv"

# Logged_in , username , page
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "Login"


def signup_page():   # usage
    if not os.path.exists(FILE_NAME):
        df = pd.DataFrame(columns=[
            "First_name", "Last_name", "Email",
            "Phone", "Username", "Password"
        ])
        df.to_csv(FILE_NAME, index=False)

    with st.form("signup"):
        df = pd.read_csv(FILE_NAME)

        first_name = st.text_input("Enter First Name:")
        last_name = st.text_input("Enter Last Name")
        email = st.text_input("Enter Email")
        phone = st.text_input("Enter Phone No")
        username = st.text_input("Enter Username")
        password = st.text_input("Enter Password", type="password")
        re_password = st.text_input("Enter Confirm Password", type="password")

        submit = st.form_submit_button("Submit")

        if st.form_submit_button("Already Registered? Login From Here"):
            st.session_state.page = "Login"
            st.rerun()

        if submit:
            if not first_name or not last_name or not email or not phone or not username or not password:
                st.error("Please Fill All Details")

            elif not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
                st.error("Please Enter Valid Email")

            elif not re.match(r"[0-9]+$", phone):
                st.error("Phone must be 10 digits")

            elif password != re_password:
                st.error("Please Enter Correct Confirm Password")

            elif not re.match(r"[A-Z]", password):
                st.error("Password must contain one uppercase")

            elif not re.search(r"[a-z]", password):
                st.error("Password must contain one Lowercase")

            elif username in df["Username"].values:
                st.error("Username Already Exists")

            elif email in df["Email"].values:
                st.error("Email Already Exists")

            else:
                new_data = pd.DataFrame([{
                    "First_name": first_name,
                    "Last_name": last_name,
                    "Email": email,
                    "Phone": phone,
                    "Username": username,
                    "Password": password
                }])

                df2 = pd.concat([df, new_data], ignore_index=True)
                df2.to_csv(FILE_NAME, index=False)

                st.success("User Registered.........")
                st.session_state.page = "Login"
                st.rerun()

def login_page():
    st.subheader("Login Page")

    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password")
    submit = st.button("Submit")

    if submit:
        df = pd.read_csv(FILE_NAME)
        user = df[(df["Username"] == username) & (df["Password"] == password)]
        if not user.empty:
            st.session_state.page = "Home"
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("User Logged in Successfully...")
            time.sleep(1)
            st.rerun()

    if st.button("New User? Register From Here"):
        st.session_state.page = "Signup"
        st.rerun()

def home_page():
    st.subheader("Home Page")

    st.set_page_config(page_title="Social Media App", page_icon="🤖", layout="wide")

    # SIDEBAR
    with st.sidebar:
        selected = option_menu("Main Menu",
                               ["Data Cleaning", "Dataset", "Overview", "Social Media Analytics", "Supervise Models",
                                "UnSupervise Models", "NLP & N-GRAM", "DeepLearning Models", "LLM","Exit"],
                               icons=["bi bi-stars", "table", "bar-chart", "graph-up", "bi bi-diagram-3-fill",
                                      "bi bi-grid-3x3-gap-fill", "bi bi-chat-square-text-fill",
                                      "bi bi-cpu-fill", "bi-flower1","box-arrow-right"], menu_icon="robot", default_index=0)

    if selected == "Data Cleaning":
        st.header("Data Cleaning")
        df = pd.read_csv("social_media.csv")
        st.subheader("Dataset Before Cleaning")
        st.dataframe(df)
        st.divider()

        # Drop Duplicate Value
        df = df.drop_duplicates()

        # Age
        df["age"] = df["age"].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].fillna(0).astype(int)
        mean_age = df["age"].mean()
        df.fillna({"age": mean_age}, inplace=True)

        # gander
        df.fillna({"gender": "male"}, inplace=True)
        df["gender"] = df["gender"].str.strip().str.title()
        df["gender"] = df["gender"].replace({
            "F": "Female",
            "M": "Male"
        })

        # social_media_usage_hr are clean

        # platform
        df["platform"] = df["platform"].str.strip().str.title()
        df["platform"] = df["platform"].replace({
            "Tiktok": "Spotify"
        })

        # sleep_hr, exercise_hr

        # gpa
        gpa_mean = df["gpa"].mean()
        df.fillna({"gpa": gpa_mean}, inplace=True)
        df["gpa"] = round(df["gpa"], 1)

        # stress_score
        stress_score_mean = df["stress_score"].mean()
        df.fillna({"stress_score": stress_score_mean}, inplace=True)
        df["stress_score"] = round(df["stress_score"], 2)

        # depression_label
        df["depression_label"] = df["depression_label"].str.strip().str.title()

        # loneliness_score
        loneliness_score_mean = df["loneliness_score"].mean()
        df.fillna({"loneliness_score": 5}, inplace=True)

        # anxiety_score, screen_time_hr, anxiety_score, depression_binary, sentence, sentiment
        st.subheader("Dataset After Cleaning")
        st.write(df)

        df.to_csv("clean_dataset.csv", index=False)

        # Label Encoding
        df = pd.read_csv("clean_dataset.csv")
        le = LabelEncoder()
        df["gender"] = le.fit_transform(df["gender"])
        df["platform"] = le.fit_transform(df["platform"])
        df["sentiment"] = le.fit_transform(df["sentiment"])

        # Ordinal Encoding
        oe = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
        df["depression_label"] = oe.fit_transform(df[["depression_label"]])

        st.subheader("Label Encoder & Ordinal Encoder")
        st.dataframe(df)
        st.divider()

        # Feature Scaling
        scaler = StandardScaler()
        df[["age", "social_media_usage_hr", "sleep_hr", "exercise_hr", "gpa", "stress_score", "loneliness_score",
            "anxiety_score", "screen_time_hr", "depression_binary"]] = scaler.fit_transform(
            df[["age", "social_media_usage_hr", "sleep_hr", "exercise_hr", "gpa", "stress_score", "loneliness_score",
                "anxiety_score", "screen_time_hr", "depression_binary"]])

        st.subheader("Label Encoder | Ordinal Encoder | StandardScaler")
        st.dataframe(df)
        st.divider()
        df.to_csv("Final.csv", index=False)

    if selected == "Dataset":
        df = pd.read_csv("social_media.csv")
        st.title("Data Exploration")
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.divider()

        # COLUMNS SELECTION
        st.subheader("select Columns")
        selected_columns = st.multiselect("Choose Columns", df.columns, default=df.columns)
        filtered_df = df[selected_columns]
        st.divider()

        # Search Dataset
        st.subheader("Search In Dataset")
        search = st.text_input("Search data From Here")
        if search:
            filtered_df = filtered_df[filtered_df.astype(str).apply(
                lambda row: row.str.contains(search, case=False).any(), axis=1
            )]
            st.dataframe(filtered_df)
        st.divider()

        # Columns Filter -Columns name | Value
        st.subheader("Column Filter")
        col1, col2 = st.columns(2)
        with col1:
            filter_column = st.selectbox("Select Column", filtered_df.columns)
        with col2:
            filter_value = st.selectbox("Select Value", filtered_df[filter_column].dropna().unique())

        filtered_df = filtered_df[filtered_df[filter_column].astype(str) == str(filter_value)]
        st.dataframe(filtered_df)
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data", csv, "filtered_social_media_data.csv", "csv")
        st.divider()

        filtered_df = df[selected_columns]
        st.subheader("Row Slider")
        row_num = st.slider("Enter Row Number", min_value=1, max_value=len(filtered_df), step=1)

        st.subheader("Row Preview")
        st.dataframe(filtered_df.iloc[row_num - 1])

    if selected == "Overview":
        st.title("Dashboard Overview")
        st.divider()
        df = pd.read_csv("clean_dataset.csv")

        col1, col2, col3, col4 = st.columns(4)
        total_students = len(df)
        avg_screen = df["screen_time_hr"].mean()
        avg_sleep = df["sleep_hr"].mean()
        avg_stress = df["stress_score"].mean()

        col1.metric("Total Students", total_students)
        col2.metric("Average Screen Time", f"{avg_screen:.1f}hr")
        col3.metric("Average Sleep", f"{avg_sleep:.1f}hr")
        col4.metric("Average Stress", f"{avg_stress:.1f}hr")

        st.divider()

        # Platform Analysis
        st.subheader("Platform usage Analysis")

        platform_stats = df.groupby("platform").agg(
            Total_Users=("platform", "count"),
            Avg_stress=("stress_score", "mean"),
            Avg_Anxiety=("anxiety_score", "mean"),
            Avg_Screen_time=("screen_time_hr", "mean")
        )

        st.dataframe(
            platform_stats.style.format({
                "Avg_stress": "{:.1f}",
                "Avg_Anxiety": "{:.1f}",
                "Avg_Screen_time": "{:.1f} hr"
            }).background_gradient(cmap="Reds"),
            use_container_width=True
        )
        st.divider()

        # GENDER ANALYSIS
        col_gen, col_dep = st.columns(2)

        with col_gen:
            st.subheader("Gender Distribution")

            gender_df = df["gender"].value_counts().reset_index()
            gender_df.columns = ["Gender", "Count"]

            fig1 = px.pie(
                gender_df,
                names="Gender",
                values="Count",
                hole=0.4
            )

            st.plotly_chart(fig1, use_container_width=True)

        with col_dep:
            st.subheader("Depression Level Distribution")

            dep_df = df["depression_label"].value_counts().reset_index()
            dep_df.columns = ["Level", "Count"]

            fig2 = px.bar(
                dep_df,
                x="Level",
                y="Count",
                color="Level"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # SENTIMENT ANALYSIS
        st.subheader("Sentiment Analysis")

        sentiment_df = df["sentiment"].value_counts().reset_index()
        sentiment_df.columns = ["Sentiment", "Count"]

        fig3 = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            hole=0.5
        )

        st.plotly_chart(fig3, use_container_width=True)
        st.divider()

        # Data Quality
        df = pd.read_csv("social_media.csv")
        with st.expander("Data Quality & Audit Logs"):

            audit1, audit2 = st.columns(2)

            audit1.write(f"Duplicate Records : {df.duplicated().sum()}")
            audit2.write(f"Missing Values : {df.isna().sum().sum()}")

            st.info("Dataset contains student mental health and social media usage patterns.")
            st.success("Dashboard Generated Successfully")

        df = pd.read_csv("clean_dataset.csv")

        # Statistics
        st.header("Columns Statistics")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select Numeric Column", numeric_cols)
            st.write(df[selected_col].describe())
        st.divider()

    if selected == "Social Media Analytics":
        st.title("Mental Health Analytics Dashboard")
        df = pd.read_csv("clean_dataset.csv")
        st.divider()

        # Screen Time vs Stress

        fig4 = px.scatter(
            df,
            x="screen_time_hr",
            y="stress_score",
            color="platform",
            size="anxiety_score",
            hover_data=["gender"]
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.divider()

        # Sleep vs anxiety
        st.subheader("sleep Hours vs Anxiety")

        fig5 = px.scatter(
            df,
            x="sleep_hr",
            y="anxiety_score",
            color="depression_label"
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.divider()

        # GPA Analysis
        st.subheader("GPA Distribution")

        fig6 = px.histogram(
            df,
            x="gpa",
            nbins=20,
            color="gender"
        )
        st.plotly_chart(fig6)
        st.divider()

        # Platform vs stress
        st.subheader("Platform Wise Stress Level")

        fig7 = px.box(
            df,
            x="platform",
            y="stress_score",
            color="platform"
        )
        st.plotly_chart(fig7, use_container_width=True)
        st.divider()

        # EXERCISE VS DEPRESSION
        st.subheader("Exercise vs Depression")
        fig8 = px.scatter(
            df,
            x="exercise_hr",
            y="depression_binary",
            color="sentiment"
        )
        st.plotly_chart(fig8, use_container_width=True)
        st.divider()

        # LONELINESS ANALYSIS
        st.subheader("Loneliness Score Distribution")
        fig9 = px.histogram(
            df,
            x="loneliness_score",
            nbins=15,
            color="gender"
        )
        st.plotly_chart(fig9, use_container_width=True)
        st.divider()

        # SCREEN TIME DISTRIBUTION
        st.subheader("Screen Time Distribution")
        fig10 = px.bar(
            df["platform"].value_counts().reset_index(),
            x="platform",
            y="count",
            color="platform"
        )
        st.plotly_chart(fig10, use_container_width=True)
        st.divider()

        # SENTIMENT DISTRIBUTION
        st.subheader("Sentiment Distribution")
        fig11 = px.pie(
            df,
            names="sentiment"
        )
        st.plotly_chart(fig11, use_container_width=True)
        st.divider()

    if selected == "Supervise Models":
        st.title("Supervise Machine Learning Models")
        st.divider()
        df = pd.read_csv("clean_dataset.csv")
        Choice = st.selectbox("Supervise Models",
                              ["Linear Regression", "Multiple Linear Regression", "Logistic Regression",
                               "Decision Tree Classifier", "Random Forest", "K-Nearest Neighbors"])
        if Choice == "Linear Regression":
            st.subheader("Linear Regression")
            st.divider()

            X = df[["screen_time_hr"]]
            y = df["stress_score"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = linear_model.LinearRegression()
            model.fit(X_train, y_train)

            input_screen_time = st.number_input(
                "Screen Time (hrs)", min_value=0.0, max_value=24.0
            )

            if st.button("Predict"):
                prediction = model.predict([[input_screen_time]])
                st.success(f"Predicted Stress Score: {prediction[0]:.2f}")

                x_line = np.linspace(df["screen_time_hr"].min(), df["screen_time_hr"].max(), 100)
                y_line = model.predict(x_line.reshape(-1, 1))

                fig = go.Figure()

                # Scatter - actual data
                fig.add_trace(go.Scatter(
                    x=df["screen_time_hr"],
                    y=df["stress_score"],
                    mode="markers",
                    name="Actual Data",
                    marker=dict(color="steelblue", opacity=0.6)
                ))

                # Regression line
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Regression Line",
                    line=dict(color="red", width=2)
                ))

                # User input point
                fig.add_trace(go.Scatter(
                    x=[input_screen_time],
                    y=[prediction[0]],
                    mode="markers",
                    name=f"Your Input ({input_screen_time}h)",
                    marker=dict(color="green", size=12, symbol="star")
                ))

                fig.update_layout(
                    title="Screen Time vs Stress Score",
                    xaxis_title="Screen Time (hrs)",
                    yaxis_title="Stress Score"
                )

                st.plotly_chart(fig)

        if Choice == "Multiple Linear Regression":
            st.subheader("Multiple Linear Regression")
            st.divider()

            features = ["age", "social_media_usage_hr", "sleep_hr", "exercise_hr",
                        "gpa", "screen_time_hr", "loneliness_score", "anxiety_score"]
            X = df[features]
            y = df["stress_score"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            mlr = linear_model.LinearRegression()
            mlr.fit(df[features], df[["stress_score"]])

            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", min_value=0, max_value=100, value=20)  # int
            social_media_usage = col2.number_input("Social Media Usage (hrs)", min_value=0.0, max_value=24.0, value=1.0,
                                                   step=0.1)
            sleep_hr = col3.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col4, col5, col6 = st.columns(3)
            exercise = col4.number_input("Exercise (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
            gpa = col5.number_input("GPA", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
            screen_time = col6.number_input("Screen Time (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col7, col8 = st.columns(2)
            loneliness_score = col7.number_input("Loneliness Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            anxiety_score = col8.number_input("Anxiety Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            if st.button("Predict"):
                input_data = pd.DataFrame([[
                    age, social_media_usage, sleep_hr, exercise, gpa,
                    screen_time, loneliness_score, anxiety_score
                ]], columns=features)

                prediction = mlr.predict(input_data)
                st.success(f"Predicted Stress Score: {prediction[0][0]:.2f}")

        if Choice == "Logistic Regression":
            st.subheader("Logistic Regression")
            st.divider()

            features = ["age", "social_media_usage_hr", "sleep_hr", "exercise_hr",
                        "gpa", "screen_time_hr", "loneliness_score", "anxiety_score"]

            X = df[features]
            y = df["depression_binary"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.subheader("Prediction System")

            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", min_value=0, max_value=100, value=20)  # int
            social_media_usage = col2.number_input("Social Media Usage (hrs)", min_value=0.0, max_value=24.0, value=1.0,
                                                   step=0.1)
            sleep_hr = col3.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col4, col5, col6 = st.columns(3)
            exercise = col4.number_input("Exercise (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
            gpa = col5.number_input("GPA", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
            screen_time = col6.number_input("Screen Time (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col7, col8 = st.columns(2)
            loneliness_score = col7.number_input("Loneliness Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            anxiety_score = col8.number_input("Anxiety Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            if st.button("Predict"):
                input_data = pd.DataFrame([[
                    age, social_media_usage, sleep_hr, exercise, gpa,
                    screen_time, loneliness_score, anxiety_score
                ]], columns=features)

                prediction = model.predict(input_data)
                # Wrong
                st.success(f"Predicted Stress Score: {prediction[0] * 100:.2f}")

                # Correct
                if prediction[0] == 1:
                    st.error(f"High chance of Depression")
                else:
                    st.success(f"Low chance of Depression")

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                col2.metric("Precision", f"{report['weighted avg']['precision'] * 100:.2f}%")
                col3.metric("Test Size", "20%")

                st.divider()

                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format(precision=2), hide_index=False)

                st.divider()

        if Choice == "Decision Tree Classifier":
            st.subheader("Decision Tree Classifier")
            st.divider()

            oe = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
            df["depression_label"] = oe.fit_transform(df[["depression_label"]])

            features = ["age", "social_media_usage_hr", "sleep_hr", "exercise_hr",
                        "gpa", "stress_score", "loneliness_score", "anxiety_score", "screen_time_hr"]

            X = df[features]
            y = df["depression_label"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)

            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", min_value=0, max_value=100, value=20)  # int
            social_media_usage = col2.number_input("Social Media Usage (hrs)", min_value=0.0, max_value=24.0, value=1.0,
                                                   step=0.1)
            sleep_hr = col3.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col4, col5, col6 = st.columns(3)
            exercise = col4.number_input("Exercise (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
            gpa = col5.number_input("GPA", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
            screen_time = col6.number_input("Screen Time (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col7, col8, col9 = st.columns(3)
            loneliness_score = col7.number_input("Loneliness Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            anxiety_score = col8.number_input("Anxiety Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            stress_score = col9.number_input("Stress Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            if st.button("Predict"):
                input_data = [[age, social_media_usage, sleep_hr, exercise, gpa, stress_score,
                               loneliness_score, anxiety_score, screen_time]]

                prediction = model.predict(input_data)
                label_map = {0: "Low", 1: "Medium", 2: "High"}
                result = label_map[int(prediction[0])]

                if result == "High":
                    st.error(f"Depression Level: {result}")
                elif result == "Medium":
                    st.warning(f"Depression Level: {result}")
                else:
                    st.success(f"Depression Level: {result}")

                y_test_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_test_pred)
                report = classification_report(y_test, y_test_pred, output_dict=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                col2.metric("Precision", f"{report['weighted avg']['precision'] * 100:.2f}%")
                col3.metric("Test Size", "20%")

                st.divider()

                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format(precision=2), hide_index=False)

                st.divider()

        if Choice == "Random Forest":
            st.subheader("Random Forest")

            oe = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
            df["depression_label"] = oe.fit_transform(df[["depression_label"]])

            features = ["age", "social_media_usage_hr", "sleep_hr", "exercise_hr",
                        "gpa", "stress_score", "loneliness_score", "anxiety_score", "screen_time_hr"]

            X = df[features]
            y = df["depression_label"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", min_value=0, max_value=100, value=20)  # int
            social_media_usage = col2.number_input("Social Media Usage (hrs)", min_value=0.0, max_value=24.0, value=1.0,
                                                   step=0.1)
            sleep_hr = col3.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col4, col5, col6 = st.columns(3)
            exercise = col4.number_input("Exercise (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
            gpa = col5.number_input("GPA", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
            screen_time = col6.number_input("Screen Time (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col7, col8, col9 = st.columns(3)
            loneliness_score = col7.number_input("Loneliness Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            anxiety_score = col8.number_input("Anxiety Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            stress_score = col9.number_input("Stress Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            if st.button("Predict"):
                input_data = [[age, social_media_usage, sleep_hr, exercise, gpa, stress_score,
                               loneliness_score, anxiety_score, screen_time]]

                prediction = model.predict(input_data)
                label_map = {0: "Low", 1: "Medium", 2: "High"}
                result = label_map[int(prediction[0])]

                if result == "High":
                    st.error(f"Depression Level: {result}")
                elif result == "Medium":
                    st.warning(f"Depression Level: {result}")
                else:
                    st.success(f"Depression Level: {result}")

                y_test_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_test_pred)
                report = classification_report(y_test, y_test_pred, output_dict=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                col2.metric("Precision", f"{report['weighted avg']['precision'] * 100:.2f}%")
                col3.metric("Test Size", "20%")

                st.divider()

                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format(precision=2), hide_index=False)

                st.divider()

        if Choice == "K-Nearest Neighbors":
            st.subheader("K-Nearest Neighbors")
            st.divider()

            oe = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
            df["depression_label"] = oe.fit_transform(df[["depression_label"]])

            features = ["age", "social_media_usage_hr", "sleep_hr", "exercise_hr",
                        "gpa", "stress_score", "loneliness_score", "anxiety_score", "screen_time_hr"]

            X = df[features]
            y = df["depression_label"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            knn = KNeighborsClassifier(n_neighbors=10, metric="euclidean")
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", min_value=0, max_value=100, value=20)  # int
            social_media_usage = col2.number_input("Social Media Usage (hrs)", min_value=0.0, max_value=24.0, value=1.0,
                                                   step=0.1)
            sleep_hr = col3.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col4, col5, col6 = st.columns(3)
            exercise = col4.number_input("Exercise (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
            gpa = col5.number_input("GPA", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
            screen_time = col6.number_input("Screen Time (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

            col7, col8, col9 = st.columns(3)
            loneliness_score = col7.number_input("Loneliness Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            anxiety_score = col8.number_input("Anxiety Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            stress_score = col9.number_input("Stress Score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            if st.button("Predict"):
                input_data = [[age, social_media_usage, sleep_hr, exercise, gpa, stress_score,
                               loneliness_score, anxiety_score, screen_time]]

                input_scaled = scaler.transform(input_data)
                prediction = knn.predict(input_scaled)

                label_map = {0: "Low", 1: "Medium", 2: "High"}
                result = label_map[int(prediction[0])]

                if result == "High":
                    st.error(f"Depression Level: {result}")
                elif result == "Medium":
                    st.warning(f"Depression Level: {result}")
                else:
                    st.success(f"Depression Level: {result}")

                y_test_pred = knn.predict(scaler.transform(X_test))
                accuracy = accuracy_score(y_test, y_test_pred)
                report = classification_report(y_test, y_test_pred, output_dict=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
                col2.metric("Precision", f"{report['weighted avg']['precision'] * 100:.2f}%")
                col3.metric("Test Size", "20%")

                st.divider()

                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format(precision=2), hide_index=False)

                st.divider()

    if selected == "UnSupervise Models":
        st.subheader("UnSupervise Models")
        df = pd.read_csv("Final.csv")
        st.divider()
        Choice1 = st.selectbox("Models", ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN"])

        if Choice1 == "K-Means Clustering":
            st.subheader("K-Means Clustering")
            st.divider()

            features = ["age", "social_media_usage_hr", "sleep_hr", "exercise_hr",
                        "gpa", "stress_score", "loneliness_score", "anxiety_score", "screen_time_hr"]
            x = df[features]

            kmeans = KMeans(n_clusters=3, random_state=42)
            df["cluster"] = kmeans.fit_predict(x)

            # Chart 1: Age vs Stress Score
            fig1 = px.scatter(
                df,
                x="age",
                y="stress_score",
                color=df["cluster"].astype(str),
                title="Age vs Stress Score Clusters",
                labels={"color": "Cluster"},
                color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.divider()

            #  Chart 2: Screen Time vs Loneliness
            fig2 = px.scatter(
                df,
                x="screen_time_hr",
                y="loneliness_score",
                color=df["cluster"].astype(str),
                title="Screen Time vs Loneliness Score Clusters",
                labels={"color": "Cluster"},
                color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.divider()

            # Cluster Summary Table
            st.subheader("Cluster Summary")
            cluster_summary = df.groupby("cluster")[features].mean().round(2)
            st.dataframe(cluster_summary, use_container_width=True)

        if Choice1 == "Hierarchical Clustering":
            st.subheader("Hierarchical Clustering")
            st.divider()

            df = pd.read_csv("clean_dataset.csv")

            features = ["age", "social_media_usage_hr", "sleep_hr", "exercise_hr",
                        "gpa", "stress_score", "loneliness_score", "anxiety_score", "screen_time_hr"]

            X_full = df[features]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_full)

            X_sample = X_scaled[:200]
            fig = ff.create_dendrogram(X_sample, orientation="bottom")
            fig.update_layout(
                title="Dendrogram (Find Optimal Clusters)",
                xaxis_title="Data Points",
                yaxis_title="Distance",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

            model = AgglomerativeClustering(n_clusters=3, linkage="ward")
            clusters = model.fit_predict(X_scaled)
            df["Cluster"] = clusters

            plt.figure(figsize=(12, 6))
            colors = ["green", "blue", "red"]

            for i in range(3):
                cluster_data = df[df["Cluster"] == i]
                plt.scatter(
                    cluster_data["age"],
                    cluster_data["stress_score"],
                    color=colors[i],
                    label=f"Cluster {i}"
                )

            plt.xlabel("Age")
            plt.ylabel("Stress Score")
            plt.title("Age vs Stress Score — Hierarchical Clusters")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(plt)
            st.divider()

            st.subheader("Cluster Summary")
            cluster_summary = df.groupby("Cluster")[features].mean().round(2)
            st.dataframe(cluster_summary)

        if Choice1 == "DBSCAN":
            st.subheader("Density-Based Spatial Clustering of Applications with Noise(DBSCAN)")
            df = pd.read_csv("clean_dataset.csv")
            st.divider()

            features = ["age", "social_media_usage_hr", "sleep_hr", "exercise_hr",
                        "gpa", "stress_score", "loneliness_score", "anxiety_score", "screen_time_hr"]
            x = df[features]

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            db = DBSCAN(eps=0.2, min_samples=5)

            label = db.fit_predict(x_scaled)

            plt.figure()
            plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=label, cmap="rainbow")
            plt.colorbar(label="clusters")
            st.pyplot(plt)
            st.divider()

    if selected == "NLP & N-GRAM":
        st.subheader("Natural Language Processing")
        df = pd.read_csv("clean_dataset.csv")

        st.divider()
        Choice2 = st.selectbox("NLP and N-GRAMS", ["NLP", "N-GRAM"])

        if Choice2 == "NLP":
            df = df[["sentence", "sentiment"]]
            df.dropna(inplace=True)

            le = LabelEncoder()
            df["sentiment"] = le.fit_transform(df["sentiment"])
            st.write("Label Mapping....", dict(zip(le.classes_, le.transform(le.classes_))))

            stop_words = set(stopwords.words("english"))

            def clean_text(text):
                text = str(text).lower()
                text = re.sub(r"[^a-zA-Z]", " ", text)
                words = text.split()
                words = [w for w in words if w not in stop_words]
                return " ".join(words)

            df["clean_text"] = df["sentence"].apply(clean_text)
            st.divider()
            tab1, tab2, tab3 = st.tabs(["TF-IDF", "Count Vectorization", "Vader Sentiment"])

            with tab1:
                st.subheader("Term-Frequency Inverse Document Frequency")
                st.divider()

                vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
                X = vectorizer.fit_transform(df["clean_text"])
                y = df["sentiment"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model = SVC(kernel="linear", C=1.5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                def predict_sentiment_tfidf(user_input):
                    text = clean_text(user_input)
                    vectorized = vectorizer.transform([text])
                    pred = model.predict(vectorized)[0]
                    return le.inverse_transform([pred])[0]

                user_input = st.text_area("\n Enter a review : ", key="tfidf_input")
                button = st.button("Predict Sentiment", key="tfidf_btn")
                if button:
                    prediction = predict_sentiment_tfidf(user_input)
                    st.write("Predicted Sentiment : ", prediction)
                    st.divider()
                    st.write("\n Accuracy: ", accuracy_score(y_test, y_pred))
                    st.write("\n Classification Report: \n", classification_report(y_test, y_pred))

            with tab2:
                st.subheader("Count Vectorization")
                st.divider()

                vectorizer2 = CountVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
                X = vectorizer2.fit_transform(df["clean_text"])
                y = df["sentiment"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model2 = SVC(kernel="linear", C=1.5)
                model2.fit(X_train, y_train)
                y_pred = model2.predict(X_test)

                def predict_sentiment_count(user_input):
                    text = clean_text(user_input)
                    vectorized = vectorizer2.transform([text])
                    pred = model2.predict(vectorized)[0]
                    return le.inverse_transform([pred])[0]

                user_input = st.text_area("\n Enter a review : ", key="count_input")
                button = st.button("Predict Sentiment", key="count_btn")
                if button:
                    prediction = predict_sentiment_count(user_input)
                    st.write("Predicted Sentiment : ", prediction)
                    st.write("Label Mapping....", dict(zip(le.classes_, le.transform(le.classes_))))
                    st.write("\n Accuracy: ", accuracy_score(y_test, y_pred))
                    st.write("\n Classification Report: \n", classification_report(y_test, y_pred))

            with tab3:
                st.subheader("Vader Sentiment Analysis")
                st.divider()

                analyzer = SentimentIntensityAnalyzer()

                user_input = st.text_area("\n Enter a review : ", key="vader_input")
                button = st.button("Predict Sentiment", key="vader_btn")
                if button:
                    scores = analyzer.polarity_scores(user_input)
                    compound = scores["compound"]

                    if compound >= 0.05:
                        label = "Positive"
                    elif compound <= -0.05:
                        label = "Negative"
                    else:
                        label = "Neutral"

                    st.write("Predicted Sentiment : ", label)
                    st.divider()
                    st.write("Vader Scores : ", scores)

        if Choice2 == "N-GRAM":
            st.subheader("N-GRAM")
            st.divider()
            df = pd.read_csv("clean_dataset.csv")

            with st.expander("All Columns"):
                st.write("Columns : ", df.columns)

            text_data = df["sentence"].dropna().astype(str)
            full_text = " ".join(text_data).lower()

            tokens = nltk.word_tokenize(full_text)

            tab1, tab2 = st.tabs(["Bigrams", "Trigrams"])

            with tab1:
                bigrams = list(ngrams(tokens, 2))

                bigrams_model = {}
                for w1, w2 in bigrams:
                    bigrams_model.setdefault(w1, []).append(w2)

                def generate_bigram(start_word, length):
                    word = start_word.lower()
                    sentence = [word]
                    for _ in range(length):
                        next_words = bigrams_model.get(word)
                        if not next_words:
                            break
                        next_word = random.choice(next_words)
                        sentence.append(next_word)
                        word = next_word
                    return " ".join(sentence)

                start = st.text_input("Enter Starting word ")
                length = st.slider("Enter sentence length : ", 1, 500, key="bigram_slider")
                button = st.button("Bigram Predict", key="bigram_btn")
                if button:
                    st.success(generate_bigram(start, length))

            with tab2:
                trigrams = list(ngrams(tokens, 3))

                trigrams_model = {}
                for w1, w2, w3 in trigrams:
                    trigrams_model.setdefault((w1, w2), []).append(w3)

                def generate_trigram(w1, w2, length):
                    w1, w2 = w1.lower(), w2.lower()
                    sentence = [w1, w2]
                    for _ in range(length):
                        next_words = trigrams_model.get((w1, w2))
                        if not next_words:
                            break
                        next_word = random.choice(next_words)
                        sentence.append(next_word)
                        w1, w2 = w2, next_word
                    return " ".join(sentence)

                start1 = st.text_input("Enter Starting word ", key="start1")
                start2 = st.text_input("Enter Starting word ", key="start2")
                length = st.slider("Enter sentence length : ", 1, 500, key="trigram_slider")
                button = st.button("Trigram Predict", key="trigram_btn")
                if button:
                    st.success(generate_trigram(start1, start2, length))

    if selected == "DeepLearning Models":
        st.subheader("DeepLearning Models")
        df = pd.read_csv("clean_dataset_updated.csv")
        st.divider()

        TEXT_COLUMN = "platform_content"
        NAME_COLUMN = "platform"

        # Clean Text Function
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            return text.strip()

        df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

        bert_model = SentenceTransformer("all-MiniLM-L6-v2")

        bert_embeddings = bert_model.encode(
            df["clean_text"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True
        )

        def recommend_bert(platform, top_n=5):
            if platform not in df[NAME_COLUMN].values:
                return "Platform Not Found"

            idx = df[df[NAME_COLUMN] == platform].index[0]

            scores = cosine_similarity(
                [bert_embeddings[idx]],
                bert_embeddings
            )[0]

            top_indices = scores.argsort()[::-1][1:top_n + 1]

            return df[NAME_COLUMN].iloc[top_indices].tolist()

        def search_text(query, top_n=5):
            query = clean_text(query)

            query_embedding = bert_model.encode(
                [query],
                normalize_embeddings=True
            )

            scores = cosine_similarity(
                query_embedding,
                bert_embeddings
            )[0]

            top_indices = scores.argsort()[::-1][:top_n]

            return df.iloc[top_indices][[NAME_COLUMN, TEXT_COLUMN]]

        # Search + Recommendation Function
        def search_and_recommend(query, top_n=5):
            query = clean_text(query)

            query_embedding = bert_model.encode(
                [query],
                normalize_embeddings=True
            )

            scores = cosine_similarity(
                query_embedding,
                bert_embeddings
            )[0]

            best_idx = scores.argmax()

            best_platform = df.iloc[best_idx][NAME_COLUMN]

            recommendation = recommend_bert(best_platform, top_n)

            return {
                "searched_platform": best_platform,
                "recommendation": recommendation
            }

        # Search Input
        user_query = st.text_input("Enter what you want to search:")

        if user_query.strip():
            st.write("### Top Matches:")
            st.dataframe(search_text(user_query))

            st.divider()

            result = search_and_recommend(user_query)
            st.write("### Best Match:", result["searched_platform"])
            st.write("### Recommendations:")
            st.write(result["recommendation"])
            st.divider()

    if selected == "LLM":
        st.subheader("LLM")
        df = pd.read_csv("clean_dataset.csv")
        st.divider()

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        st.title("Ai Social Media Limitation Platform")

        st.header("Enter Details")

        df = pd.read_csv("clean_dataset_updated.csv")

        concern = st.sidebar.text_area("Your Mental Health Concern or Query")

        platform = st.sidebar.selectbox(
            "Select Platform",
            options=sorted(df["platform"].dropna().unique().tolist())
        )

        gender = st.sidebar.selectbox(
            "Gender",
            options=sorted(df["gender"].dropna().unique().tolist())
        )

        age = st.sidebar.number_input(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=int(df["age"].mean())
        )

        usage_hr = st.sidebar.slider(
            "Daily Social Media Usage (hours)",
            min_value=float(df["social_media_usage_hr"].min()),
            max_value=float(df["social_media_usage_hr"].max()),
            value=float(df["social_media_usage_hr"].mean())
        )

        sleep_hr = st.sidebar.slider(
            "Sleep Hours",
            min_value=float(df["sleep_hr"].min()),
            max_value=float(df["sleep_hr"].max()),
            value=float(df["sleep_hr"].mean())
        )

        exercise_hr = st.sidebar.slider(
            "Exercise Hours per Day",
            min_value=float(df["exercise_hr"].min()),
            max_value=float(df["exercise_hr"].max()),
            value=float(df["exercise_hr"].mean())
        )

        tab1, tab2 = st.tabs(["Mental Health Analysis", "Wellness Roadmap"])

        def get_similar_users():
            filtered = df[(df["platform"] == platform) & (df["gender"] == gender)]
            return filtered if not filtered.empty else df[df["platform"] == platform]

        def analyze_mental_health():
            similar = get_similar_users()
            avg_stress = round(similar["stress_score"].mean(), 2)
            avg_anxiety = round(similar["anxiety_score"].mean(), 2)
            avg_loneliness = round(similar["loneliness_score"].mean(), 2)
            avg_sleep = round(similar["sleep_hr"].mean(), 2)
            avg_usage = round(similar["social_media_usage_hr"].mean(), 2)
            depression_pct = round(similar["depression_binary"].mean() * 100, 1)
            common_sentiment = (
                similar["sentiment"].mode()[0]
                if not similar["sentiment"].empty else "N/A"
            )

            prompt = f"""
            You are a Mental Health Expert and Data Analyst.

            User Profile:
            - Age                        : {age}
            - Gender                     : {gender}
            - Platform Used              : {platform}
            - Daily Social Media Usage   : {usage_hr} hours
            - Sleep Hours                : {sleep_hr} hours
            - Exercise Hours             : {exercise_hr} hours
            - Personal Concern           : {concern if concern.strip() else "General mental health analysis"}

            Similar Users Data (same platform & gender from dataset):
            - Average Stress Score       : {avg_stress}
            - Average Anxiety Score      : {avg_anxiety}
            - Average Loneliness Score   : {avg_loneliness}
            - Average Sleep Hours        : {avg_sleep}
            - Average Social Media Usage : {avg_usage} hours
            - Depression Rate            : {depression_pct}%
            - Common Sentiment           : {common_sentiment}

            Provide a structured mental health analysis:
            1. Overall Mental Health Assessment
            2. Impact of {platform} Usage on Mental Health
            3. Sleep & Exercise Analysis
            4. Stress & Anxiety Risk Level
            5. Depression Risk based on dataset patterns
            6. Loneliness & Social Isolation Indicators
            7. Sentiment Pattern Analysis
            8. Key Strengths & Concerns
            9. Personalized Recommendations
            10. Final Verdict (Healthy / Moderate Risk / High Risk)

            Keep it structured, empathetic, and data-driven.
            """

            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a mental health analyst and data scientist."},
                    {"role": "user", "content": prompt}
                ]
            )
            return res.choices[0].message.content

        def generate_roadmap():
            similar = get_similar_users()
            avg_stress = round(similar["stress_score"].mean(), 2)
            avg_anxiety = round(similar["anxiety_score"].mean(), 2)
            avg_sleep = round(similar["sleep_hr"].mean(), 2)

            prompt = f"""
            You are a Mental Health Coach and Wellness Mentor.

            User Profile:
            - Age            : {age}
            - Gender         : {gender}
            - Platform       : {platform}
            - Daily Usage    : {usage_hr} hours
            - Sleep          : {sleep_hr} hours
            - Exercise       : {exercise_hr} hours
            - Concern        : {concern if concern.strip() else "General wellness improvement"}

            Dataset Benchmarks (similar users):
            - Avg Stress Score  : {avg_stress}
            - Avg Anxiety Score : {avg_anxiety}
            - Avg Sleep Hours   : {avg_sleep}

            Create a practical mental wellness improvement roadmap:
            1. First 5 Immediate Steps to Improve Mental Health
            2. Healthy Social Media Usage Plan for {platform}
            3. Sleep & Exercise Improvement Strategy
            4. Stress & Anxiety Management Techniques
            5. Long-term Wellness Goals (30 / 60 / 90 days)
            6. Common Mistakes to Avoid

            Keep it actionable, beginner-friendly, and encouraging.
            """

            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a mental health coach."},
                    {"role": "user", "content": prompt}
                ]
            )
            return res.choices[0].message.content

        with tab1:
            st.subheader("Analyze Your Mental Health")

            similar_users = get_similar_users()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Stress Score", round(similar_users["stress_score"].mean(), 1))
            col2.metric("Avg Anxiety Score", round(similar_users["anxiety_score"].mean(), 1))
            col3.metric("Avg Sleep Hours", round(similar_users["sleep_hr"].mean(), 1))
            col4.metric("Depression Rate", f"{round(similar_users['depression_binary'].mean() * 100, 1)}%")

            st.divider()

            if st.button("Analyze Mental Health"):
                if concern.strip() == "":
                    st.warning("Please enter your concern or query in the sidebar.")
                else:
                    with st.spinner("Analyzing your mental health profile..."):
                        result = analyze_mental_health()
                    st.success("Analysis Completed")
                    st.write(result)

        with tab2:
            st.header("Your Wellness Roadmap")

            if st.button("Generate Roadmap"):
                if concern.strip() == "":
                    st.warning("Please enter your concern or query in the sidebar.")
                else:
                    with st.spinner("Generating your wellness roadmap..."):
                        roadmap = generate_roadmap()
                    st.success("Roadmap Generated Successfully")
                    st.write(roadmap)

    if selected == "Exit":
        st.subheader("Exit")
        st.divider()
        if st.button("Log Out"):
            st.session_state.page = "Login"
            st.session_state.logged_in = False
            st.session_state.username = ""

if st.session_state.page == "Signup":
    signup_page()

elif st.session_state.page == "Login":
    login_page()

elif st.session_state.page == "Home":
    if st.session_state.logged_in:
        home_page()
    else:
        login_page()










































































































# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 📥 Load and clean dataset
@st.cache_data
def load_data():
    df = pd.read_csv('AB_NYC_2019.CSV')
    df['name'].fillna('Unknown', inplace=True)
    df['host_name'].fillna('Unknown', inplace=True)
    df['last_review'].fillna(pd.to_datetime('2000-01-01'), inplace=True)
    df['reviews_per_month'].fillna(0.0, inplace=True)
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df = df.drop_duplicates()

    # Outlier removal
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    df = remove_outliers_iqr(df, 'price')
    df = remove_outliers_iqr(df, 'minimum_nights')

    def cap_outliers(df, column):
        lower_cap = df[column].quantile(0.01)
        upper_cap = df[column].quantile(0.99)
        df[column] = df[column].clip(lower_cap, upper_cap)
        return df

    df = cap_outliers(df, 'price')
    return df

df = load_data()

# 🎯 Sidebar - Navigation Menu
st.sidebar.title("🏠 Airbnb NYC Dashboard")
menu = st.sidebar.radio("Choose Section", [
    "Dataset Overview",
    "Custom Questions",
    "Analysis Questions"
])

# --- Dataset Overview ---
if menu == "Dataset Overview":
    st.title("📊 Airbnb NYC 2019 Dataset Overview")
    st.write("Shape of Dataset:", df.shape)
    st.dataframe(df.head(3))
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    st.subheader("Boxplot: Price")
    st.pyplot(sns.boxplot(x=df['price']).figure)

# --- Custom Q1–Q9 Questions ---
elif menu == "Custom Questions":
    st.title("❓ Answer Custom Data Questions")
    question = st.selectbox("Choose a question", [
        "1. Neighbourhood with most listings",
        "2. Host with most properties",
        "3. Top host in top neighbourhood",
        "4. Average property price",
        "5. Reason for high prices",
        "6. Most preferred room type by group",
        "7. Total availability by room type",
        "8. Busiest host (reviews)",
        "9. Property with most reviews"
    ])

    if question.startswith("1"):
        result = df['neighbourhood_group'].value_counts().idxmax()
        st.write(f"Neighbourhood group with maximum listings: **{result}**")
    elif question.startswith("2"):
        result = df['host_name'].value_counts().idxmax()
        st.write(f"Host with most properties: **{result}**")
    elif question.startswith("3"):
        top_group = df['neighbourhood_group'].value_counts().idxmax()
        result = df[df['neighbourhood_group'] == top_group]['host_name'].value_counts().idxmax()
        st.write(f"Top host in {top_group}: **{result}**")
    elif question.startswith("4"):
        avg_price = df['price'].mean()
        st.write(f"Average price: **${avg_price:.2f}**")
    elif question.startswith("5"):
        st.write("Possible reasons for high prices:")
        st.markdown("""
        - Prime location  
        - Tourist attractions  
        - High demand  
        - Better infrastructure  
        - Premium listings  
        """)
    elif question.startswith("6"):
        result = df.groupby('neighbourhood_group')['room_type'].agg(lambda x: x.value_counts().idxmax())
        st.write(result)
    elif question.startswith("7"):
        result = df.groupby('room_type')['availability_365'].sum()
        st.write(result)
    elif question.startswith("8"):
        busiest = df.groupby('host_name')['number_of_reviews'].sum().idxmax()
        st.write(f"Busiest host: **{busiest}**")
    elif question.startswith("9"):
        max_review = df['number_of_reviews'].idxmax()
        property_info = df.loc[max_review]
        st.write("Property with maximum reviews:")
        st.write(property_info[['name', 'host_name', 'neighbourhood_group', 'number_of_reviews']])

# --- Analysis Questions (from menu-based input) ---
elif menu == "Analysis Questions":
    st.title("📌 Choose an Analysis Question")

    analysis_choice = st.selectbox(
        "Select an analysis to perform:",
        [
            "1. Count of reviews per month",
            "2. Show total room types",
            "3. Room type count plot",
            "4. Filter by last reviewed date",
            "5. Prices by neighborhood group",
            "6. Host count per neighborhood group"
        ]
    )

    if "1" in analysis_choice:
        st.subheader("Reviews per Month Distribution")
        bins = [0, 0.1, 0.5, 1, 2, 3, 5, 10, df['reviews_per_month'].max()]
        labels = ['0-0.1', '0.1-0.5', '0.5-1', '1-2', '2-3', '3-5', '5-10', '10+']
        df['reviews_per_month_bin'] = pd.cut(df['reviews_per_month'], bins=bins, labels=labels, include_lowest=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='reviews_per_month_bin', data=df, palette='Blues_d', ax=ax)
        ax.set_title("Distribution of Reviews per Month", fontsize=16)
        ax.set_xlabel("Reviews per Month Range", fontsize=12)
        ax.set_ylabel("Number of Properties", fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif "2" in analysis_choice:
        st.subheader("Room Types and Their Counts")
        st.write(df['room_type'].value_counts())

    elif "3" in analysis_choice:
        st.subheader("Room Type Count Plot")
        fig, ax = plt.subplots(figsize=(7, 4))
        df['room_type'].value_counts().plot(kind='bar', color=['red', 'green', 'blue'], ax=ax)
        ax.set_title("Room Type Distribution")
        ax.set_xlabel("Room Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif "4" in analysis_choice:
        st.subheader("Filter by Last Reviewed Date")
        date_input = st.date_input("Select a date to filter properties reviewed after this date:", pd.to_datetime('2018-01-01'))
        filtered = df[df['last_review'] >= pd.to_datetime(date_input)]
        st.write(f"Total properties reviewed after {date_input}: {len(filtered)}")
        st.dataframe(filtered[['name', 'last_review', 'neighbourhood_group']].head())

    elif "5" in analysis_choice:
        st.subheader("Average Price by Neighborhood Group")
        avg_price = df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        avg_price.plot(kind='bar', color='orange', ax=ax)
        ax.set_title("Average Price by Neighbourhood Group")
        ax.set_ylabel("Average Price")
        st.pyplot(fig)

    elif "6" in analysis_choice:
        st.subheader("Host Count per Neighborhood Group")
        table = df.groupby(['host_name', 'neighbourhood_group']).size().unstack(fill_value=0)
        st.dataframe(table.head(10))

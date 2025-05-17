# movie_dashboard_app.py
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from fcmeans import FCM

# --- Load Data ---
@st.cache_data

def load_data():
    movies1 = pd.read_csv("df_movies_clean1.csv")
    ratings = pd.read_csv("ratings1.csv")
    tags = pd.read_csv("tags.csv")
    return movies1, ratings, tags

movies1, ratings, tags = load_data()

# --- Preprocess Movies Data ---
movies1['Year'] = movies1['title'].str.extract(r'\((\d{4})\)').astype(float)
movies1['genres'] = movies1['genres'].str.split('|')
genre_year_df = movies1.explode('genres').rename(columns={'genres': 'Genre'})

# --- Genre-Year Heatmap Data ---
genre_counts = (
    genre_year_df.groupby(['Genre', 'Year'])
    .size()
    .reset_index(name='Movie Count')
)
heatmap = genre_counts.pivot(index='Genre', columns='Year', values='Movie Count').fillna(0)

# --- Preprocess Ratings Data ---
ratings1['timestamp'] = pd.to_datetime(ratings1['timestamp'], unit='s')
ratings1['dayofweek'] = ratings1['timestamp'].dt.day_name()
ratings1['month'] = ratings1['timestamp'].dt.month
ratings1['year'] = ratings1['timestamp'].dt.year

ratings_week = ratings1.groupby('dayofweek', sort=False)['rating'].mean().reset_index()

# --- Yearly Aggregates ---
rating_count_by_year = ratings1.groupby('year').size().reset_index(name='Rating Count')
avg_rating_by_year = ratings1.groupby('year')['rating'].mean().reset_index(name='Average Rating')
ratings_yearly = rating_count_by_year.merge(avg_rating_by_year, on='year')

# --- Genre Trends Prep ---
df_genre = pd.read_csv("df_genre.csv")

df_melted = df_genre.melt(
    id_vars=['rating', 'release_year'],
    value_vars=['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
    var_name='Genre',
    value_name='IsGenre'
)
df_melted = df_melted[df_melted['IsGenre'] == 1].copy()

# --- Clustering Setup ---
numeric_features = [
    'avg_rating_movie', 'num_ratings_movie', 'rating_std_movie',
    'rating_year_range', 'rating_trend', 'genre_count', 'release_decade'
]

# --- Dashboard Tabs ---
tabs = st.tabs([
    "Genre-Year Heatmap", "Ratings Overview", "Yearly Ratings",
    "Genre Trends", "K-Means Clustering", "K-Medoids Clustering", "Fuzzy C-Means Clustering"
])

# Tabs 0–3 already implemented above...

# ------------------------------
# Tab 5: K-Means Clustering
# ------------------------------
with tabs[4]:
    st.title("\U0001F4C9 K-Means Clustering")
    x_col = st.selectbox("Select X-axis Feature", numeric_features, index=0)
    y_col = st.selectbox("Select Y-axis Feature", numeric_features, index=1)
    k = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=3)

    df_clean = movies1[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    df_clean['Cluster'] = labels
    centroids = scaler.inverse_transform(model.cluster_centers_)

    fig = px.scatter(
        df_clean, x=x_col, y=y_col,
        color=df_clean['Cluster'].astype(str),
        title=f"K-Means Clustering (k={k})",
        height=550
    )
    fig.add_scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers',
        marker=dict(color='black', size=14, symbol='x'),
        name='Centroids'
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 6: K-Medoids Clustering
# ------------------------------
with tabs[5]:
    st.title("\U0001F52C K-Medoids Clustering")
    x_col = st.selectbox("X-axis Feature", numeric_features, index=0, key="kmed_x")
    y_col = st.selectbox("Y-axis Feature", numeric_features, index=1, key="kmed_y")
    k = st.slider("Number of Clusters", min_value=2, max_value=8, value=3, key="kmed_k")

    df_clean = movies1[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    df_clean['Cluster'] = labels
    centroids = scaler.inverse_transform(model.cluster_centers_)

    fig = px.scatter(
        df_clean, x=x_col, y=y_col,
        color=df_clean['Cluster'].astype(str),
        title=f"K-Medoids Clustering (k={k})",
        height=550
    )
    fig.update_traces(marker=dict(size=6, opacity=0.6))
    fig.add_scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers',
        marker=dict(color='black', size=14, symbol='x'),
        name='Medoids'
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 7: Fuzzy C-Means Clustering
# ------------------------------
with tabs[6]:
    st.title("\U0001F440 Fuzzy C-Means Clustering")
    x_col = st.selectbox("X-axis Feature", numeric_features, index=0, key="fuzzy_x")
    y_col = st.selectbox("Y-axis Feature", numeric_features, index=1, key="fuzzy_y")

    df_clean = movies1[[x_col, y_col]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    fcm = FCM(n_clusters=3, random_state=42)
    fcm.fit(X_scaled)
    labels = fcm.predict(X_scaled)
    memberships = fcm.u.max(axis=0)
    centroids = scaler.inverse_transform(fcm.centers)

    df_clean['Cluster'] = labels
    df_clean['Membership'] = memberships

    fig = px.scatter(
        df_clean, x=x_col, y=y_col,
        color=df_clean['Cluster'].astype(str),
        size='Membership',
        opacity=0.65,
        title="Fuzzy C-Means Clustering (c=3)",
        height=550
    )
    fig.add_scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode='markers',
        marker=dict(color='black', size=14, symbol='x'),
        name='Centroids'
    )
    st.plotly_chart(fig, use_container_width=True)

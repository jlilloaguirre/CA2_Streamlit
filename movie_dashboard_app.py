# movie_dashboard_app.py
import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv", encoding="latin1")
    ratings1 = pd.read_csv("ratings1.csv", encoding="latin1")
    df_genre = pd.read_csv("df_genre.csv", encoding="latin1")
    return movies, ratings1, df_genre

movies, ratings1, df_genre = load_data()

# --- Preprocess Movies Data ---
movies['Year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['genres'] = movies['genres'].str.split('|')
genre_year_df = movies.explode('genres').rename(columns={'genres': 'Genre'})

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

rating_count_by_year = ratings1.groupby('year').size().reset_index(name='Rating Count')
avg_rating_by_year = ratings1.groupby('year')['rating'].mean().reset_index(name='Average Rating')
ratings_yearly = rating_count_by_year.merge(avg_rating_by_year, on='year')

# --- Genre Trends Prep ---
df_melted = df_genre.melt(
    id_vars=['rating', 'release_year'],
    value_vars=['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
    var_name='Genre',
    value_name='IsGenre'
)
df_melted = df_melted[df_melted['IsGenre'] == 1].copy()

# --- Dashboard Tabs ---
tabs = st.tabs([
    "Genre-Year Heatmap", "Ratings Overview", "Yearly Ratings", "Genre Trends"
])

# ------------------------------
# Tab 1: Genre-Year Movie Heatmap
# ------------------------------
with tabs[0]:
    st.title("\U0001F3AC Genre-Year Movie Heatmap")
    st.markdown("Use the slider below to explore how movie genres evolved over time:")

    min_year = int(heatmap.columns.min())
    max_year = int(heatmap.columns.max())
    year_range = st.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    filtered = heatmap.loc[:, year_range[0]:year_range[1]]
    fig = px.imshow(
        filtered,
        aspect='auto',
        color_continuous_scale='viridis',
        labels=dict(color='Movie Count'),
        title="Movie Genre Frequency by Year"
    )
    fig.update_layout(
        font=dict(size=13),
        height=600,
        xaxis_title="Year",
        yaxis_title="Genre",
        margin=dict(l=60, r=30, t=60, b=60),
        coloraxis_colorbar=dict(title='Number of Movies')
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 2: Ratings Overview
# ------------------------------
with tabs[1]:
    st.title("\U0001F4CA Ratings Data Overview")
    st.markdown("**This dashboard explores user rating behavior by weekday and month.**")

    custom_scale = px.colors.sequential.Viridis[2:10] + px.colors.sequential.OrRd[1:7]

    fig_day = px.bar(
        ratings_week,
        x='dayofweek',
        y='rating',
        title='Average Rating by Day of the Week',
        color='rating',
        color_continuous_scale=custom_scale,
        labels={'dayofweek': 'Day of Week', 'rating': 'Average Rating'},
        template='plotly_white'
    )
    fig_day.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Average Rating',
        font=dict(size=13),
        height=450,
        yaxis=dict(range=[3.46, 3.54])
    )
    st.plotly_chart(fig_day, use_container_width=True)

    monthly_counts = ratings1['month'].value_counts().sort_index().reset_index()
    monthly_counts.columns = ['Month', 'Rating Count']

    fig_month = px.bar(
        monthly_counts,
        x='Month',
        y='Rating Count',
        title='Number of Ratings per Month',
        color='Rating Count',
        color_continuous_scale=custom_scale,
        template='plotly_white'
    )
    fig_month.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Ratings',
        font=dict(size=13),
        height=450
    )
    st.plotly_chart(fig_month, use_container_width=True)

# ------------------------------
# Tab 3: Yearly Ratings
# ------------------------------
with tabs[2]:
    st.title("\U0001F4C8 Yearly Ratings Analysis")
    st.markdown("**Click below to explore yearly metrics for rating count and average rating.**")

    metric = st.selectbox("Select Metric", options=['Rating Count', 'Average Rating'])

    fig = px.bar(
        ratings_yearly,
        x='year',
        y=metric,
        title=f'{metric} per Year',
        color=metric,
        color_continuous_scale=custom_scale,
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=metric,
        font=dict(size=13),
        height=500,
        margin=dict(l=60, r=30, t=60, b=60)
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tab 4: Genre Rating Trends
# ------------------------------
with tabs[3]:
    st.title("\U0001F4D1 Genre Rating Trends")
    st.markdown("Select one or more genres and a year range to explore rating trends.")

    genres = sorted(df_melted['Genre'].unique())
    selected_genres = st.multiselect("Select Genres", options=genres, default=['Drama'])

    year_range = st.slider(
        "Select Release Year Range:",
        min_value=int(df_melted['release_year'].min()),
        max_value=int(df_melted['release_year'].max()),
        value=(2000, 2015),
        step=1
    )

    filtered = df_melted[
        (df_melted['Genre'].isin(selected_genres)) &
        (df_melted['release_year'].between(year_range[0], year_range[1]))
    ]

    if filtered.empty:
        st.warning("No data available for the selected filters.")
    else:
        trend_data = (
            filtered.groupby(['Genre', 'release_year'])['rating']
            .mean().reset_index()
        )

        fig = px.line(
            trend_data,
            x='release_year',
            y='rating',
            color='Genre',
            markers=True,
            template='plotly_white',
            title='Average Rating by Genre Over Time',
            labels={'rating': 'Average Rating', 'release_year': 'Release Year'}
        )

        fig.update_layout(
            height=550,
            font=dict(size=13),
            xaxis=dict(tickmode='linear'),
            margin=dict(l=40, r=20, t=50, b=60),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

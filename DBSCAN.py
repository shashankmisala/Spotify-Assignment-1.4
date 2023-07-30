import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_csv('spotify_tracks.csv')

# Select the relevant features for clustering
df = df[['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','artists','album_name','track_name']]

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']])

# Train the DBSCAN model
eps = 0.2 
min_samples = 2  
dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_model.fit(X_scaled)

clusters = dbscan_model.labels_

# Function to recommend songs
def recommend_songs(user_preferences):
    # Preprocess user data
    user = scaler.transform([user_preferences])

    # Get the cluster label for the user's preferences
    user_cluster = dbscan_model.labels_[-1]

    # Filter dataset to get recommendations from the same cluster
    recommended_songs = df[clusters == user_cluster][['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                                                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

    return recommended_songs
    
# Main Streamlit app
def main():
    st.title('Spotify Recommendation System With DBSCAN')
    st.write('Input your preferences below and get song recommendations!')

    danceability = st.slider('Danceability', 0, 100, 25,10)
    energy = st.slider('Energy', 0, 100, 25,10)
    key = st.slider('Key', 0, 100, 25,10)
    loudness = st.slider('Loudness', 0, 100, 25,10)
    mode = st.slider('Mode', 0, 100, 25,10)
    speechiness = st.slider('Speechiness', 0, 100, 25,10)
    acousticness = st.slider('Acousticness', 0, 100, 25,10)
    instrumentalness = st.slider('Instrumentalness', 0,100,25,10)
    liveness = st.slider('Liveness', 0, 100, 25,10)
    valence = st.slider('Valence', 0, 100, 25,10)
    tempo = st.slider('Tempo', 0, 100, 25,10)

    user_preferences = [danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo]
#predicting recommendations
    if st.button('Get Recommendations'):
        recommended_songs = recommend_songs(user_preferences)

        st.subheader('Recommended Songs')
        st.write(recommended_songs)
        
    #User Interaction to choose their favourite songs
        st.subheader('Select your favorite songs from the recommendations:')
        selected_songs = st.multiselect('Favorite Songs', recommended_songs.index)
    #User Feedback
        if st.button('Submit Feedback'):
            if len(selected_songs) > 0:
                st.success('Thanks for providing feedback on your favorite songs!')
            else:
                st.warning('Please select at least one favorite song before submitting.')

if __name__ == '__main__':
    main()



import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import streamlit as st


df = pd.read_csv('spotify_tracks.csv')

df = df[['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','artists','album_name','track_name']]

df.dropna(inplace=True)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']])


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_scaled)


def main():
    st.title('Spotify Recommendation System')
    st.write('Input your preferences below and get song recommendations!')

    # Create sliders for user preferences
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
    

    if st.button('Get Recommendations'):
        # Preprocess user input
        user_input = scaler.transform([[danceability, energy, key, loudness, mode, speechiness,
       acousticness, instrumentalness, liveness, valence, tempo]])
        # Predict user's cluster using the K-means model
        user_cluster = kmeans.predict(user_input)
        # Get recommended songs based on the cluster
        recommended_songs = df[kmeans.labels_ == user_cluster[0]][['artists','album_name']]

        # Display the recommended songs
        st.subheader('Recommended Songs:')
        st.write(recommended_songs)

if __name__ == '__main__':
    main()


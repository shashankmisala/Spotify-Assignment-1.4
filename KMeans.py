# importing relevant libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import streamlit as st
from scipy import stats

#Loading the dataset
df = pd.read_csv('spotify_tracks.csv')


df = df[['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','artists','album_name','track_name']]

#Dropping missing values
df.dropna(inplace=True)

#Handling outliers using Z-score
def handle_outliers_zscore(data, threshold=3):
    z_scores = stats.zscore(data)
    return data[(z_scores > -threshold) & (z_scores < threshold)]
tracks = df[["danceability", "energy", "loudness","mode", "speechiness", "acousticness","instrumentalness","liveness","valence"]]
tracks = handle_outliers_zscore(tracks)

#Scaling the dataset
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['danceability','energy','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence']])


#KMeans Clustering
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(data_scaled)

#Making a user-friendly website
def main():
    st.title('Spotify Recommendation System With K-Means')
    st.write('Input your preferences below and get song recommendations!')

    # Create sliders for user preferences
    danceability = st.slider('Danceability', 0, 100, 25,10)
    energy = st.slider('Energy', 0, 100, 25,10)
    loudness = st.slider('Loudness', 0, 100, 25,10)
    mode = st.slider('Mode', 0, 100, 25,10)
    speechiness = st.slider('Speechiness', 0, 100, 25,10)
    acousticness = st.slider('Acousticness', 0, 100, 25,10)
    instrumentalness = st.slider('Instrumentalness', 0,100,25,10)
    liveness = st.slider('Liveness', 0, 100, 25,10)
    valence = st.slider('Valence', 0, 100, 25,10)
    

    if st.button('Get Recommendations'):
        # Preprocess user input
        user_input = scaler.transform([[danceability, energy, loudness, mode, speechiness,
       acousticness, instrumentalness, liveness, valence]])
        # Predict user's cluster using the K-means model
        user_cluster = kmeans.predict(user_input)
        # Get recommended songs based on the cluster
        recommended_songs = df[kmeans.labels_ == user_cluster[0]][['artists','album_name']]

        # Display the recommended songs
        st.subheader('Recommended Songs:')
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


import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🎵 Spotify Hit Predictor", layout="wide")
st.title("🎵 Spotify Hit Predictor")
st.write("Will your song be a HIT? Let's find out!")

# Load pre-trained model
with open('spotify_model.pkl', 'rb') as f:
    model = pickle.load(f)

features = ['danceability', 'energy', 'valence', 'tempo', 
            'acousticness', 'instrumentalness', 'speechiness', 'liveness']

st.subheader("🎸 Test Your Song")

col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5)
    valence = st.slider("😊 Valence (Happiness)", 0.0, 1.0, 0.5)
    tempo = st.slider("🎼 Tempo (BPM)", 50, 200, 120)

with col2:
    acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.5)
    speechiness = st.slider("🎤 Speechiness", 0.0, 1.0, 0.5)
    liveness = st.slider("🎤 Liveness", 0.0, 1.0, 0.5)

if st.button("🔮 Predict If This Will Be Popular!", use_container_width=True):
    input_data = pd.DataFrame([[danceability, energy, valence, tempo, 
                               acousticness, instrumentalness, speechiness, liveness]], 
                              columns=features)
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.balloons()
        st.success("🔥 THIS WILL BE A HIT! 🔥")
    else:
        st.warning("❌ This might not be popular")

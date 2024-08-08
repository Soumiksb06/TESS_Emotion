import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle

# Load the pre-trained model and encoder
model = load_model('TESS_latest_trained_model.keras')

with open('TESS_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Function to extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc, y, sr

# Function to plot waveform
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(f'Waveform of {emotion}', size=20)
    librosa.display.waveshow(data, sr=sr)
    st.pyplot(plt)

# Function to plot spectrogram
def spectrogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(f'Spectrogram of {emotion}', size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    st.pyplot(plt)

st.title('Emotion Recognition from Speech')
st.write("Upload an audio file to analyze its emotion")

# File upload
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Extract features and plot visualizations
    mfcc, y, sr = extract_mfcc(uploaded_file)
    waveplot(y, sr, 'Uploaded Audio')
    spectrogram(y, sr, 'Uploaded Audio')
    
    # Reshape features for prediction
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)

    # Predict emotion
    prediction = model.predict(mfcc)
    predicted_label = encoder.inverse_transform(prediction)
    st.write(f"The predicted emotion is: **{predicted_label[0][0].upper()}**")

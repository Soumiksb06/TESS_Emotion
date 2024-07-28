import streamlit as st
import librosa
import numpy as np
import base64
import pickle
from scipy.io.wavfile import write
import tensorflow as tf
import tensorflow.lite as tflite
import os
import plotly.graph_objects as go
import sounddevice as sd

# Create the uploaded_audio directory if it doesn't exist
if not os.path.exists("uploaded_audio"):
    os.makedirs("uploaded_audio")

# Convert Keras model to TensorFlow Lite if not already done
keras_model_path = "TESS_latest_trained_model.h5"  # Replace with your Keras model file path
tflite_model_path = "TESS_latest_trained_model.tflite"  # Provide the desired output path

if not os.path.exists(tflite_model_path):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tflite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Load the label encoder
with open("C:/Users/soumi/Downloads/TESS_encoder.pkl", 'rb') as f:
    enc = pickle.load(f)

# Extract MFCC features
def extract_mfcc(audio, sr):
    duration = 3  # adjust duration if necessary
    offset = 0.5  # adjust offset if necessary

    audio, _ = librosa.load(audio, sr=sr, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc

# Emotion prediction function
def predict_emotion(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)  # Load audio with original sample rate
    mfcc_features = extract_mfcc(audio, sr)
    mfcc_features = mfcc_features.astype(np.float32)
    
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], mfcc_features)
    interpreter.invoke()
    
    output_details = interpreter.get_output_details()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = enc.inverse_transform(prediction)
    predicted_emotion = predicted_label[0][0].upper()

    _, _, spectrogram = librosa.reassigned_spectrogram(audio, sr=sr)
    return predicted_emotion, spectrogram

# Streamlit UI
st.title("Emotion Prediction")
st.markdown("<h1 style='text-align: center; color: black;'>Emotion Prediction</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if st.button("Record Audio"):
    duration = 3  # seconds
    default_sr = sd.query_devices(None, 'input')['default_samplerate']
    st.write("Recording...")
    audio = sd.rec(int(duration * default_sr), samplerate=int(default_sr), channels=1)
    sd.wait()
    audio_path = "./uploaded_audio/recorded_audio.wav"
    write(audio_path, int(default_sr), audio)
    st.write("Recording finished.")
    emotion, spectrogram = predict_emotion(audio_path)
    st.write(f"The predicted emotion is: {emotion}")
    
    fig = go.Figure(data=go.Heatmap(z=spectrogram, colorscale='Hot'))
    fig.update_layout(title='Spectrogram', xaxis_title='Time', yaxis_title='Frequency')
    st.plotly_chart(fig)

if uploaded_file is not None:
    audio_path = os.path.join("uploaded_audio", uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    emotion, spectrogram = predict_emotion(audio_path)
    st.write(f"The predicted emotion is: {emotion}")
    
    fig = go.Figure(data=go.Heatmap(z=spectrogram, colorscale='Hot'))
    fig.update_layout(title='Spectrogram', xaxis_title='Time', yaxis_title='Frequency')
    st.plotly_chart(fig)

if __name__ == "__main__":
    st.run()

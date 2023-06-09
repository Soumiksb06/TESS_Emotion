import dash
from dash import dcc
from dash import html
import librosa
import numpy as np
import base64
import pickle
from scipy.io.wavfile import write
import tensorflow as tf
import tensorflow.lite as tflite
import os
import plotly.graph_objects as go

# Create the Dash app
app = dash.Dash(__name__, title='Toronto Speech Emotion Recognizer')

# Create the uploaded_audio directory if it doesn't exist
if not os.path.exists("uploaded_audio"):
    os.makedirs("uploaded_audio")

app.layout = html.Div(
    style={'backgroundColor': 'LightBlue', 'padding': '30px'},
    children=[
        html.H1(
            "Emotion Prediction",
            style={'textAlign': 'center', 'color': '333', 'fontFamily': 'Arial, sans-serif'}
        ),
        dcc.Upload(
            id='upload-audio',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Audio File')
            ]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'backgroundColor': 'Gold',
                'color': '333',
                'fontFamily': 'Arial, sans-serif'
            },
            multiple=False
        ),
        html.Button(
            'Record Audio',
            id='record-audio-button',
            style={'margin': '10px'}
        ),
        html.Div(
            id='output-prediction',
            style={'marginTop': '20px', 'fontFamily': 'Arial sans-serif', 'fontSize': '20px'}
        ),
        dcc.Graph(
            id='spectrogram-graph',
            style={'marginTop': '20px'}
        )
    ]
)


def extract_mfcc(audio, sr):
    duration = 3  # adjust duration if necessary
    offset = 0.5  # adjust offset if necessary

    audio, _ = librosa.load(audio, sr=sr, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc


# Convert Keras model to TensorFlow Lite
keras_model_path = "TESS_latest_trained_model.h5"  # Replace with your Keras model file path
model = tf.keras.models.load_model(keras_model_path)
converter = tflite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
tflite_model_path = "TESS_latest_trained_model.tflite"  # Provide the desired output path
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)


@app.callback(
    dash.dependencies.Output('output-prediction', 'children'),
    dash.dependencies.Output('spectrogram-graph', 'figure'),
    [dash.dependencies.Input('upload-audio', 'contents')],
    [dash.dependencies.State('upload-audio', 'filename')],
    [dash.dependencies.Input('record-audio-button', 'n_clicks')]
)
def predict_emotion(contents, filename, n_clicks):
    if contents is not None:
        content_type, content_string = contents.split(',')

        audio_path = f"./uploaded_audio/{filename}"
        with open(audio_path, 'wb') as f:
            f.write(base64.b64decode(content_string))

        audio, sr = librosa.load(audio_path, sr=None)  # Load audio with original sample rate
    elif n_clicks is not None and n_clicks > 0:
        # Recording parameters
        duration = 3  # adjust duration if necessary
        default_sr = sd.query_devices(None, 'input')['default_samplerate']
        channels = 1

        print("Recording started...")
        audio = sd.rec(int(duration * default_sr), samplerate=default_sr, channels=channels)
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # Save the recorded audio to a file
        audio_path = "./uploaded_audio/recorded_audio.wav"
        write(audio_path, default_sr, audio)

        sr = default_sr  # Use the default sample rate for MFCC extraction
    else:
        return '', {}

    # Perform emotion prediction
    mfcc_features = extract_mfcc(audio_path, sr)

    # Convert mfcc_features to float32
    mfcc_features = mfcc_features.astype(np.float32)

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Set the input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], mfcc_features)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Load the label encoder
    with open("C:/Users/soumi/Downloads/TESS_encoder.pkl", 'rb') as f:
        enc = pickle.load(f)

    predicted_label = enc.inverse_transform(prediction)
    predicted_emotion = predicted_label[0][0].upper()

    # Generate spectrogram
    _, _, spectrogram = librosa.reassigned_spectrogram(audio, sr=sr)

    fig = go.Figure(data=go.Heatmap(
        z=spectrogram,
        colorscale='Hot',
    ))

    fig.update_layout(
        title='Spectrogram',
        xaxis_title='Time',
        yaxis_title='Frequency',
    )

    return (
        html.H3(
            f"The predicted emotion is: {predicted_emotion}",
            style={'color': 'red', 'textAlign': 'center'}
        ),
        fig
    )


if __name__ == "__main__":
    app.run_server(debug=False, port=1540)

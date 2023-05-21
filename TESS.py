import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import random


paths = []
labels = []

# Collect paths and labels
for dirname, _, filenames in os.walk("F:/Docs/Data Science/Projects/TESS Toronto emotional speech set data"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())

# Combine paths and labels into a list of tuples
data = list(zip(paths, labels))

# Shuffle the data
random.shuffle(data)

# Unzip the shuffled data into paths and labels
paths, labels = zip(*data)

# Print the loaded dataset
print('Dataset is Loaded')


# In[59]:


paths[:5]


# In[60]:


labels[:5]


# In[3]:


## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()


# In[4]:


count=df['label'].value_counts()
count


# In[10]:


# Plot the bar chart
plt.bar(count.index, count.values)

# Set the labels and title
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.title('Count of Emotions')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Display the chart
plt.show()


# In[5]:


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveplot(data, sr=sr)
    plt.show()
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


# In[6]:


emotion = 'fear'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[7]:


emotion = 'angry'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[8]:


emotion = 'disgust'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[9]:


emotion = 'neutral'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[10]:


emotion = 'sad'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[11]:


emotion = 'ps'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[13]:


emotion = 'happy'
path = np.array(df['speech'][df['label']==emotion])[1]
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)


# In[14]:


# Extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# In[15]:


extract_mfcc(df['speech'][0])


# In[16]:


# Perform train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# In[17]:


X_train = np.array(train_df['speech'].apply(lambda x: extract_mfcc(x)).tolist())
X_test = np.array(test_df['speech'].apply(lambda x: extract_mfcc(x)).tolist())


# In[16]:


X_train = np.array([extract_mfcc(x) for x in train_df['speech']])
X_test = np.array([extract_mfcc(x) for x in test_df['speech']])


# In[18]:


# Reshape the input
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)


# In[19]:


# Encode the labels
enc = OneHotEncoder()
y_train = enc.fit_transform(train_df[['label']]).toarray()
y_test = enc.transform(test_df[['label']]).toarray()

import pickle
# Save the encoder
with open('TESS_encoder.pkl', 'wb') as f:
    pickle.dump(enc, f)


# In[20]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
# Define the model
model = Sequential([
    LSTM(123, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[21]:


# Train the model
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=512, shuffle=True, callbacks=[early_stopping])


# In[22]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# In[23]:


##Plot the results
acc = history.history['accuracy']
val_acc= history.history['val_accuracy']
epochs = list(range(len(acc)))  # Use the number of epochs as the x-axis values

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[24]:


print('Accuracy: %.2f%%' % (val_acc[-1] * 100))


# In[25]:


##Plot the results
loss = history.history['loss']
val_loss= history.history['val_loss']

plt.plot(epochs, loss, label= 'train loss')
plt.plot(epochs, val_loss, label= 'val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# # Test Results

# In[26]:


# Load the audio file
new_audio_path = "F:/Docs/Data Science/Projects/TESS Toronto emotional speech set data/YAF_disgust/YAF_back_disgust.wav"
new_audio, sr = librosa.load(new_audio_path, duration=3, offset=0.5)

# Extract MFCC features for the new audio file
new_mfcc = np.mean(librosa.feature.mfcc(y=new_audio, sr=sr, n_mfcc=40).T, axis=0)
new_mfcc = np.expand_dims(new_mfcc, axis=0)
new_mfcc = np.expand_dims(new_mfcc, axis=-1)

# Make a prediction using the trained model
prediction = model.predict(new_mfcc)
predicted_label = enc.inverse_transform(prediction)
print("The predicted emotion is:", predicted_label[0][0].upper())


# In[27]:



new_audio_path = "C:/Users/soumi/OneDrive/Desktop/test audio/OAF_bar_angry.wav"
new_audio, sr = librosa.load(new_audio_path, duration=3, offset=0.5)

# Extract MFCC features for the new audio file
new_mfcc = np.mean(librosa.feature.mfcc(y=new_audio, sr=sr, n_mfcc=40).T, axis=0)
new_mfcc = np.expand_dims(new_mfcc, axis=0)
new_mfcc = np.expand_dims(new_mfcc, axis=-1)

# Make a prediction using the trained model
prediction = model.predict(new_mfcc)
predicted_label = enc.inverse_transform(prediction)
print("The predicted emotion is:", predicted_label[0][0].upper())


# <font
# size="5">**New Data**</font>

# In[39]:


import sounddevice as sd
import numpy as np
import librosa

# Set the duration and sample rate
duration = 4
sample_rate = 22050

# Record audio
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
print("Recording audio...")

# Wait for recording to complete
sd.wait()

# Extract MFCC features for the recorded audio
audio_data = np.squeeze(recording)
mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T

# Reshape the audio to a vector of size 4
reshaped_audio = np.resize(mfcc_features, (4,))

# Make a prediction using the reshaped audio
reshaped_audio = np.expand_dims(reshaped_audio, axis=0)
reshaped_audio = np.expand_dims(reshaped_audio, axis=-1)
prediction = model.predict(reshaped_audio)
predicted_label = enc.inverse_transform(prediction)
print("The predicted emotion is:", predicted_label[0][0].upper())

# Play the recorded audio
print("Playing audio...")
sd.play(recording.flatten(), sample_rate)
sd.wait()


# In[40]:


saved=model.save('TESS_latest_trained_model.h5')


# In[41]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import librosa
import numpy as np
import base64
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
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


def extract_mfcc(audio):
    sr = 22050  # adjust sample rate if necessary
    duration = 3  # adjust duration if necessary
    offset = 0.5  # adjust offset if necessary

    audio, _ = librosa.load(audio, sr=sr, duration=duration, offset=offset)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc


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

        audio, sr = librosa.load(audio_path)
    elif n_clicks is not None and n_clicks > 0:
        # Recording parameters
        duration = 3  # adjust duration if necessary
        fs = 22050  # adjust sample rate if necessary

        print("Recording started...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # Save the recorded audio to a file
        audio_path = "./uploaded_audio/recorded_audio.wav"
        write(audio_path, fs, audio)

        # Set the sample rate for MFCC extraction
        sr = fs
    else:
        return '', {}

    # Perform emotion prediction
    mfcc_features = extract_mfcc(audio_path)
    prediction = model.predict(mfcc_features)
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
    # Load the trained model and encoder
    model = load_model("C:/Users/soumi/Downloads/TESS_latest_trained_model.h5")  # Replace with your trained model file path

    with open("C:/Users/soumi/Downloads/TESS_encoder.pkl", 'rb') as f:
        enc = pickle.load(f)

    app.run_server(debug=False, port=1540)

from flask import Flask, render_template, request, redirect
from pydub import AudioSegment
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

app = Flask(__name__)

model = joblib.load("finalized_model.sav")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def record():
    # get the recorded audio data from the POST request
    recorded_audio = request.data

    # convert the recorded audio data to an AudioSegment object
    audio_segment = AudioSegment.from_file(io.BytesIO(recorded_audio), format="wav")

    # export the AudioSegment object as an MP3 file
    audio_file_path = 'recorded_audio.mp3'
    audio_segment.export(audio_file_path, format='mp3')

    #Pretty stuff --> make these show up on webpage
    # waveform, sample_rate = torchaudio.load(PATHTOMP3FILE) #rarrrrrrrrh
    #
    # print_stats(waveform, sample_rate=sample_rate)
    # plot_waveform(waveform, sample_rate)
    # plot_specgram(waveform, sample_rate)
    # play_audio(waveform, sample_rate)

    # Extract features from MP3 file
    features = list()
    audio, _ = librosa.load(path, sr=48000)
    mfcc = librosa.feature.mfcc(y=audio, sr=48000)
    for el in mfcc:
        features.append(np.mean(el))

    # Standardise features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(np.array(features).reshape(-1, 1))

    # Choose 15 best features
    selected_features = np.load('selected_features.npy')  # saved from when we trained the model, DONT FORGET TO UPLOAD
    df_features = []
    for i in range(len(selected_features)):
        if selected_features[i]:
            df_features.append(scaled_features[i])

    df_features = np.concatenate(df_features).reshape(1, -1)

    # Make model do prediction
    prediction = model.predict(df_features) #will return something of form array([1]) = male (i think, tests dont seem to work good with females)

    #translate predicition into female/male/other
    if output == 0:
        return render_template('index.html',
                               prediction_text="Predicted Price is negative, values entered not reasonable")
    elif output == 1:
        return render_template('index.html', prediction_text='Predicted Price of the house is: ${}'.format(output))
    elif output == 0.5:
        return render_template()

    # redirect to the index page
    return redirect('/results')

@app.route('/results')
def results():
    return render_template("results.html")

if __name__ == '__main__':
    app.run(debug=True)

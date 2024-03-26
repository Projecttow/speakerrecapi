from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import librosa
import numpy as np
import os
app = Flask(__name__)



def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}", e)
        return None 
    return mfccs_processed

@app.route('/predict', methods=['POST'])
def predict_audio():
    speaker_name = request.form['model_name']
    
    model = load_model(f'{speaker_name}_model.h5')
    scaler = joblib.load(f'{speaker_name}_scaler.save')
    file = request.files['audio_file']
    file_path = "temp_file.wav"
    file.save(file_path)
    
    mfccs = extract_features(file_path)
    if mfccs is None:
        return jsonify({"error": "Could not extract features."})
    
    mfccs = mfccs.reshape(1, -1)
    mfccs_scaled = scaler.transform(mfccs)
    prediction = model.predict(mfccs_scaled)
    
    os.remove(file_path)  
    
    result = "Cloned Voice" if prediction[0][0] > 0.5 else "Real Voice"
    return jsonify({"prediction": result})


if __name__ == '__main__':
    app.run(debug=True)

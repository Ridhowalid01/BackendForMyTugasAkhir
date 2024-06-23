from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import librosa
import numpy as np
from fastdtw import fastdtw
import noisereduce as nr
from scipy.spatial.distance import cosine
from module import remove_silence, preprocessing, extraction, calculate_dtw, scoring, main

app = Flask(__name__)
CORS(app)

@app.after_request
def after_request(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

@app.route('/score', methods=['POST'])
def score_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    index_input = int(request.form.get('index', 0))

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to a temporary location
    temp_file_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(temp_file_path)

    # Calculate the score
    try:
        score = main(temp_file_path, index_input)
        os.remove(temp_file_path)  # Remove the temporary file
        return str(score)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)  # Ensure temp file is removed in case of error
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import tensorflow as tf
import io
import os
import neurokit2 as nk
from scipy.signal import find_peaks
from biosppy.signals import ecg
from ecg_image_processor import process_ecg_image

app = Flask(__name__)

# Load the trained model and label encoder
model = tf.keras.models.load_model('../models/ecg_multiclass_cnn.keras')
le = joblib.load('../models/label_encoder.pkl')

CONDITION_DESCRIPTIONS = {
    'N': 'Normal heartbeat — no arrhythmia detected',
    'V': 'Premature Ventricular Contraction (PVC) — early heartbeat from ventricles',
    'A': 'Atrial Premature Beat — early heartbeat from atria',
    'L': 'Left Bundle Branch Block — electrical conduction delay',
    'R': 'Right Bundle Branch Block — electrical conduction delay'
}

def extract_beats_manually(signal):
    """Manual beat extraction using scipy find_peaks as last resort"""
    peaks, _ = find_peaks(signal, height=np.mean(signal) + 0.3 * np.std(signal), distance=30)
    beats = []
    for p in peaks:
        if p - 50 >= 0 and p + 50 <= len(signal):
            beats.append(signal[p-50:p+50])
    return np.array(beats) if beats else np.array([])

def run_prediction(signal):
    if len(signal) < 100:
        raise ValueError("Signal too short")

    templates = []

    # Method 1: biosppy
    try:
        out = ecg.ecg(signal=signal, sampling_rate=360, show=False)
        templates = out['templates']
    except Exception:
        pass

    # Method 2: neurokit2
    if len(templates) < 2:
        try:
            signals_df, info = nk.ecg_process(signal, sampling_rate=360)
            peaks = info['ECG_R_Peaks']
            peaks = [p for p in peaks if not np.isnan(p)]
            peaks = [int(p) for p in peaks]
            templates = []
            for p in peaks:
                if p - 50 >= 0 and p + 50 <= len(signal):
                    templates.append(signal[p-50:p+50])
            templates = np.array(templates)
        except Exception:
            pass

    # Method 3: manual scipy peak detection
    if len(templates) < 2:
        templates = extract_beats_manually(signal)

    if len(templates) == 0:
        raise ValueError("No beats detected — please upload a clearer ECG image")

    # Predict
    beats = np.array(templates)
    if len(beats.shape) == 1:
        beats = beats.reshape(1, -1)
    if beats.shape[1] != 100:
        beats = np.array([b[:100] if len(b) >= 100 else np.pad(b, (0, 100-len(b))) for b in beats])

    beats_cnn = beats.reshape(beats.shape[0], beats.shape[1], 1)
    predictions = model.predict(beats_cnn)
    pred_classes = np.argmax(predictions, axis=1)
    pred_labels = le.inverse_transform(pred_classes)

    from collections import Counter
    counts = Counter(pred_labels)
    total = len(pred_labels)

    results = []
    for cls, count in counts.most_common():
        results.append({
            'condition': cls,
            'description': CONDITION_DESCRIPTIONS[cls],
            'beats': int(count),
            'percentage': round(count / total * 100, 1)
        })

    return {'total_beats': total, 'results': results, 'dominant': results[0]['condition']}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename.lower()
    file_bytes = file.read()

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            signal = process_ecg_image(file_bytes)
            source = 'image'
        elif filename.endswith(('.csv', '.txt', '.dat')):
            content = file_bytes.decode('utf-8')
            signal = np.loadtxt(io.StringIO(content), delimiter=',', dtype=np.float32)
            source = 'signal'
        else:
            return jsonify({'error': 'Unsupported file type. Upload .png, .jpg, .csv, or .dat'}), 400

        result = run_prediction(signal)
        result['source'] = source
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
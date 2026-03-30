from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import tensorflow as tf
import io
import os
from biosppy.signals import ecg

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        content = file.read().decode('utf-8')
        data = np.loadtxt(io.StringIO(content), delimiter=',', dtype=np.float32)
    except Exception as e:
        return jsonify({'error': f'Could not read file: {str(e)}'}), 400

    if len(data) < 100:
        return jsonify({'error': 'Signal too short'}), 400

    # Process with biosppy
    try:
        out = ecg.ecg(signal=data, sampling_rate=360, show=False)
        templates = out['templates']
    except Exception as e:
        return jsonify({'error': f'Could not process ECG signal: {str(e)}'}), 400

    if len(templates) == 0:
        return jsonify({'error': 'No beats detected'}), 400

    # Predict each beat
    beats = np.array(templates)
    if beats.shape[1] != 100:
        beats = np.array([b[:100] if len(b) >= 100 else np.pad(b, (0, 100-len(b))) for b in beats])
    
    beats_cnn = beats.reshape(beats.shape[0], beats.shape[1], 1)
    predictions = model.predict(beats_cnn)
    pred_classes = np.argmax(predictions, axis=1)
    pred_labels = le.inverse_transform(pred_classes)

    # Summarize results
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

    return jsonify({
        'total_beats': total,
        'results': results,
        'dominant': results[0]['condition']
    })

if __name__ == '__main__':
    app.run(debug=True)
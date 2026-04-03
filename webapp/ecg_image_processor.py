import cv2
import numpy as np
from scipy.signal import resample, medfilt, find_peaks

def extract_signal_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Convert to float for processing
    gray_float = gray.astype(np.float32)

    # Enhance contrast
    gray_eq = cv2.equalizeHist(gray)

    # Multiple thresholding attempts to find the best one
    best_signal = None
    best_score = -1

    for thresh_val in [60, 80, 100, 120, 140]:
        _, binary = cv2.threshold(gray_eq, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Extract signal
        signal = []
        last_valid = height // 2
        
        for x in range(width):
            col = cleaned[:, x]
            white_pixels = np.where(col > 0)[0]
            if len(white_pixels) > 0:
                y = int(np.median(white_pixels))
                last_valid = y
            else:
                y = last_valid
            signal.append(y)
        
        signal = np.array(signal, dtype=np.float32)
        signal = (height - signal) / height
        signal = medfilt(signal, kernel_size=3)
        
        # Score this signal by how many peaks it has
        normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        peaks, _ = find_peaks(normalized, height=0.3, distance=20)
        
        if len(peaks) > best_score:
            best_score = len(peaks)
            best_signal = signal

    if best_signal is None or best_score < 1:
        raise ValueError("Could not extract a valid ECG signal from image")

    signal = best_signal

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Resample to 3600 samples
    signal = resample(signal, 3600).astype(np.float32)

    # Final normalization
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    return signal


def process_ecg_image(image_bytes):
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        signal = extract_signal_from_image(tmp_path)
    finally:
        os.unlink(tmp_path)

    return signal
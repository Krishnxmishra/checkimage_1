# sudoku_marker_detector.py
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import math

app = Flask(__name__)

# --- DCT helpers ---
def dct_matrix(n=8):
    M = np.zeros((n, n), dtype=float)
    for k in range(n):
        for i in range(n):
            M[k, i] = math.cos(math.pi * k * (2*i + 1) / (2*n))
    M[0,:] *= 1 / math.sqrt(2)
    M *= math.sqrt(2.0 / n)
    return M

DCT8 = dct_matrix(8)
def dct2(block):
    return DCT8 @ block @ DCT8.T

# --- Detection params ---
MID_POS = [(2+i, 2+j) for i in range(3) for j in range(3)]
RANK_TOLERANCE = 0.1  # relative difference threshold for rank check

def has_sudoku_pattern(block):
    coeffs = dct2(block)
    vals = np.array([coeffs[r, c] for (r, c) in MID_POS], dtype=float)
    ranks = np.argsort(np.argsort(vals)) + 1  # 1..9
    # Check if all ranks are distinct
    if len(set(ranks)) != 9:
        return False
    # Check minimum spacing between ranks relative to coefficient std
    std = np.std(vals)
    if std < 1e-3:
        return False
    diffs = np.diff(np.sort(vals))
    if np.any(diffs / std < RANK_TOLERANCE):
        return False
    return True

def detect_sudoku_marker(img):
    img_gray = img.convert("L")
    arr = np.array(img_gray, dtype=float)
    H, W = arr.shape
    # iterate over 8x8 blocks
    for by in range(0, H-7, 8):
        for bx in range(0, W-7, 8):
            block = arr[by:by+8, bx:bx+8]
            if has_sudoku_pattern(block):
                return True
    return False

# --- Flask route ---
@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['image']
    try:
        img = Image.open(file)
    except Exception as e:
        return jsonify({"error": "Cannot open image"}), 400
    detected = detect_sudoku_marker(img)
    return jsonify({"detected": detected})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

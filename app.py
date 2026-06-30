"""Webcam emotion detection backend.

Frontend sends a webcam frame, DeepFace tells us the emotion. That's it.
"""
import base64

import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")
    if "," in image_data:  # strip "data:image/jpeg;base64," prefix if present
        image_data = image_data.split(",", 1)[1]
    if not image_data:
        return jsonify({"error": "no image"}), 400

    # base64 -> bytes -> OpenCV image
    img_bytes = base64.b64decode(image_data)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "bad image"}), 400

    # enforce_detection=False so an empty/no-face frame returns neutral instead of 500
    result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
    if isinstance(result, list):  # newer DeepFace returns a list of faces
        result = result[0]

    scores = {k: round(float(v), 2) for k, v in result["emotion"].items()}
    return jsonify({"emotion": result["dominant_emotion"], "scores": scores})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

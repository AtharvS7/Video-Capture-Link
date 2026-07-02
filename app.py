"""Webcam emotion detection backend.

Frontend sends a webcam frame. We find the face ourselves (Haar cascade),
crop it, and hand just the face to DeepFace. Cropping to the face makes the
emotion model much more accurate than feeding it the whole frame, and it lets
us honestly report "no face" instead of guessing on an empty image.
"""
import base64

import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# bundled with opencv, no extra download
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # normalize lighting -> steadier detection
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # biggest by area


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")
    if "," in image_data:  # strip "data:image/jpeg;base64," prefix
        image_data = image_data.split(",", 1)[1]
    if not image_data:
        return jsonify({"error": "no image"}), 400

    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR
    )
    if img is None:
        return jsonify({"error": "bad image"}), 400

    h, w = img.shape[:2]
    box = largest_face(img)
    if box is None:
        return jsonify({"face": False})

    # square crop with ~30% margin, centred on the face. Square matters: the
    # emotion model resizes to 48x48, so a non-square crop squishes features.
    x, y, fw, fh = box
    cx, cy = x + fw / 2, y + fh / 2
    half = int(max(fw, fh) * 0.65)
    x0, y0 = max(int(cx - half), 0), max(int(cy - half), 0)
    x1, y1 = min(int(cx + half), w), min(int(cy + half), h)
    face = img[y0:y1, x0:x1]

    result = DeepFace.analyze(
        face, actions=["emotion"], enforce_detection=False, detector_backend="skip"
    )
    if isinstance(result, list):
        result = result[0]

    scores = {k: round(float(v), 2) for k, v in result["emotion"].items()}
    return jsonify(
        {
            "face": True,
            "emotion": result["dominant_emotion"],
            "scores": scores,
            # normalized (0..1) so the frontend can scale it to any display size
            "box": {"x": x0 / w, "y": y0 / h, "w": (x1 - x0) / w, "h": (y1 - y0) / h},
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

# Video Capture — Emotion Detector 🎭

A small web app that uses your webcam to detect facial emotions (happy, sad,
angry, surprise, fear, disgust, neutral) in real time.

The browser captures webcam frames and sends them to a Flask backend, which
runs them through [DeepFace](https://github.com/serengil/deepface) (a
pre-trained emotion model) and sends the result back to the page.

## How it works

```
Browser (getUserMedia) ──frame (base64)──▶ Flask /predict ──▶ DeepFace
        ▲                                                         │
        └──────────────── emotion + scores (JSON) ◀──────────────┘
```

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

> The first run downloads the DeepFace emotion model (~5 MB) automatically.

## Run

```bash
python app.py
```

Open <http://127.0.0.1:5000>, allow camera access, and make some faces.
The detected emotion updates about once per second.

## Files

| File                  | Purpose                                  |
| --------------------- | ---------------------------------------- |
| `app.py`              | Flask backend + DeepFace emotion analysis |
| `templates/index.html`| Minimal frontend (webcam + display)      |
| `test_predict.py`     | Quick self-check for the `/predict` route |

## Quick test (no webcam needed)

With the server running in one terminal:

```bash
python test_predict.py
```

It posts a generated frame and checks the response shape.

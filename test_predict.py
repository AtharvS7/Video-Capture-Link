"""Self-check for the /predict endpoint.

Run `python app.py` in another terminal first, then `python test_predict.py`.
Posts a generated frame and asserts the response looks right.
"""
import base64

import cv2
import numpy as np
import requests

URL = "http://127.0.0.1:5000/predict"
VALID = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}


def make_frame_b64():
    # plain gray frame is enough to exercise the endpoint (no real face needed)
    img = np.full((240, 320, 3), 127, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def main():
    res = requests.post(URL, json={"image": make_frame_b64()}, timeout=60)
    res.raise_for_status()
    data = res.json()

    assert "emotion" in data, f"no emotion in {data}"
    assert data["emotion"] in VALID, f"unexpected emotion: {data['emotion']}"
    assert "scores" in data and isinstance(data["scores"], dict), "missing scores"
    assert abs(sum(data["scores"].values()) - 100) < 5, "scores should sum to ~100"

    print("OK ->", data["emotion"], data["scores"])


if __name__ == "__main__":
    main()

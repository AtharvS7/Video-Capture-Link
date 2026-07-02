"""Self-check for the /predict endpoint.

Run `python app.py` in another terminal first, then `python test_predict.py`.
A blank frame must report no face (this exercises decode -> cascade -> JSON).
The real emotion path is verified live in the browser with a webcam.
"""
import base64

import cv2
import numpy as np
import requests

URL = "http://127.0.0.1:5000/predict"


def main():
    blank = np.full((360, 480, 3), 127, np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    assert ok
    image = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    res = requests.post(URL, json={"image": image}, timeout=60)
    res.raise_for_status()
    data = res.json()

    assert data.get("face") is False, f"blank frame should report no face: {data}"
    print("OK  blank frame -> no face:", data)


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__, static_folder="static")

model = YOLO("yolo26n.pt")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    print("1. DETECT REQUEST RECEIVED", flush=True)

    try:
        if "frame" not in request.files:
            print("2. NO FRAME", flush=True)
            return jsonify({"error": "No frame received"}), 400

        file = request.files["frame"]
        image_bytes = file.read()

        print("2. FRAME SIZE:", len(image_bytes), flush=True)

        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("3. INVALID FRAME", flush=True)
            return jsonify({"error": "Invalid image"}), 400

        print("3. FRAME SHAPE:", frame.shape, flush=True)
        print("4. STARTING YOLO...", flush=True)

        results = model(frame, verbose=False)

        print("5. YOLO FINISHED", flush=True)

        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": model.names[class_id],
                "confidence": confidence,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            })

        print(
            "6. DETECTIONS:",
            len(detections),
            flush=True
        )

        return jsonify({
            "detections": detections,
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0])
        })

    except Exception as e:
        print("ERROR:", repr(e), flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__, static_folder="static")

# Load pretrained model
model = YOLO("yolo26n.pt")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "frame" not in request.files:
            return jsonify({
                "error": "No frame received"
            }), 400

        file = request.files["frame"]

        # Convert uploaded JPEG bytes to OpenCV frame
        image_bytes = file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "error": "Could not decode frame"
            }), 400

        # YOLO detection
        results = model(frame, verbose=False)

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

        return jsonify({
            "detections": detections,
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0])
        })

    except Exception as e:
        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,
        
    )
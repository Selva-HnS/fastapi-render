# Directory structure:
# functions/
# ├── main.py
# ├── requirements.txt
# └── model/
#     ├── model.pt
#     └── labels.txt
#
# -----------------------------
# functions/main.py
# -----------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("model/model.pt")

# Load label list
def load_labels(path="model/labels.txt"):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    try:
        results = model(image)
        predictions = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
                predictions.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "box": [round(x, 2) for x in xyxy]
                })
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Realtime detection generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(image)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls.item())
                label = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
                conf = box.conf.item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head><title>YOLO Live Camera</title></head>
    <body>
        <h1>Realtime Detection</h1>
        <img src="/video_feed" width="720" />
    </body>
    </html>
    """
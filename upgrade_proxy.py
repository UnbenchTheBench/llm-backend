from flask_cors import CORS
import requests
import json
from flask import Flask, request, jsonify, Response
from PIL import Image
import io
from ultralytics import YOLO
import base64

app = Flask(__name__)
CORS(app)

url = "http://localhost:11434/api/chat"

def generate(payload):
        with requests.post(url, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    # Each line is a chunk of JSON from Ollama
                    try:
                        data = json.loads(line)
                        # Extract partial text from data structure (depends on Ollama format)
                        partial_text = data.get('response', '')
                        yield partial_text
                    except Exception:
                        # If not JSON or error, just yield raw line
                        yield line.decode('utf-8')

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.json
    payload['stream'] = True  # enable streaming on Ollama side

    

    return Response(generate(payload), mimetype='text/plain')

model = YOLO("model_current.pt")

def validate_and_visualize(model: YOLO, pil_image: Image.Image, conf: float = 0.4):
    import numpy as np
    import cv2

    # Convert PIL Image to OpenCV format
    img = np.array(pil_image)
    img_bgr = img[:, :, ::-1].copy()  # RGB to BGR

    # Run YOLO model prediction
    results = model.predict(img_bgr, conf=conf)[0]
    class_ids = results.boxes.cls
    if class_ids is None or len(class_ids) == 0:
        return [], pil_image 

    detections = []
    for cls_id, box, score in zip(results.boxes.cls, results.boxes.xyxy, results.boxes.conf):
        name = results.names[int(cls_id)]
        confidence = float(score)
        x1, y1, x2, y2 = [int(coord) for coord in box]
        color = (0, 0, 0)

        if name == "Horseradish":
            color = (0, 255, 0)
        elif name == "Weed":
            color = (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color=color, thickness=5)
        label = f"{name} {confidence:.2f}"
        cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)

        # Save detection info
        detections.append({
            "name": name,
            "confidence": confidence,
            "bbox": [float(coord) for coord in box]
        })

    # Convert image back to RGB and then to PIL Image
    img_annotated = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    

    return detections, img_annotated

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get detections and annotated image
        detections, annotated_image = validate_and_visualize(model, pil_image, conf=0.4)

        # Convert annotated image to base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "detections": detections,
            "image": encoded_image  # base64-encoded image
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    

def initialize_model():

    model = "llava"

    file = "prompt.txt"
    payload = {"model": "llava","stream": False}

    print("Loading prompt from file:", file)

    with open(file, "r", encoding="utf-8") as f:
        payload["prompt"] = f.read()

    print("Sending initialization request to Ollama with payload:", payload)
        
    response = requests.post(url, json=payload)

    print(response.json()["response"])
    


if __name__ == '__main__':
    #initialize_model()
    app.run(host='0.0.0.0', port=5000)
    

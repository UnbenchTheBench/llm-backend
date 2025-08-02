import requests
import base64

# Load image and encode it in base64
with open("other.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Define Ollama API URL
url = "http://localhost:11434/api/generate"

# Payload for LLaVA
payload = {
    "model": "llava",
    "prompt": "Describe this image",
    "images": [encoded_image],
    "stream": False
}

# Send POST request
response = requests.post(url, json=payload)

# Output result
print(response.json()["response"])

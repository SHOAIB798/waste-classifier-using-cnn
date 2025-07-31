import os
import re
import requests
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from googleapiclient.discovery import build
import io

app = Flask(__name__)

# Load the trained model
model = load_model('waste_classifier_model.h5')

# Set up the Google Custom Search API
API_KEY = ''  # Replace with your Google API Key
CSE_ID = ''  # Replace with your Custom Search Engine ID

# Ensure the 'static/images' directory exists
if not os.path.exists('static/images'):
    os.makedirs('static/images')

# Function to fetch images using Google Custom Search API
def fetch_images(query, num_results=3):
    service = build("customsearch", "v1", developerKey=API_KEY)
    res = service.cse().list(q=query, cx=CSE_ID, searchType='image', num=num_results).execute()
    image_urls = []
    if 'items' in res:
        for item in res['items']:
            image_urls.append(item['link'])
    return image_urls

# Function to sanitize image file names
def sanitize_filename(filename):
    # Replace any character that is not a letter, number, or underscore with an underscore
    return re.sub(r'[^a-zA-Z0-9_]', '_', filename)

# Function to get the correct file extension from the image URL
def get_file_extension(url):
    valid_extensions = ['jpg', 'jpeg', 'png', 'gif']
    # Try to infer the file extension from the URL
    for ext in valid_extensions:
        if ext in url:
            return ext
    return 'jpg'  # Default to 'jpg' if extension cannot be found

# Function to preprocess and predict a single image
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))  # Resize to the size expected by the model
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image pixels to [0, 1]

    prediction = model.predict(img_array)
    result = 'Recycle' if prediction > 0.5 else 'Organic'
    return result, prediction[0][0]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route for handling the image prediction
@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['query']
    image_urls = fetch_images(query)  # Get images based on the query
    if image_urls:
        # Download the first image for prediction
        img_url = image_urls[0]
        img_data = requests.get(img_url).content
        img = Image.open(io.BytesIO(img_data))

        # Get the correct file extension
        file_extension = get_file_extension(img_url)
        
        # Sanitize the image file name and save it with the correct extension
        sanitized_filename = sanitize_filename('uploaded_image') + '.' + file_extension
        img_path = os.path.join('static', 'images', sanitized_filename)
        img.save(img_path)

        # Predict the class of the fetched image
        result, prediction = predict_image(img_path)

        return render_template('index.html', query=query, image_url=img_url, result=result, score=prediction)
    else:
        return jsonify({"error": "No images found for this query."})

if __name__ == '__main__':
    app.run(debug=True)




import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify,render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('C:\\Users\\sri kumar\\Desktop\\MAJOR  PROJECT\\Flask\\model.h5')
# Define the class names
class_names = ('dirty', 'clean')
image = ""
# Define the image size
img_size = (128, 128, 3)
@app.route('/')
@app.route('/predict', methods=['POST','GET'])
def predict():
    # Get the image file from the reques
    if 'image' not in request.files:
        return render_template("index.html")
    else : 
        #return render_template("index.html")
        image_file = request.files['image']
    
        
        
        # Read and preprocess the image
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, img_size[0:2])[:, :, ::-1]
        processed_image = np.asarray(resized_image) / 255.0
        
        # Make the prediction
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        predicted_class = class_names[np.argmax(prediction)]
        
        # Return the result as JSON
        result = {'prediction': predicted_class}
        return jsonify(result)
        #    return render_template("index.html",prediction=result)
if __name__ == '__main__':
    app.run(debug=True)


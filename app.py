from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load the model
model = load_model('project4/pneumo_detect.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = image.load_img(img, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_path = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = str(uuid.uuid4()) + file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            if prediction[0][0] > prediction[0][1]:
                prediction = "Normal"
            else:
                prediction = "Pneumonia"
            image_path = file_path
    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)

import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model("../models/kidney_model.h5")

IMG_SIZE = 128

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Stone Detected"
        confidence = prediction * 100
    else:
        label = "No Stone Detected"
        confidence = (1 - prediction) * 100

    return label, round(confidence, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)

            # Simple risk logic
            if label == "Stone Detected":
                if confidence > 90:
                    risk = "High Risk"
                elif confidence > 70:
                    risk = "Moderate Risk"
                else:
                    risk = "Low Risk"
            else:
                risk = "No Risk"

            return render_template(
                "result.html",
                label=label,
                confidence=confidence,
                risk=risk,
                image_path=filepath
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
import sys
sys.path.insert(0, r"C:\Users\Tishya\tf220")

from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

IMAGE_SIZE = 180

MODEL_PATH = "xray_model.keras"

CLASS_NAMES = [
    "Cardiomegaly",
    "Normal",
    "Other Abnormality",
    "Pneumonia",
    "Pneumothorax",
    "Tuberculosis (TB)"
]

model = tf.keras.models.load_model(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():

    results = None
    prediction = None
    score = None

    if request.method == "POST":

        file = request.files["xray"]

        if file:

            # Save uploaded image temporarily
            image_path = "uploaded_xray.png"
            file.save(image_path)

            # Read image as grayscale
            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:

                # Resize
                image = cv2.resize(
                    image,
                    (IMAGE_SIZE, IMAGE_SIZE)
                )

                # IMPORTANT:
                # Do NOT divide by 255.
                # The model already has Rescaling.
                image = image.astype("float32")

                # Correct model shape
                image = image.reshape(
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    1
                )

                # Prediction
                prediction_values = model.predict(
                    image,
                    verbose=0
                )[0]

                predicted_index = np.argmax(
                    prediction_values
                )

                prediction = CLASS_NAMES[
                    predicted_index
                ]

                score = (
                    prediction_values[
                        predicted_index
                    ] * 100
                )

                results = []

                for i in range(6):

                    results.append({
                        "name": CLASS_NAMES[i],
                        "percentage": round(
                            prediction_values[i] * 100,
                            2
                        )
                    })

            # Delete temporary image
            if os.path.exists(image_path):
                os.remove(image_path)

    return render_template(
        "index.html",
        results=results,
        prediction=prediction,
        score=score
    )


if __name__ == "__main__":
    app.run(debug=True)
